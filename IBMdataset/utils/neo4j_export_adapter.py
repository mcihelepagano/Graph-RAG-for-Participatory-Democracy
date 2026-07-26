"""
neo4j_export_adapter.py
=======================
Drop-in replacement for the Neo4j connection used by sweep scripts.

Loads the JSON export once and exposes the same function signatures as
the Cypher queries in sweep_uc1_k.py / sweep_uc2_k.py.

USAGE in sweep scripts:

    # OLD: connects to Neo4j
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    ...
    pros_t, pros_v, cons_t, cons_v = fetch_constrained_pool(driver, policy)

    # NEW: load from JSON export
    from neo4j_export_adapter import load_export, fetch_constrained_pool, fetch_system_b_pool
    driver = load_export("data/neo4j_export_with_new_edges.json")
    ...
    pros_t, pros_v, cons_t, cons_v = fetch_constrained_pool(driver, policy)

The `driver` is now a `GraphHandle` object that holds the loaded data
in memory. It supports the same `.close()` method as a Neo4j driver
(no-op) so existing teardown code works.

EDGE-KEY AUTO-DETECTION
    Exports have carried two endpoint-key conventions over time:
    older edge dumps used {"arg1_id", "arg2_id"}, while edges written
    by generate_relations.py use {"source", "target"}. Schema
    mismatches here have caused silent failures before, so GraphHandle
    now detects the convention PER EDGE at load time, exposes the
    majority convention as `driver._edge_src` / `driver._edge_dst`,
    and refuses to load (loudly) if any edge has neither key pair.

Functions provided (matching the sweep script signatures exactly):
  load_export(path) -> GraphHandle
  fetch_constrained_pool(driver, policy_name)
      -> (pro_texts, pro_vecs, con_texts, con_vecs)
  fetch_system_b_pool(driver, policy_name)
      -> (pro_texts, pro_vecs, pro_pageranks,
          con_texts, con_vecs, con_pageranks)
  fetch_system_b_pool_with_ids(driver, policy_name)
      -> (pro_ids, pro_texts, pro_vecs, pro_pageranks,
          con_ids, con_texts, con_vecs, con_pageranks)
      Same pool as fetch_system_b_pool BY CONSTRUCTION (the latter is a
      thin wrapper that drops the ids), with argument ids included for
      callers that need per-pool argument identity on top of the
      identical candidate set.
  fetch_full_corpus(driver)
      -> (texts, vecs, policies)   # all arguments, for vanilla-RAG use
"""

import json
import os
from collections import defaultdict
from typing import Optional


# ============================================================
# EDGE-KEY DETECTION
# ============================================================
def _edge_endpoints(e: dict):
    """Return (src, dst) for an edge dict under EITHER key convention,
    or (None, None) if neither pair is present."""
    a, b = e.get("source"), e.get("target")
    if a is not None and b is not None:
        return a, b
    a, b = e.get("arg1_id"), e.get("arg2_id")
    if a is not None and b is not None:
        return a, b
    return None, None


# ============================================================
# DATA HANDLE
# ============================================================
class GraphHandle:
    """In-memory replacement for a Neo4j driver.

    Holds the loaded export and pre-built indices for fast lookup.
    Mirrors the .close() / .session() API surface enough that existing
    sweep code requires no changes beyond the import + load call.
    """

    def __init__(self, export: dict):
        self.arguments = export["arguments"]
        self.policies  = export["policies"]
        self.edges     = export.get("edges", {})
        self.meta      = export.get("meta", {})

        # ---- detect the endpoint-key convention ----
        # Count both conventions across all edge lists; expose the
        # majority as _edge_src/_edge_dst, but ALWAYS read edges per-edge
        # via _edge_endpoints so mixed-convention exports still work.
        n_st, n_aa, n_bad = 0, 0, 0
        for rel_type, edge_list in self.edges.items():
            for e in edge_list:
                if e.get("source") is not None and e.get("target") is not None:
                    n_st += 1
                elif e.get("arg1_id") is not None and e.get("arg2_id") is not None:
                    n_aa += 1
                else:
                    n_bad += 1
        if n_bad:
            raise ValueError(
                f"{n_bad} edges have neither (source,target) nor "
                f"(arg1_id,arg2_id) endpoint keys — export is malformed. "
                f"Refusing to load rather than silently dropping edges."
            )
        if n_aa > n_st:
            self._edge_src, self._edge_dst = "arg1_id", "arg2_id"
        else:
            self._edge_src, self._edge_dst = "source", "target"
        if n_st and n_aa:
            print(f"  NOTE: mixed edge-key conventions in export "
                  f"(source/target: {n_st}, arg1_id/arg2_id: {n_aa}) — "
                  f"handled per edge.")

        # Build edge index for fast multi-hop queries.
        # contradicts_by_arg[arg_id] = [other_arg_id, ...]
        self.contradicts_by_arg = defaultdict(list)
        for e in self.edges.get("contradicts", []):
            a, b = _edge_endpoints(e)
            self.contradicts_by_arg[a].append(b)
            # CONTRADICTS is symmetric in meaning — index both directions
            self.contradicts_by_arg[b].append(a)

    def edge_pairs(self, rel_type: str):
        """Yield (src, dst) for every edge of `rel_type`, regardless of
        which endpoint-key convention each edge uses."""
        for e in self.edges.get(rel_type, []):
            a, b = _edge_endpoints(e)
            if a is not None and b is not None:
                yield a, b

    def close(self):
        """No-op — exists so existing `driver.close()` calls don't break."""
        pass


# ============================================================
# LOADER
# ============================================================
_DEFAULT_PATH = "data/neo4j_export_with_new_edges.json"


def load_export(path: Optional[str] = None) -> GraphHandle:
    """Load the Neo4j JSON export and return a GraphHandle.

    Args:
        path: Path to the JSON export. Defaults to
              data/neo4j_export_with_new_edges.json.

    Returns:
        GraphHandle with arguments, policies, edges, and indices loaded.
    """
    path = path or _DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Neo4j export not found at {path}. "
            f"Run export_neo4j_full.py on the machine running Neo4j."
        )

    print(f"Loading Neo4j export from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        export = json.load(f)

    handle = GraphHandle(export)
    print(f"  {handle.meta.get('n_arguments', '?')} arguments, "
          f"{handle.meta.get('n_policies', '?')} policies, "
          f"{handle.meta.get('n_contradicts', '?')} contradicts edges "
          f"(edge keys: {handle._edge_src}/{handle._edge_dst})")
    return handle


# ============================================================
# RETRIEVAL FUNCTIONS (mirror Cypher behaviour)
# ============================================================
def fetch_constrained_pool(driver: GraphHandle, policy_name: str,
                           pool_size: Optional[int] = None):
    """Replicates the Cypher query:

        MATCH (p:Policy {name: $policy_name})
        OPTIONAL MATCH (pro:Argument)-[:SUPPORTS]->(p)
        WITH p, collect({text, vec})[..$pool] AS pro_data
        OPTIONAL MATCH (con:Argument)-[:ATTACKS]->(p)
        RETURN pro_data, collect({text, vec})[..$pool] AS con_data

    Returns:
        (pro_texts, pro_vecs, con_texts, con_vecs)

    Notes:
        - If pool_size is None, returns all arguments for the policy.
          The sweep scripts in this project pass pool_size implicitly
          via their global POOL_SIZE constant by slicing the lists
          afterwards, but to mirror the Cypher `[..$pool]` cap exactly
          you can pass pool_size here and it'll truncate.
        - Argument order matches the order stored in the export. If you
          need deterministic alignment with prior Neo4j runs, make sure
          the export was generated with a stable ORDER BY.
    """
    if policy_name not in driver.policies:
        return [], [], [], []

    p = driver.policies[policy_name]
    pro_ids = p.get("pros", [])
    con_ids = p.get("cons", [])

    if pool_size is not None:
        pro_ids = pro_ids[:pool_size]
        con_ids = con_ids[:pool_size]

    pro_texts, pro_vecs = [], []
    for aid in pro_ids:
        a = driver.arguments.get(aid)
        if a is None or a.get("embedding") is None:
            continue
        pro_texts.append(a["text"])
        pro_vecs.append(a["embedding"])

    con_texts, con_vecs = [], []
    for aid in con_ids:
        a = driver.arguments.get(aid)
        if a is None or a.get("embedding") is None:
            continue
        con_texts.append(a["text"])
        con_vecs.append(a["embedding"])

    return pro_texts, pro_vecs, con_texts, con_vecs


def fetch_system_b_pool_with_ids(driver: GraphHandle, policy_name: str,
                                 pool_size: Optional[int] = None,
                                 hop_size: Optional[int] = None):
    """Replicates the System B multi-hop Cypher query, returning ids too:

      Direct: PRO -[:SUPPORTS]-> policy, with PageRank.
      Direct: CON -[:ATTACKS]-> policy,  with PageRank.
      Multi-hop: CON arguments that CONTRADICT direct PROs.
      Multi-hop: PRO arguments that CONTRADICT direct CONs.
      Union, deduplicate by arg_id, preserving direct-first order.

    This is the single source of truth for the System B candidate pool.
    fetch_system_b_pool (used by System B) is a thin wrapper that drops
    the ids.

    Returns:
        (pro_ids, pro_texts, pro_vecs, pro_pageranks,
         con_ids, con_texts, con_vecs, con_pageranks)
    """
    if policy_name not in driver.policies:
        return [], [], [], [], [], [], [], []

    p = driver.policies[policy_name]
    direct_pro_ids = p.get("pros", [])
    direct_con_ids = p.get("cons", [])

    if pool_size is not None:
        direct_pro_ids = direct_pro_ids[:pool_size]
        direct_con_ids = direct_con_ids[:pool_size]

    # Multi-hop: counter-CONs reached from direct PROs (must ATTACK the policy)
    direct_pro_set = set(direct_pro_ids)
    direct_con_set = set(direct_con_ids)

    hop_con_ids = []
    seen = set()
    for pid in direct_pro_ids:
        for other in driver.contradicts_by_arg.get(pid, []):
            if other in direct_con_set or other in seen:
                continue
            # other must be a CON for this policy (stance == -1 and policy match)
            a = driver.arguments.get(other)
            if a is None:
                continue
            if a.get("stance") == -1 and a.get("policy") == policy_name:
                hop_con_ids.append(other)
                seen.add(other)

    hop_pro_ids = []
    seen = set()
    for cid in direct_con_ids:
        for other in driver.contradicts_by_arg.get(cid, []):
            if other in direct_pro_set or other in seen:
                continue
            a = driver.arguments.get(other)
            if a is None:
                continue
            if a.get("stance") == 1 and a.get("policy") == policy_name:
                hop_pro_ids.append(other)
                seen.add(other)

    if hop_size is not None:
        hop_con_ids = hop_con_ids[:hop_size]
        hop_pro_ids = hop_pro_ids[:hop_size]

    # Build final pools — direct first, then multi-hop, deduplicated
    pro_ids, pro_pool, seen = [], [], set()
    for aid in list(direct_pro_ids) + list(hop_pro_ids):
        if aid in seen:
            continue
        a = driver.arguments.get(aid)
        if a is None or a.get("embedding") is None:
            continue
        seen.add(aid)
        pro_ids.append(aid)
        pro_pool.append(a)

    con_ids, con_pool, seen = [], [], set()
    for aid in list(direct_con_ids) + list(hop_con_ids):
        if aid in seen:
            continue
        a = driver.arguments.get(aid)
        if a is None or a.get("embedding") is None:
            continue
        seen.add(aid)
        con_ids.append(aid)
        con_pool.append(a)

    pro_texts     = [a["text"]                    for a in pro_pool]
    pro_vecs      = [a["embedding"]               for a in pro_pool]
    pro_pageranks = [a.get("pagerank_score", 0.0) for a in pro_pool]

    con_texts     = [a["text"]                    for a in con_pool]
    con_vecs      = [a["embedding"]               for a in con_pool]
    con_pageranks = [a.get("pagerank_score", 0.0) for a in con_pool]

    return (pro_ids, pro_texts, pro_vecs, pro_pageranks,
            con_ids, con_texts, con_vecs, con_pageranks)


def fetch_system_b_pool(driver: GraphHandle, policy_name: str,
                        pool_size: Optional[int] = None,
                        hop_size: Optional[int] = None):
    """System B pool — identical to fetch_system_b_pool_with_ids with the
    ids dropped (see that function's docstring for the query semantics).

    Returns:
        (pro_texts, pro_vecs, pro_pageranks,
         con_texts, con_vecs, con_pageranks)
    """
    (_pro_ids, pro_texts, pro_vecs, pro_pr,
     _con_ids, con_texts, con_vecs, con_pr) = fetch_system_b_pool_with_ids(
        driver, policy_name, pool_size=pool_size, hop_size=hop_size)
    return (pro_texts, pro_vecs, pro_pr,
            con_texts, con_vecs, con_pr)


# ============================================================
# OPTIONAL: full-corpus fetch for vanilla-RAG baselines
# ============================================================
def fetch_full_corpus(driver: GraphHandle):
    """Returns all arguments across all policies."""
    texts, vecs, policies = [], [], []
    for aid, a in driver.arguments.items():
        if a.get("embedding") is None or not a.get("text"):
            continue
        texts.append(a["text"])
        vecs.append(a["embedding"])
        policies.append(a.get("policy", ""))
    return texts, vecs, policies


# ============================================================
# SMOKE TEST
# ============================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATH
    driver = load_export(path)

    test_policy = "We should ban fast food"
    pt, pv, ct, cv = fetch_constrained_pool(driver, test_policy, pool_size=200)
    print(f"\nConstrained pool for '{test_policy}':")
    print(f"  {len(pt)} PROs, {len(ct)} CONs")
    if pt:
        print(f"  First PRO: {pt[0][:100]}")

    pt2, pv2, pr2, ct2, cv2, cr2 = fetch_system_b_pool(
        driver, test_policy, pool_size=200, hop_size=100
    )
    print(f"\nSystem B pool for '{test_policy}':")
    print(f"  {len(pt2)} PROs (with multi-hop), {len(ct2)} CONs")

    (pi3, pt3, pv3, pr3,
     ci3, ct3, cv3, cr3) = fetch_system_b_pool_with_ids(
        driver, test_policy, pool_size=200, hop_size=100
    )
    same = (pt3 == pt2 and ct3 == ct2 and pr3 == pr2 and cr3 == cr2)
    print(f"  with_ids pool identical to System B pool: {same}")
    n_equiv = sum(1 for _ in driver.edge_pairs("equivalent"))
    print(f"  EQUIVALENT edges readable: {n_equiv}")