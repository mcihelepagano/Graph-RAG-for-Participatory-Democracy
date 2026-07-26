"""
refresh_pagerank_and_reset.py
=============================
One-shot consistency repair after regenerating CONTRADICTS edges without
recomputing PageRank. Four phases, in order:

  PHASE 1 — VERIFY the export.
    Confirms data/neo4j_export_with_new_edges.json contains the NEW
    CONTRADICTS edge set (expected ~344K; warns outside +/-10%), reports
    all relation types, node counts, and which edge schema is in use
    (source/target vs arg1_id/arg2_id — both have appeared before).

  PHASE 2 — CHECK PageRank state.
    Detects PageRank-like numeric properties on argument nodes
    (pagerank / page_rank / pr / ppr ...), reports coverage and
    distribution. If neo4j_export_adapter.py is importable/readable, its
    source is scanned to report WHICH property the pipeline actually
    consumes. Also looks for a personalized-PageRank sidecar
    (policy_ppr_scores.json).

  PHASE 3 — RECOMPUTE PageRank on the export itself.
    Robust scipy-sparse power iteration over the CONTRADICTS graph as it
    exists IN THE EXPORT — by construction the scores can never disagree
    with the edges the cluster pipeline walks (the inconsistency that
    caused this mess). Two products:
      (a) Global PageRank  -> written back onto each argument node under
          the detected property name (default 'pagerank').
      (b) Personalized PageRank per policy, seeded uniformly on that
          policy's PRO+CON arguments (equivalent to the GDS seed-at-
          policy-node walk, one hop in)  -> data/policy_ppr_scores.json.
    Details: damping 0.85, symmetrized edges (CONTRADICTS scoring was
    min(fwd,bwd), i.e. effectively undirected), edge weights used if
    present, dangling mass redistributed to the teleport vector,
    L1 tolerance 1e-10, max 200 iterations, deterministic.
    Reports Spearman correlation old-vs-new global PageRank — a direct
    measure of how much the staleness mattered.
    The export is updated ATOMICALLY with a timestamped .bak of the
    original. Old per-node values are preserved in a sidecar report.

  PHASE 4 — DELETE/STRIP stale outputs (System B only).
    Per-policy checkpointing would otherwise silently reuse the stale
    retrievals. Removed/stripped:
      UC1  results/sweep_uc1/configs/pool*_k*/system_b.json
           results/sweep_uc1/sweep_summary.{json,csv}, plots/
           results/eval_uc1/{summary.json,summary.csv,per_policy.csv}
           results/eval_uc1/deterministic.json      -> strip B key
           results/eval_uc1/rubric_scores.json      -> strip B key
           results/eval_uc1/pairwise_axes.json      -> delete (System B
              is the anchor of every pairwise comparison; nothing in the
              file survives a System B change. Judge is disabled anyway.)
      UC2  results/sweep_uc2/retrieval/system_b_k*.json
           results/sweep_uc2/summaries/system_b_k*.json
           results/sweep_uc2/metrics_deterministic.json -> strip B
           results/sweep_uc2/sweep_summary.csv, plots/
           results/eval_uc2/<fmt>/summaries/system_b_k*.json
           results/eval_uc2/<fmt>/{coverage,faithfulness,bertscore,
              stance_balance,rubric_scores}.json    -> strip B key
           results/eval_uc2/<fmt>/pairwise.json     -> delete
           results/eval_uc2/<fmt>/{summary.json,summary.csv}
           results/eval_uc2/format_comparison.json
    KEPT: ground truth, System A / Baseline A / Baseline B / SimpleRAG
    retrievals, summaries and judge scores, simplerag_retrieval.json,
    UC2 reference summaries. Every stripped JSON gets a .bak first.

SAFETY
  - DRY-RUN BY DEFAULT. Nothing is written or deleted without --apply.
  - Phase 3 refuses to run if Phase 1 fails (edge count far from
    expected), unless --force.

USAGE (from /home/pagano/evaluation/ on the cluster):
    python refresh_pagerank_and_reset.py            # dry run, full report
    python refresh_pagerank_and_reset.py --apply    # do it
    python refresh_pagerank_and_reset.py --apply --skip-pagerank
    python refresh_pagerank_and_reset.py --expected-contradicts 344000

AFTER THIS SCRIPT: re-run sweep_uc1_k.py, sweep_uc2_k.py, eval_uc1.py,
eval_uc2.py. Only System B work is redone; everything else loads from
checkpoint.

NOTE: this updates the EXPORT, not the Neo4j database. The pagerank
property inside Neo4j on the local Windows machine remains stale; if you
later re-export from Neo4j, either recompute there (GDS) first or re-run
Phase 3 on the fresh export.
"""

import os
import re
import csv
import sys
import glob
import json
import time
import shutil
import argparse
from collections import Counter, defaultdict

import numpy as np

# ============================================================
# CONFIG (defaults; override via CLI)
# ============================================================
EXPORT_PATH          = "data/neo4j_export_with_new_edges.json"
PPR_OUT_PATH         = "data/policy_ppr_scores.json"
ADAPTER_PATH         = "neo4j_export_adapter.py"
EXPECTED_CONTRADICTS = 344_000
EDGE_TOLERANCE       = 0.10          # warn if outside +/-10% of expected

DAMPING   = 0.85
MAX_ITER  = 200
TOL       = 1e-10

PR_PROP_CANDIDATES = ["pagerank", "page_rank", "pr", "pr_score",
                      "pagerank_score", "ppr"]

ADAPTER_PATH = "neo4j_export_adapter.py"


def load_handle(export_path):
    """Load the export through the project's own adapter so edge-key
    detection, dict-of-lists-by-relation-type structure, and per-edge
    (source/target vs arg1_id/arg2_id) handling are guaranteed to match
    what System B/C actually consume — no second, divergent parser.

    The adapter is looked up next to THIS script (not the cwd), since
    this script may be invoked from a different directory than
    neo4j_export_adapter.py lives in (e.g. `python3 scripts/refresh_*.py`
    run from the parent of scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, ADAPTER_PATH),  # next to this script
        ADAPTER_PATH,                            # cwd (fallback)
    ]
    adapter_file = next((c for c in candidates if os.path.exists(c)), None)
    if adapter_file is None:
        raise SystemExit(
            f"FATAL: {ADAPTER_PATH} not found. Looked in:\n"
            + "\n".join(f"  {os.path.abspath(c)}" for c in candidates)
            + "\nPlace this script next to neo4j_export_adapter.py, or "
              "run from that directory.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(adapter_file)))
    import neo4j_export_adapter as adapter
    handle = adapter.load_export(export_path)
    return handle, adapter

STALE_SYSTEMS    = ["System B"]
STALE_SAFE_NAMES = ["system_b"]


# ============================================================
# SMALL HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def backup(path, apply):
    """Timestamped .bak next to the file. Returns the backup path."""
    bak = f"{path}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
    if apply:
        shutil.copy2(path, bak)
    return bak


class Plan:
    """Collects actions; executes only with --apply."""
    def __init__(self, apply):
        self.apply = apply
        self.n_delete = self.n_strip = 0

    def delete(self, path):
        self.n_delete += 1
        print(f"  [delete] {path}")
        if self.apply:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def strip_keys(self, path, keys):
        """Remove top-level keys from a JSON checkpoint (with .bak)."""
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN]   {path}: unreadable ({e}) -> deleting instead")
            self.delete(path)
            return
        present = [k for k in keys if k in data]
        if not present:
            print(f"  [ok]     {path}: no {keys} keys present")
            return
        self.n_strip += 1
        print(f"  [strip]  {path}: removing {present} "
              f"(backup -> {backup(path, self.apply)})")
        if self.apply:
            for k in present:
                del data[k]
            save_json(data, path)


# ============================================================
# PHASE 1 — VERIFY EXPORT
# ============================================================
def phase1_verify(export, handle, expected, tolerance):
    print("\n" + "=" * 70)
    print("PHASE 1 — VERIFY EXPORT")
    print("=" * 70)
    n_args = len(export.get("arguments", {}))
    n_pols = len(export.get("policies", {}))
    print(f"  arguments: {n_args:,} | policies: {n_pols}")
    print(f"  edge-key convention detected by adapter: "
          f"{handle._edge_src}/{handle._edge_dst}")

    by_type = Counter()
    for rel_type, edge_list in handle.edges.items():
        by_type[rel_type.upper()] = len(edge_list)
    print("  relation counts:")
    for t, n in by_type.most_common():
        print(f"    {t:<22} {n:,}")

    n_contra = sum(n for t, n in by_type.items() if "CONTRADICT" in t
                  and "COHERENCE" not in t)
    lo, hi = expected * (1 - tolerance), expected * (1 + tolerance)
    ok = lo <= n_contra <= hi
    verdict = "OK" if ok else "** OUTSIDE EXPECTED RANGE **"
    print(f"\n  CONTRADICTS: {n_contra:,} "
          f"(expected ~{expected:,} +/-{int(tolerance*100)}%) -> {verdict}")
    if not ok:
        print("  -> This export does NOT look like the new edge set. "
              "Check that the post-regeneration export was used. "
              "Use --force to proceed anyway.")
    return ok, n_contra


# ============================================================
# PHASE 2 — CHECK PAGERANK STATE
# ============================================================
def phase2_check(export, adapter):
    print("\n" + "=" * 70)
    print("PHASE 2 — CHECK PAGERANK STATE")
    print("=" * 70)
    args = export.get("arguments", {})
    found = {}
    for prop in PR_PROP_CANDIDATES:
        vals = [a[prop] for a in args.values()
                if isinstance(a.get(prop), (int, float))]
        if vals:
            found[prop] = vals
    if not found:
        print("  No PageRank-like property on argument nodes.")
    for prop, vals in found.items():
        v = np.array(vals, dtype=float)
        cov = len(vals) / max(1, len(args))
        print(f"  '{prop}': coverage {cov:.1%} | "
              f"min={v.min():.3e} med={np.median(v):.3e} "
              f"max={v.max():.3e} | zeros={(v == 0).mean():.1%}")

    src = open(adapter.__file__, encoding="utf-8").read()
    hits = sorted({m for m in re.findall(
        r"""\.get\(["'](%s)["']""" % "|".join(PR_PROP_CANDIDATES), src)})
    prop_in_use = hits[0] if hits else None
    if prop_in_use:
        print(f"  neo4j_export_adapter.py reads: {hits} "
              f"-> will update '{prop_in_use}'")
    else:
        print("  neo4j_export_adapter.py: no PageRank property reference "
              "detected — VERIFY MANUALLY which score System B consumes.")

    if os.path.exists(PPR_OUT_PATH):
        try:
            ppr = load_json(PPR_OUT_PATH)
            print(f"  PPR sidecar {PPR_OUT_PATH}: {len(ppr)} policies "
                  f"(STALE if computed pre-regeneration; will be rebuilt).")
        except json.JSONDecodeError:
            print(f"  PPR sidecar {PPR_OUT_PATH}: unreadable.")
    else:
        print(f"  No PPR sidecar at {PPR_OUT_PATH}.")

    # min-max normalization inside calculate_mmr/_community makes absolute
    # scale irrelevant — only within-pool ordering/spacing matters, so a
    # standard global-PageRank simplex is a safe drop-in for this property.
    if prop_in_use is None:
        prop_in_use = (next(iter(found)) if found
                       else PR_PROP_CANDIDATES[0])
        print(f"  Defaulting to property '{prop_in_use}'.")
    return prop_in_use, found


# ============================================================
# PHASE 3 — RECOMPUTE PAGERANK + PPR ON THE EXPORT GRAPH
# ============================================================
def build_contradicts_matrix(export, handle, adapter):
    """Symmetric sparse adjacency over the CONTRADICTS subgraph, built
    from handle.edges["contradicts"] via the adapter's own
    _edge_endpoints — the exact same edges System B/C walk, parsed the
    exact same way (per-edge source/target vs arg1_id/arg2_id)."""
    from scipy import sparse
    arg_ids = list(export["arguments"].keys())
    idx = {a: i for i, a in enumerate(arg_ids)}
    n = len(arg_ids)

    contra_edges = handle.edges.get("contradicts", [])
    rows, cols, vals = [], [], []
    skipped = 0
    for e in contra_edges:
        s, d = adapter._edge_endpoints(e)
        i, j = idx.get(s), idx.get(d)
        if i is None or j is None or i == j:
            skipped += 1
            continue
        w = float(e.get("weight", e.get("score", 1.0)) or 1.0)
        if w <= 0:
            w = 1.0
        # symmetrize: CONTRADICTS scoring was min(fwd,bwd) — undirected.
        rows += [i, j]; cols += [j, i]; vals += [w, w]
    if skipped:
        print(f"  [WARN] {skipped:,} CONTRADICTS edges referenced unknown "
              f"or self node ids (skipped) — schema drift if large.")
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A, arg_ids, idx


def power_iteration(A, teleport, damping=DAMPING, max_iter=MAX_ITER,
                    tol=TOL):
    """PageRank by power iteration. A: symmetric weighted adjacency.
    teleport: probability vector (sums to 1). Dangling mass is
    redistributed to the teleport vector. Returns (scores, iters, resid)."""
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    dangling = out == 0
    inv_out = np.zeros(n)
    inv_out[~dangling] = 1.0 / out[~dangling]

    x = teleport.copy()
    for it in range(1, max_iter + 1):
        # column-stochastic transition applied to x
        spread = A.T @ (x * inv_out)
        dangle_mass = x[dangling].sum()
        x_new = damping * (spread + dangle_mass * teleport) \
            + (1.0 - damping) * teleport
        x_new /= x_new.sum()
        resid = np.abs(x_new - x).sum()
        x = x_new
        if resid < tol:
            return x, it, resid
    return x, max_iter, resid


def spearman(a, b):
    """Spearman rho without scipy.stats (rank + Pearson)."""
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        return r
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def phase3_recompute(export, handle, adapter, pr_prop, apply, export_path):
    print("\n" + "=" * 70)
    print(f"PHASE 3 — RECOMPUTE PAGERANK (property: '{pr_prop}') "
          f"{'' if apply else '[DRY RUN — nothing written]'}")
    print("=" * 70)
    try:
        from scipy import sparse  # noqa: F401
    except ImportError:
        raise SystemExit("FATAL: scipy required for sparse PageRank "
                         "(pip install scipy --user).")

    A, arg_ids, idx = build_contradicts_matrix(export, handle, adapter)
    n = len(arg_ids)
    nnz_deg = int((np.asarray(A.sum(axis=1)).ravel() > 0).sum())
    print(f"  graph: {n:,} argument nodes, "
          f"{A.nnz // 2:,} undirected CONTRADICTS edges, "
          f"{nnz_deg:,} non-isolated nodes")

    # ---- (a) Global PageRank --------------------------------------
    uniform = np.full(n, 1.0 / n)
    pr, iters, resid = power_iteration(A, uniform)
    print(f"  global PageRank: converged in {iters} iterations "
          f"(L1 residual {resid:.2e})")

    old = np.array([export["arguments"][a].get(pr_prop) for a in arg_ids],
                   dtype=object)
    have_old = np.array([isinstance(v, (int, float)) for v in old])
    if have_old.sum() > 10:
        rho = spearman(pr[have_old],
                       np.array(old[have_old], dtype=float))
        print(f"  old-vs-new Spearman rho = {rho:.4f} on "
              f"{int(have_old.sum()):,} nodes "
              f"(low rho = staleness mattered a lot)")

    # ---- (b) Personalized PageRank per policy ---------------------
    policies = export.get("policies", {})
    ppr_all, n_empty = {}, 0
    print(f"  personalized PageRank for {len(policies)} policies...")
    for pname, pdata in policies.items():
        seeds = [idx[a] for a in (list(pdata.get("pros", []))
                                  + list(pdata.get("cons", [])))
                 if a in idx]
        if not seeds:
            n_empty += 1
            continue
        teleport = np.zeros(n)
        teleport[seeds] = 1.0 / len(seeds)
        scores, _, _ = power_iteration(A, teleport)
        # store only this policy's own arguments (what the pool needs)
        ppr_all[pname] = {arg_ids[i]: float(scores[i])
                          for i in seeds}
    if n_empty:
        print(f"  [WARN] {n_empty} policies had no resolvable arguments.")

    # ---- write back ------------------------------------------------
    sidecar = {"property": pr_prop, "damping": DAMPING,
               "old_values": {a: (float(v) if isinstance(v, (int, float))
                                  else None)
                              for a, v in zip(arg_ids, old)}}
    if apply:
        for a, v in zip(arg_ids, pr):
            export["arguments"][a][pr_prop] = float(v)
        bak = backup(export_path, apply=True)
        tmp = export_path + ".tmp"
        save_json(export, tmp)
        os.replace(tmp, export_path)
        print(f"  export updated atomically (backup: {bak})")
        if os.path.exists(PPR_OUT_PATH):
            backup(PPR_OUT_PATH, apply=True)
        save_json(ppr_all, PPR_OUT_PATH)
        save_json(sidecar, "results/pagerank_refresh_report.json")
        print(f"  PPR -> {PPR_OUT_PATH} | old values -> "
              f"results/pagerank_refresh_report.json")
    else:
        print("  [dry run] would update export in place (+.bak), write "
              f"{PPR_OUT_PATH} and results/pagerank_refresh_report.json")


# ============================================================
# PHASE 4 — DELETE / STRIP STALE OUTPUTS
# ============================================================
def phase4_cleanup(apply):
    print("\n" + "=" * 70)
    print(f"PHASE 4 — REMOVE STALE SYSTEM B OUTPUTS "
          f"{'' if apply else '[DRY RUN — nothing deleted]'}")
    print("=" * 70)
    plan = Plan(apply)

    def rm_glob(pattern):
        for p in sorted(glob.glob(pattern)):
            plan.delete(p)

    # ---- UC1 sweep ----
    for safe in STALE_SAFE_NAMES:
        rm_glob(f"results/sweep_uc1/configs/pool*_k*/{safe}.json")
    rm_glob("results/sweep_uc1/sweep_summary.json")
    rm_glob("results/sweep_uc1/sweep_summary.csv")
    rm_glob("results/sweep_uc1/plots")

    # ---- UC1 eval ----
    for f in ["results/eval_uc1/deterministic.json",
              "results/eval_uc1/rubric_scores.json"]:
        if os.path.exists(f):
            plan.strip_keys(f, STALE_SYSTEMS)
    rm_glob("results/eval_uc1/pairwise_axes.json")  # B anchors every pair
    rm_glob("results/eval_uc1/summary.json")
    rm_glob("results/eval_uc1/summary.csv")
    rm_glob("results/eval_uc1/per_policy.csv")

    # ---- UC2 sweep ----
    for safe in STALE_SAFE_NAMES:
        rm_glob(f"results/sweep_uc2/retrieval/{safe}_k*.json")
        rm_glob(f"results/sweep_uc2/summaries/{safe}_k*.json")
    if os.path.exists("results/sweep_uc2/metrics_deterministic.json"):
        plan.strip_keys("results/sweep_uc2/metrics_deterministic.json",
                        STALE_SYSTEMS)
    rm_glob("results/sweep_uc2/sweep_summary.csv")
    rm_glob("results/sweep_uc2/plots")

    # ---- UC2 eval (every format dir: paragraph, structured, ...) ----
    for d in sorted(glob.glob("results/eval_uc2/*/")):
        for safe in STALE_SAFE_NAMES:
            rm_glob(os.path.join(d, "summaries", f"{safe}_k*.json"))
        for name in ["coverage.json", "faithfulness.json",
                     "bertscore.json", "stance_balance.json",
                     "rubric_scores.json"]:
            f = os.path.join(d, name)
            if os.path.exists(f):
                plan.strip_keys(f, STALE_SYSTEMS)
        rm_glob(os.path.join(d, "pairwise.json"))   # B anchors every pair
        rm_glob(os.path.join(d, "summary.json"))
        rm_glob(os.path.join(d, "summary.csv"))
    rm_glob("results/eval_uc2/format_comparison.json")

    print(f"\n  plan: {plan.n_delete} delete(s), {plan.n_strip} "
          f"surgical strip(s)."
          + ("" if apply else "  Re-run with --apply to execute."))


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--apply", action="store_true",
                    help="actually write/delete (default: dry run)")
    ap.add_argument("--export", default=EXPORT_PATH)
    ap.add_argument("--expected-contradicts", type=int,
                    default=EXPECTED_CONTRADICTS)
    ap.add_argument("--skip-pagerank", action="store_true",
                    help="skip Phase 3 (verification + cleanup only)")
    ap.add_argument("--skip-cleanup", action="store_true",
                    help="skip Phase 4 (verification + pagerank only)")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if Phase 1 verification fails")
    args = ap.parse_args()

    if not os.path.exists(args.export):
        raise SystemExit(f"FATAL: export not found: {args.export}")
    print(f"Loading export: {args.export} ...")
    export = load_json(args.export)
    handle, adapter = load_handle(args.export)

    ok, _n_contra = phase1_verify(export, handle, args.expected_contradicts,
                                  EDGE_TOLERANCE)
    pr_prop, _ = phase2_check(export, adapter)

    if not ok and not args.force:
        raise SystemExit("\nABORTING before Phase 3/4: edge verification "
                         "failed (use --force to override).")

    if not args.skip_pagerank:
        phase3_recompute(export, handle, adapter, pr_prop, args.apply,
                         args.export)
    if not args.skip_cleanup:
        phase4_cleanup(args.apply)

    print("\nDONE." + ("" if args.apply else
          "  This was a DRY RUN — re-run with --apply to execute."))
    if args.apply:
        print("Next: python sweep_uc1_k.py && python sweep_uc2_k.py && "
              "python eval_uc1.py && python eval_uc2.py")


if __name__ == "__main__":
    main()