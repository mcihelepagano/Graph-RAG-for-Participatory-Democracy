"""
build_hard_pools.py  (PATCHED — interleave fix)
===============================================
Builds HARDER per-policy candidate pools by injecting topically-adjacent
"distracting" arguments (Cuconasu et al., SIGIR 2024, sec 4.2) drawn from
SIBLING policies, to break the relevance-metric saturation diagnosed on the
native pools (~57.7% relevant -> random nDCG ~= 0.39, non-discriminative).

WHAT CHANGED IN THIS REVISION
-----------------------------
The previous revision APPENDED distractors AFTER the native arguments in
each policy's pro/con id list. Downstream, fetch_constrained_pool /
fetch_system_b_pool_with_ids slice that list with [:pool_size] (pool_size
up to 200). Because the native block already filled positions 0..199, the
appended distractors (positions 200+) were ALWAYS sliced off, so NO system
ever retrieved a distractor — confirmed by a random Baseline B retrieving
zero distractors (statistically impossible if they were in the sampled
pool). The hardened ground truth was therefore being scored against an
unhardened retrieval pool.

FIX: after building the combined (native + distractor) id list per stance,
INTERLEAVE it with a deterministic, per-(policy, stance, level) seeded
shuffle. A downstream [:pool_size] slice now draws a representative
native+distractor mix, so the retrieval pool carries the same base rate
that was annotated. The shuffle is applied ONLY when distractors were
actually injected, so the level-0 export remains byte-identical to native
(the control is preserved).

ALGORITHM (per evaluation policy P, per stance s in {pros, cons}):
  1. Find P's SIBLING policies:
       - MetaPolicy co-membership if the export carries that field, else
       - the K policies whose centroid is most cosine-similar to P's.
  2. Gather candidate distractors = same-stance arguments of the siblings
     (sibling PROs -> P's PRO pool; sibling CONs -> P's CON pool), skipping
     any that duplicate (by text) an argument P already owns.
  3. Rank candidates NEAREST-first by cosine to P's centroid (the hardest,
     most "distracting" ones) -- or RANDOM with --selection random as the
     system-agnostic control -- and take the top D, for each D in LEVELS.
  4. NATURALISE each chosen distractor into P (Option 1): create a
     POLICY-SCOPED COPY of the argument with:
         id     = "<orig_id>__distractor_into__<P_slug>"   (fresh, unique)
         policy = P                                        (re-stamped)
         stance = +1 if pros else -1                       (re-stamped)
         is_distractor = True
         distractor_source_policy = <orig policy>
         distractor_source_id     = <orig id>
     append the copy's id to the combined list, INTERLEAVE (seeded shuffle),
     and add the copy to the arguments map.

WHY RE-STAMP policy+stance (Option 1)?
  Every retrieval path in the pipeline keys off the target policy:
    - fetch_constrained_pool reads policies[P]["pros"/"cons"] directly
      (Systems A, baselines) -> distractor copies appear as candidates.
    - fetch_system_b_pool_with_ids guards multi-hop with
        a["stance"]==s and a["policy"]==policy_name
      Re-stamping policy=P and stance=s lets both A and B see the IDENTICAL
      hardened pool by construction.
  NOTE: injected distractors carry NO cross-policy CONTRADICTS edges, so a
  re-stamped distractor enters System B as a graph-ISOLATED direct (no hop
  neighbours within P). This must be stated in the methodology.

LEVEL 0 IS A BYTE-IDENTICAL CONTROL
  At level 0 nothing is injected, no shuffle is applied, so the level-0
  export reproduces the native pool exactly. Use it as the sanity gate.

DOWNSTREAM (per level export):
  1. Re-run annotate_distractors.py pointed at the level export (each level
     needs its OWN dual-LLM annotation of the injected distractors).
     NOTE: the GROUND TRUTH is keyed by argument TEXT, so the interleave
     shuffle does NOT require re-annotation if you already annotated this
     level — the existing ground_truth_<sel>_L<lvl>.json still scores the
     reshuffled pool correctly. Only the EXPORT changes order.
  2. sweep_uc1_k -> eval_uc1 (and UC2) against that export + its GT.
  3. Plot a discriminating metric vs. realised base rate per level.

Usage:
  python build_hard_pools.py
  python build_hard_pools.py --selection random          # agnostic control
  python build_hard_pools.py --levels 0 100 -1           # cheaper sweep
"""

import os
import re
import json
import random
import argparse
import numpy as np

from common import load_json, save_json


# ============================================================
# CONFIG
# ============================================================
DEFAULT_EXPORT        = "data/neo4j_export_with_new_edges.json"
DEFAULT_POLICIES_FILE = "data/selected_policies.json"
OUTPUT_DIR            = "data/hard_pools"

# Distractors injected PER STANCE per level. 0 = native (control).
# -1 == "full" (take all available sibling distractors).
DISTRACTOR_LEVELS = [0, 50, 100, 200, -1]

# "nearest": hardest distractors first (centroid cosine).
# "random" : uniform from siblings -- system-agnostic control.
DISTRACTOR_SELECTION = "nearest"

# Sibling count when the export has no MetaPolicy field (centroid fallback).
N_NEAREST_SIBLING_POLICIES = 4

# Candidate MetaPolicy field names probed on policy / argument records.
METAPOLICY_KEYS = ["meta_policy", "metapolicy", "meta", "topic", "cluster"]

NATIVE_BASE_RATE = 0.577   # from generate_ground_truth report, for the
                           # base-rate lower-bound projection only.
SEED = 42


# ============================================================
# POLICY / SIBLING RESOLUTION
# ============================================================
def slug(name):
    """Filesystem/id-safe slug for a policy name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_eval_policies(policies_file):
    """Return the list of evaluation policy names (the stratified 30).
    Prefers selected_policies.json; falls back to policies.py POLICIES."""
    if os.path.exists(policies_file):
        data = load_json(policies_file)
        sel = data.get("selected", data) if isinstance(data, dict) else data
        names = [s["policy"] if isinstance(s, dict) else s for s in sel]
        if names:
            print(f"  Eval policies: {len(names)} from {policies_file}")
            return names
    try:
        from policies import POLICIES
        print(f"  Eval policies: {len(POLICIES)} from policies.py")
        return list(POLICIES)
    except Exception as e:
        raise SystemExit(
            f"Could not load eval policies from {policies_file} or "
            f"policies.py ({e}). Run select_policies_stratified.py first."
        )


def detect_metapolicy_key(export):
    """Return (level, key): level in {'policy','argument',None}."""
    pol = export.get("policies", {})
    if pol:
        sample = next(iter(pol.values()))
        for k in METAPOLICY_KEYS:
            if sample.get(k) not in (None, ""):
                return "policy", k
    args = export.get("arguments", {})
    if args:
        sample = next(iter(args.values()))
        for k in METAPOLICY_KEYS:
            if sample.get(k) not in (None, ""):
                return "argument", k
    return None, None


def policy_centroids(export):
    """L2-normalised centroid per policy from its own arguments' embeddings.
    Returns (names, matrix[P,dim])."""
    arguments = export["arguments"]
    names, vecs = [], []
    for name, data in export["policies"].items():
        embs = []
        for aid in list(data.get("pros", [])) + list(data.get("cons", [])):
            a = arguments.get(aid)
            if a is not None and a.get("embedding") is not None:
                embs.append(a["embedding"])
        if not embs:
            continue
        c = np.mean(np.asarray(embs, dtype=float), axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        names.append(name)
        vecs.append(c)
    return names, np.asarray(vecs, dtype=float)


def build_sibling_map(export, eval_policies):
    """Return ({policy: [sibling,...]}, (mode, param))."""
    level, key = detect_metapolicy_key(export)
    siblings = {}

    if level in ("policy", "argument"):
        args = export["arguments"]
        pol_mp = {}
        if level == "policy":
            for name, data in export["policies"].items():
                pol_mp[name] = data.get(key)
            print(f"  Sibling source: MetaPolicy '{key}' on policy records")
        else:
            for name, data in export["policies"].items():
                mps = [args[a].get(key)
                       for a in list(data.get("pros", [])) + list(data.get("cons", []))
                       if a in args and args[a].get(key) not in (None, "")]
                if mps:
                    pol_mp[name] = max(set(mps), key=mps.count)
            print(f"  Sibling source: MetaPolicy '{key}' on argument records "
                  f"(policy = majority vote)")
        groups = {}
        for name, mp in pol_mp.items():
            groups.setdefault(mp, []).append(name)
        for p in eval_policies:
            mp = pol_mp.get(p)
            siblings[p] = [q for q in groups.get(mp, []) if q != p]
        return siblings, ("metapolicy", key)

    # Fallback: centroid-cosine nearest policies.
    print(f"  Sibling source: centroid cosine (no MetaPolicy field); "
          f"K={N_NEAREST_SIBLING_POLICIES}")
    names, M = policy_centroids(export)
    idx = {n: i for i, n in enumerate(names)}
    sims = M @ M.T
    for p in eval_policies:
        if p not in idx:
            siblings[p] = []
            continue
        order = np.argsort(sims[idx[p]])[::-1]
        siblings[p] = [names[j] for j in order
                       if names[j] != p][:N_NEAREST_SIBLING_POLICIES]
    return siblings, ("centroid", N_NEAREST_SIBLING_POLICIES)


# ============================================================
# DISTRACTOR SELECTION
# ============================================================
def gather_candidates(export, target, siblings, stance_key):
    """Candidate (orig_id, embedding) for one stance. Same-stance only,
    deduped by text against the target's existing pool and within siblings."""
    arguments = export["arguments"]
    own = set(export["policies"].get(target, {}).get("pros", []) +
              export["policies"].get(target, {}).get("cons", []))
    own_texts = {arguments[a]["text"].strip()
                 for a in own if a in arguments and arguments[a].get("text")}

    cand, seen_text = [], set()
    for sib in siblings:
        for aid in export["policies"].get(sib, {}).get(stance_key, []):
            if aid in own:
                continue
            a = arguments.get(aid)
            if a is None or a.get("embedding") is None or not a.get("text"):
                continue
            t = a["text"].strip()
            if t in own_texts or t in seen_text:
                continue
            seen_text.add(t)
            cand.append((aid, np.asarray(a["embedding"], dtype=float)))
    return cand


def rank_candidates(cand, target_centroid, selection, rng):
    """Order candidate (id, emb) -> list of orig_ids in selection order."""
    if not cand:
        return []
    if selection == "random":
        order = list(range(len(cand)))
        rng.shuffle(order)
        return [cand[i][0] for i in order]
    embs = np.stack([c[1] for c in cand])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (embs / norms) @ target_centroid
    return [cand[i][0] for i in np.argsort(sims)[::-1]]


# ============================================================
# NATURALISATION (Option 1: policy-scoped COPY + re-stamp)
# ============================================================
def make_distractor_copy(orig_record, orig_id, target_policy, stance_val):
    """Return (new_id, new_record): a policy-scoped COPY of orig_record,
    re-stamped to belong to target_policy with the given stance."""
    new_id = f"{orig_id}__distractor_into__{slug(target_policy)}"
    r = dict(orig_record)
    r["policy"] = target_policy
    r["stance"] = stance_val
    r["is_distractor"] = True
    # Do NOT inherit the source PageRank: it was computed in the source
    # policy's CONTRADICTS neighbourhood and is meaningless in the target
    # pool. Set to None; the floor pass assigns the target policy's minimum
    # native PageRank (graph-isolated => no in-network centrality).
    r["pagerank_score"] = None
    r["distractor_source_policy"] = orig_record.get("policy")
    r["distractor_source_id"] = orig_id
    return new_id, r


def build_level_export(base_export, eval_policies, sibling_map,
                       centroid_lookup, level, selection, rng, seed):
    """Produce one expanded export for `level`. Distractors are combined
    with natives then INTERLEAVED via a deterministic per-(policy, stance,
    level) seeded shuffle, so a downstream [:pool_size] cap draws a mix.
    Level 0 injects nothing and is NOT shuffled (byte-identical control).
    Returns (export, stats)."""
    export = json.loads(json.dumps(base_export))   # deep copy
    arguments = export["arguments"]
    stats = {}

    for p in eval_policies:
        sibs = sibling_map.get(p, [])
        centroid = centroid_lookup.get(p)
        p_stats = {}
        for stance_key, stance_val in (("pros", 1), ("cons", -1)):
            cand = gather_candidates(base_export, p, sibs, stance_key)
            ordered = rank_candidates(cand, centroid, selection, rng)
            take = ordered if level == -1 else ordered[:level]

            new_ids = []
            for orig_id in take:
                new_id, rec = make_distractor_copy(
                    base_export["arguments"][orig_id], orig_id, p, stance_val)
                arguments[new_id] = rec          # policy-scoped copy
                new_ids.append(new_id)

            combined = list(export["policies"][p][stance_key]) + new_ids
            # Interleave distractors among natives so a downstream pool cap
            # (fetch_constrained_pool slices [:pool_size]) draws a
            # representative native+distractor mix rather than all-native.
            # Deterministic per (policy, stance, level). When no distractors
            # were injected (level 0), combined == native list and we DO NOT
            # shuffle -> preserves the level-0 byte-identical control.
            if new_ids:
                shuf = random.Random(f"{seed}|{p}|{stance_key}|{level}")
                shuf.shuffle(combined)
            export["policies"][p][stance_key] = combined

            p_stats[stance_key] = {
                "n_native": len(base_export["policies"][p][stance_key]),
                "n_candidates": len(ordered),
                "n_injected": len(new_ids),
            }

        # --- PageRank floor pass ---------------------------------------
        # Distractors are graph-isolated (no CONTRADICTS edges into P were
        # mined), so they have no in-network centrality. Assign each the
        # MINIMUM native PageRank of its stance pool. Keyed by id and read
        # from base_export, so the interleave shuffle does NOT affect it.
        for stance_key in ("pros", "cons"):
            native_prs = [
                base_export["arguments"][a].get("pagerank_score", 0.0)
                for a in base_export["policies"][p][stance_key]
                if a in base_export["arguments"]
            ]
            floor = min(native_prs) if native_prs else 0.0
            for aid in export["policies"][p][stance_key]:
                rec = arguments[aid]
                if rec.get("is_distractor") and rec.get("pagerank_score") is None:
                    rec["pagerank_score"] = floor

        stats[p] = p_stats
    return export, stats


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_EXPORT)
    ap.add_argument("--policies-file", default=DEFAULT_POLICIES_FILE)
    ap.add_argument("--out-dir", default=OUTPUT_DIR)
    ap.add_argument("--selection", choices=["nearest", "random"],
                    default=DISTRACTOR_SELECTION)
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="override levels (-1 == full)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    levels = args.levels if args.levels is not None else DISTRACTOR_LEVELS
    rng = random.Random(args.seed)

    print("=" * 70)
    print("BUILD HARD POOLS — distracting-argument injection (Option 1)")
    print("  interleave fix: distractors shuffled into the pool, not appended")
    print("=" * 70)

    if not os.path.exists(args.export):
        raise SystemExit(f"Export not found: {args.export} "
                         f"(this runs on the cluster where the export lives).")

    print(f"Loading export: {args.export}")
    base_export = load_json(args.export)
    print(f"  {len(base_export.get('arguments', {}))} arguments, "
          f"{len(base_export.get('policies', {}))} policies")

    eval_policies = load_eval_policies(args.policies_file)
    sibling_map, sib_meta = build_sibling_map(base_export, eval_policies)

    names, M = policy_centroids(base_export)
    centroid_lookup = {n: M[i] for i, n in enumerate(names)}

    empty = [p for p in eval_policies if not sibling_map.get(p)]
    if empty:
        print(f"\n  WARNING: {len(empty)} eval policies have NO siblings and "
              f"will get zero distractors at every level:")
        for p in empty:
            print(f"    - {p}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {
        "source_export": args.export,
        "selection": args.selection,
        "sibling_definition": {"mode": sib_meta[0], "param": sib_meta[1]},
        "naturalisation": "option1_policy_scoped_copy_restamp",
        "pool_order": "interleaved_seeded_shuffle",
        "seed": args.seed,
        "levels": [],
    }

    print(f"\nSelection: {args.selection} | Levels: {levels}")
    print("-" * 70)
    for level in levels:
        label = "full" if level == -1 else str(level)
        export, stats = build_level_export(
            base_export, eval_policies, sibling_map,
            centroid_lookup, level, args.selection, rng, args.seed)

        tot_inj = sum(s["pros"]["n_injected"] + s["cons"]["n_injected"]
                      for s in stats.values())
        tot_native = sum(s["pros"]["n_native"] + s["cons"]["n_native"]
                         for s in stats.values())
        max_dilution = (tot_native / (tot_native + tot_inj)
                        if tot_native + tot_inj else 1.0)
        proj_lower = NATIVE_BASE_RATE * max_dilution

        out_path = os.path.join(
            args.out_dir,
            f"neo4j_export_distractors_{args.selection}_L{label}.json")
        stats_path = os.path.join(
            args.out_dir,
            f"distractor_stats_{args.selection}_L{label}.json")
        save_json(export, out_path)
        save_json(stats, stats_path)

        print(f"  L{label:>4}: injected {tot_inj:>6} (native {tot_native}); "
              f"base-rate >= {proj_lower:.3f} (worst case; confirm by re-annotation)")
        print(f"         -> {out_path}")

        manifest["levels"].append({
            "level": label, "export": out_path, "stats": stats_path,
            "total_injected": tot_inj, "total_native": tot_native,
            "projected_base_rate_lower_bound": round(proj_lower, 4),
        })

    manifest_path = os.path.join(args.out_dir, f"manifest_{args.selection}.json")
    save_json(manifest, manifest_path)
    print("-" * 70)
    print(f"Manifest -> {manifest_path}")
    print("\nSANITY GATE: re-run the pipeline on the L0 export FIRST — it is")
    print("byte-identical to native and MUST reproduce current results.")
    print("Then verify with verify_interleave.py that distractors fall within")
    print("the first 200 of each pool. THEN re-run the sweeps.")


if __name__ == "__main__":
    main()