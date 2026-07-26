"""
annotate_distractors.py
=======================
INCREMENTAL ground-truth annotation for the hardened pools produced by
build_hard_pools.py.

WHY THIS EXISTS (read before running)
-------------------------------------
The native ground truth (data/ground_truth.json) already contains the
graded labels for every native pool argument. Those labels DO NOT CHANGE
when distractors are injected next to them, so we reuse them verbatim.

What is missing is a grade for each INJECTED DISTRACTOR: a distractor is a
sibling-policy argument that was never judged against the target policy, so
it is unjudged. The relevance metrics need a grade for every pool member, so
the distractors MUST be judged. This script judges ONLY the distractors and
merges their grades into the existing ground truth.

This is incremental, not from-scratch:
  - Native grades: reused unchanged (frozen).
  - Distractor grades: produced here, by the SAME judging path
    (annotate_stance + resolve_consensus from generate_ground_truth.py,
    same dual-LLM setup, same MIN-rule consensus), so the distractor labels
    are methodologically identical to the native ones.

WHAT IT DOES NOT DO
-------------------
  - It does NOT re-judge natives (that would waste compute and, because the
    LLM annotators are not perfectly deterministic, would perturb the
    existing labels and break the level-0 == native control).
  - It does NOT assume distractors are irrelevant. Sibling-policy arguments
    are topically close; some genuinely apply to the target policy. Stamping
    them grade-0 would fabricate the base rate instead of measuring it. They
    are judged on their merits.

KAPPA REPORTING (native vs distractor SPLIT)
--------------------------------------------
Inter-annotator agreement is reported SEPARATELY for natives and
distractors. The native kappa is the existing ~0.590 (unchanged, reused).
The distractor kappa is computed fresh over the distractor judgments. This
makes the story transparent: "native ground-truth reliability is unchanged;
distractor judging achieved kappa = X". A pooled kappa is NOT used as the
headline because natives and distractors are different judging difficulties.

OUTPUT (per level)
------------------
  data/ground_truth_<selection>_L<level>.json
      Drop-in replacement for data/ground_truth.json. Same schema; each
      policy's graded_pros / graded_cons contain natives + judged
      distractors merged. eval_uc1.py reads graded_pros / graded_cons
      directly (eval_uc1.py lines 463-464), so to evaluate a level you
      point eval_uc1 at this file (copy it to data/ground_truth.json, or
      add a GT_FILE override).
  data/ground_truth_<selection>_L<level>_report.json
      Realised base rate (overall + per policy) and the native-vs-distractor
      kappa split for the level.

The level-0 export contains NO distractors, so this script is a no-op there:
its merged GT equals the native GT (the control).

Usage:
  python annotate_distractors.py --level 50
  python annotate_distractors.py --level 100 --selection random
  python annotate_distractors.py --all-levels        # process every level in the manifest
"""

import os
import json
import argparse
import numpy as np

from common import load_json, save_json

# Reuse the EXACT judging path used for the natives. Importing from
# generate_ground_truth.py guarantees the distractor labels are produced by
# the identical prompt, batching, consensus and agreement code.
import generate_ground_truth as ggt


# ============================================================
# CONFIG
# ============================================================
NATIVE_GT_FILE   = "data/ground_truth.json"
HARD_POOL_DIR    = "data/hard_pools"
OUT_DIR          = "data"               # where merged per-level GTs are written
RELEVANT_GRADE   = ggt.RELEVANT_GRADE_THRESHOLD   # grade >= this == relevant


# ============================================================
# HELPERS
# ============================================================
def manifest_path(selection):
    return os.path.join(HARD_POOL_DIR, f"manifest_{selection}.json")


def level_export_path(selection, level_label):
    return os.path.join(
        HARD_POOL_DIR,
        f"neo4j_export_distractors_{selection}_L{level_label}.json")


def distractor_records(export, policy, stance_key):
    """Return the list of injected-distractor argument records for one
    policy/stance, in pool order. Each record has text + embedding and is
    tagged is_distractor (from build_hard_pools.py)."""
    args = export["arguments"]
    recs = []
    for aid in export["policies"].get(policy, {}).get(stance_key, []):
        a = args.get(aid)
        if a is not None and a.get("is_distractor"):
            recs.append(a)
    return recs


def judge_distractors(policy, stance_label, records):
    """Judge a list of distractor records with BOTH annotators and resolve
    the consensus, exactly as generate_ground_truth does for natives.

    Returns (consensus_grades, primary_grades, secondary_grades), each a
    list aligned with `records` (None sentinel where a model left an item
    unjudged after retries).
    """
    if not records:
        return [], [], []

    g_primary = ggt.annotate_stance(
        policy, stance_label, records, ggt.ANNOTATOR_PRIMARY)
    g_secondary = ggt.annotate_stance(
        policy, stance_label, records, ggt.ANNOTATOR_SECONDARY)

    # annotate_stance returns None if a whole batch failed all retries.
    # Treat a hard failure as all-unjudged rather than crashing the level;
    # the sentinel path downstream excludes unjudged items from metrics.
    if g_primary is None:
        g_primary = [None] * len(records)
    if g_secondary is None:
        g_secondary = [None] * len(records)

    consensus = []
    for gp, gs in zip(g_primary, g_secondary):
        if gp is None and gs is None:
            consensus.append(None)
        elif gp is None:
            consensus.append(gs)
        elif gs is None:
            consensus.append(gp)
        else:
            consensus.append(ggt.resolve_consensus(gp, gs))
    return consensus, g_primary, g_secondary


def kappa_split(primary, secondary):
    """Quadratic-weighted + binary Cohen's kappa over a distractor set,
    using generate_ground_truth's own functions so the statistic matches
    the native computation. Returns dict (None entries if too few pairs)."""
    qwk = ggt.quadratic_weighted_kappa(primary, secondary)
    kbin = ggt.binary_cohen_kappa(primary, secondary)
    return {"kappa_quadratic": qwk, "kappa_binary": kbin,
            "n_pairs": sum(1 for a, b in zip(primary, secondary)
                           if a is not None and b is not None)}


# ============================================================
# CORE: merge one level
# ============================================================
def process_level(selection, level_label):
    """Judge distractors for one level and write a merged ground-truth file."""
    native_gt = load_json(NATIVE_GT_FILE)
    export    = load_json(level_export_path(selection, level_label))

    # Deep-copy the native GT; we only ADD to graded_pros / graded_cons.
    merged = json.loads(json.dumps(native_gt))

    # Accumulators for the level's native-vs-distractor kappa split and the
    # realised base rate.
    distractor_primary_all, distractor_secondary_all = [], []
    per_policy_report = []
    tot_pool, tot_rel = 0, 0

    policies = list(export["policies"].keys())
    # Only policies that are actually in the native GT are evaluation policies.
    eval_policies = [p for p in policies if p in merged]

    for p in eval_policies:
        rec = merged[p]
        p_dist_primary, p_dist_secondary = [], []

        for stance_key, graded_key, stance_label in (
                ("pros", "graded_pros", "PRO"),
                ("cons", "graded_cons", "CON")):

            recs = distractor_records(export, p, stance_key)
            if not recs:
                continue

            consensus, gp, gs = judge_distractors(p, stance_label, recs)

            graded_map = rec.setdefault(graded_key, {})
            relevant_list_key = ("relevant_pros" if stance_key == "pros"
                                 else "relevant_cons")
            relevant_list = rec.setdefault(relevant_list_key, [])

            for r, c in zip(recs, consensus):
                if c is None:
                    continue  # unjudged sentinel: exclude (do NOT fabricate 0)
                text = r["text"].strip()
                # MAX on duplicate stripped text — same tie-break as the
                # native pipeline (eval_uc1 _dedup_grades).
                if text in graded_map:
                    graded_map[text] = max(graded_map[text], int(c))
                else:
                    graded_map[text] = int(c)
                if int(c) >= RELEVANT_GRADE and text not in relevant_list:
                    relevant_list.append(text)

            p_dist_primary.extend(gp)
            p_dist_secondary.extend(gs)

        # Per-policy distractor kappa + base rate over the MERGED pool.
        if p_dist_primary:
            distractor_primary_all.extend(p_dist_primary)
            distractor_secondary_all.extend(p_dist_secondary)

        pool_n = len(rec.get("graded_pros", {})) + len(rec.get("graded_cons", {}))
        rel_n  = (sum(1 for g in rec.get("graded_pros", {}).values() if g >= RELEVANT_GRADE)
                  + sum(1 for g in rec.get("graded_cons", {}).values() if g >= RELEVANT_GRADE))
        tot_pool += pool_n
        tot_rel  += rel_n
        per_policy_report.append({
            "policy": p,
            "pool_size": pool_n,
            "relevant": rel_n,
            "base_rate": round(rel_n / pool_n, 4) if pool_n else None,
            "n_distractors_judged": len(p_dist_primary),
            "distractor_kappa": (kappa_split(p_dist_primary, p_dist_secondary)
                                 if p_dist_primary else None),
        })

    # Level-wide distractor kappa split.
    distractor_kappa = (kappa_split(distractor_primary_all,
                                    distractor_secondary_all)
                        if distractor_primary_all else None)

    # Write merged GT (drop-in for data/ground_truth.json).
    gt_out = os.path.join(OUT_DIR,
                          f"ground_truth_{selection}_L{level_label}.json")
    save_json(merged, gt_out)

    report = {
        "selection": selection,
        "level": level_label,
        "native_gt_file": NATIVE_GT_FILE,
        "level_export": level_export_path(selection, level_label),
        "realised_base_rate_overall": round(tot_rel / tot_pool, 4) if tot_pool else None,
        "total_pool": tot_pool,
        "total_relevant": tot_rel,
        "native_kappa_note": ("native labels reused unchanged from "
                              f"{NATIVE_GT_FILE}; see its report for native kappa"),
        "distractor_kappa_level": distractor_kappa,
        "consensus_rule": ggt.CONSENSUS_RULE,
        "annotator_primary": ggt.ANNOTATOR_PRIMARY,
        "annotator_secondary": ggt.ANNOTATOR_SECONDARY,
        "per_policy": sorted(per_policy_report, key=lambda r: r["base_rate"] or 1.0),
    }
    rep_out = os.path.join(OUT_DIR,
                           f"ground_truth_{selection}_L{level_label}_report.json")
    save_json(report, rep_out)

    print(f"  L{level_label}: realised base rate "
          f"{report['realised_base_rate_overall']} over {tot_pool} pool args; "
          f"distractor kappa(quad)="
          f"{distractor_kappa['kappa_quadratic'] if distractor_kappa else 'n/a'}")
    print(f"         GT     -> {gt_out}")
    print(f"         report -> {rep_out}")
    return gt_out


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["nearest", "random"], default="nearest")
    ap.add_argument("--level", help="level label (e.g. 50, 100, 200, full)")
    ap.add_argument("--all-levels", action="store_true",
                    help="process every level listed in the manifest")
    args = ap.parse_args()

    print("=" * 70)
    print("ANNOTATE DISTRACTORS — incremental ground truth (natives frozen)")
    print("=" * 70)

    if not os.path.exists(NATIVE_GT_FILE):
        raise SystemExit(f"Native ground truth not found: {NATIVE_GT_FILE}. "
                         f"Run generate_ground_truth.py first.")

    if args.all_levels:
        mpath = manifest_path(args.selection)
        if not os.path.exists(mpath):
            raise SystemExit(f"Manifest not found: {mpath}. "
                             f"Run build_hard_pools.py --selection {args.selection} first.")
        manifest = load_json(mpath)
        levels = [lvl["level"] for lvl in manifest["levels"]]
    elif args.level:
        levels = [args.level]
    else:
        raise SystemExit("Provide --level <L> or --all-levels.")

    print(f"Selection: {args.selection} | Levels: {levels}\n")
    for lvl in levels:
        if lvl == "0":
            print(f"  L0: no distractors — merged GT equals native GT (control). Skipping judge.")
            # Still emit a copy so the eval workflow is uniform.
            native = load_json(NATIVE_GT_FILE)
            save_json(native, os.path.join(
                OUT_DIR, f"ground_truth_{args.selection}_L0.json"))
            continue
        process_level(args.selection, lvl)

    print("\nNEXT: to evaluate a level, point eval_uc1 at its GT file")
    print("  (copy ground_truth_<sel>_L<lvl>.json to data/ground_truth.json,")
    print("   or add a GT_FILE override). Run the matching level EXPORT through")
    print("   sweep_uc1_k first so retrievals come from the hardened pool.")


if __name__ == "__main__":
    main()