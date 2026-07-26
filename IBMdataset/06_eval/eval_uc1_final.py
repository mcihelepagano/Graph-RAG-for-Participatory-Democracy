"""
eval_uc1_grid_lean.py
======================
Standalone grid evaluation for UC1, computing ONLY the 5 metrics the
paper actually reports: nDCG@k (exponential gain), MeanRel@k, EILD,
Distractor-rate@k, Conflict density. No Precision, no Recall, no
S-Recall, no linear-gain nDCG, no rubric judge, no pairwise judge.

WHY STANDALONE (not "import eval_uc1 as E" like eval_uc1_grid.py does).
    eval_uc1.py has heavy import-time side effects (get_ollama(), NLI
    model load) that are irrelevant here since the rubric judge is not
    being computed. This script only imports from common.py (lightweight
    utilities) and reimplements the 5 metric functions directly, copied
    verbatim from eval_uc1.py's nDCG / MeanRel / EILD / distractor-rate /
    conflict-density implementations, so results are numerically
    identical to eval_uc1.py's own numbers at any shared cell.

    It also does NOT depend on eval_uc1.py's OUT_ROOT constant, which
    currently has a typo ("hard_poolsllpllpkok") -- worth fixing in
    eval_uc1.py separately, but irrelevant to this script since output
    paths are defined fresh below.

WHAT IT READS
    Per-cell retrieval files at:
        <SWEEP_CONFIGS_ROOT>/pool<P>_k<K>/<system>.json
    (produced by sweep_uc1_k.py -- same files eval_uc1_grid.py reads).
    SimpleRAG is generated fresh per-k from the full corpus, cached by k.

WHAT IT WRITES
    <GRID_OUT_ROOT>/grid_summary.json   -- {cell: {system: metrics}}
    <GRID_OUT_ROOT>/grid_summary.csv    -- long form, one row per (cell, system)

ALWAYS FULLY RECOMPUTES on every run (no checkpoint/skip logic) --
run it fresh after regenerating any system's retrieval files.

Usage:
    Edit the CONFIG block below to point at the hardened-pool config
    you're evaluating (isolated / with_distractor_edges / no_dd), then:

    python eval_uc1_grid_lean.py
"""

import os
import csv
import json
from itertools import product

import numpy as np

from policies import POLICIES
from common import (load_json, save_json, embed_texts, nli_label_batch,
                    SEMANTIC_MATCH_THRESHOLD, load_corpus,
                    simple_rag_retrieve)


# ============================================================
# CONFIG -- edit per hardened-pool config being evaluated.
# ============================================================
SWEEP_CONFIGS_ROOT = "results/sweep_uc1/configs"
GRID_OUT_ROOT       = "results/eval_uc1/native/grid_lean"   # pick any output name you like, this is new
NEO4J_EXPORT        = "data/neo4j_export_with_new_edges.json"

GT_FILE             = "data/ground_truth_nearest_L100.json"
DISTRACTOR_MANIFEST = os.environ.get(
    "DISTRACTOR_MANIFEST", "data/distractor_manifest_nearest_L100.json")


POOL_SIZES = [20, 50, 100, 200]
TOP_KS     = [5, 10, 15, 20, 30]
RELEVANT_GRADE = 2

SYSTEMS = ["System A", "System B", "Baseline A", "Baseline B", "SimpleRAG"]

os.makedirs(GRID_OUT_ROOT, exist_ok=True)


# ============================================================
# GROUND TRUTH + DISTRACTOR MANIFEST (loaded once, shared across cells)
# ============================================================
def load_ground_truth():
    if not os.path.exists(GT_FILE):
        raise FileNotFoundError(f"Ground truth not found: {GT_FILE}")
    gt = load_json(GT_FILE)
    print(f"  Loaded ground truth: {len(gt)} policies")
    return gt


def load_distractor_manifest():
    if not os.path.exists(DISTRACTOR_MANIFEST):
        print(f"  Distractor manifest not found ({DISTRACTOR_MANIFEST}); "
              f"distractor-rate@k will be None.")
        return None
    man = load_json(DISTRACTOR_MANIFEST)
    prepared = {}
    for policy, d in man.items():
        prepared[policy] = {
            "pros": {t.strip() for t in d.get("pros", [])},
            "cons": {t.strip() for t in d.get("cons", [])},
        }
    return prepared


# ============================================================
# MATCHING -- exact text first, semantic fallback (verbatim from eval_uc1.py)
# ============================================================
_gt_cache = {}


def _dedup_grades(graded_map):
    out = {}
    for t, g in graded_map.items():
        s = t.strip()
        g = float(g)
        if s not in out or g > out[s]:
            out[s] = g
    return out


def _gt_entry(policy, stance, graded_map):
    key = (policy, stance)
    if key not in _gt_cache:
        lookup = _dedup_grades(graded_map)
        texts  = list(lookup.keys())
        grades = np.array([lookup[t] for t in texts], dtype=float)
        _gt_cache[key] = {"texts": texts, "grades": grades,
                          "lookup": lookup, "embs": None}
    return _gt_cache[key]


def match_retrieved(retrieved, graded_map, policy, stance,
                    threshold=SEMANTIC_MATCH_THRESHOLD):
    if not retrieved:
        return []
    if not graded_map:
        return [{"grade": 0.0, "match": "none", "gt_text": None}
                for _ in retrieved]

    gt = _gt_entry(policy, stance, graded_map)
    matches, fallback_idx = [], []
    for i, t in enumerate(retrieved):
        s = t.strip()
        if s in gt["lookup"]:
            matches.append({"grade": gt["lookup"][s], "match": "exact",
                            "gt_text": s})
        else:
            matches.append(None)
            fallback_idx.append(i)

    if fallback_idx:
        if gt["embs"] is None:
            gt["embs"] = embed_texts(gt["texts"])
        r_embs = embed_texts([retrieved[i] for i in fallback_idx])
        for row, i in zip(r_embs, fallback_idx):
            sims = gt["embs"] @ row
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                matches[i] = {"grade": float(gt["grades"][j]),
                              "match": "semantic",
                              "gt_text": gt["texts"][j].strip()}
            else:
                matches[i] = {"grade": 0.0, "match": "none", "gt_text": None}
    return matches


# ============================================================
# THE 5 REPORTED METRICS (verbatim from eval_uc1.py)
# ============================================================
def _gain_exp(g):
    return 2.0 ** g - 1.0


def ndcg_from_matches(matches, graded_map, k, gain=_gain_exp):
    if not matches or not graded_map:
        return 0.0
    dcg = 0.0
    for rank, m in enumerate(matches[:k], start=1):
        dcg += gain(m["grade"]) / np.log2(rank + 1)
    ideal = sorted(_dedup_grades(graded_map).values(), reverse=True)[:k]
    idcg = 0.0
    for rank, g in enumerate(ideal, start=1):
        idcg += gain(float(g)) / np.log2(rank + 1)
    return dcg / idcg if idcg > 0 else 0.0


def mean_relevance_from_matches(matches, k):
    if not matches or k <= 0:
        return 0.0
    return float(sum(m["grade"] for m in matches[:k]) / k)


def intra_list_distance(texts):
    if len(texts) < 2:
        return 0.0
    embs = embed_texts(texts)
    sim  = embs @ embs.T
    iu   = np.triu_indices(len(embs), k=1)
    return 1.0 - float(np.mean(sim[iu]))


def relevance_constrained_ild(texts, matches, relevant_grade=RELEVANT_GRADE):
    if not texts or not matches or len(texts) != len(matches):
        return 0.0
    rel_texts = [t for t, m in zip(texts, matches)
                 if m is not None and m.get("match") != "none"
                 and m.get("grade", 0.0) >= relevant_grade]
    return intra_list_distance(rel_texts)


def distractor_rate(retrieved, distractor_set, k):
    if distractor_set is None:
        return None
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for t in top if t.strip() in distractor_set)
    return hits / len(top)


def conflict_density(pros, cons):
    if not pros or not cons:
        return 0.0
    premises, hypotheses = [], []
    for p in pros:
        for c in cons:
            premises.append(p)
            hypotheses.append(c)
    probs   = nli_label_batch(premises, hypotheses)
    contras = [pr[0] for pr in probs]
    return float(np.mean(contras)) if contras else 0.0


def _pair_mean(a, b):
    vals = [v for v in (a, b) if v is not None]
    return float(np.mean(vals)) if vals else None


# ============================================================
# RETRIEVAL LOADING (per cell; SimpleRAG cached per-k)
# ============================================================
_simplerag_cache = {}


def simplerag_for_k(k):
    if k in _simplerag_cache:
        return _simplerag_cache[k]
    if not os.path.exists(NEO4J_EXPORT):
        raise FileNotFoundError(f"Neo4j export not found: {NEO4J_EXPORT}")
    corpus = load_corpus(NEO4J_EXPORT)
    recs = {}
    for policy in POLICIES:
        pros, cons = simple_rag_retrieve(policy, corpus, k)
        recs[policy] = {"policy": policy, "retrieved_pros": pros,
                        "retrieved_cons": cons}
    _simplerag_cache[k] = recs
    return recs


def load_retrievals_for_cell(pool_size, top_k):
    cfg_dir = os.path.join(SWEEP_CONFIGS_ROOT, f"pool{pool_size}_k{top_k}")
    out = {}
    for system in SYSTEMS:
        if system == "SimpleRAG":
            out[system] = simplerag_for_k(top_k)
            continue
        path = os.path.join(cfg_dir, system.lower().replace(" ", "_") + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing retrieval file for pool{pool_size}_k{top_k}: {path}")
        out[system] = {r["policy"]: r for r in load_json(path)}
    return out


# ============================================================
# PER-CELL COMPUTATION
# ============================================================
def compute_cell(retrievals, ground_truth, k, distractor_manifest):
    cell_summary = {}
    for system in SYSTEMS:
        ndcg_pros_l, ndcg_cons_l = [], []
        meanrel_pros_l, meanrel_cons_l = [], []
        eild_l = []
        dr_pros_l, dr_cons_l = [], []
        cd_l = []

        for policy in POLICIES:
            if policy not in retrievals[system]:
                continue
            rec  = retrievals[system][policy]
            pros = rec.get("retrieved_pros", [])
            cons = rec.get("retrieved_cons", [])
            gt   = ground_truth.get(policy, {})
            g_pros = gt.get("graded_pros", {})
            g_cons = gt.get("graded_cons", {})

            if distractor_manifest is None:
                d_pros = d_cons = None
            else:
                dm = distractor_manifest.get(policy, {})
                d_pros = dm.get("pros")
                d_cons = dm.get("cons")

            m_pros = match_retrieved(pros, g_pros, policy, "pros")
            m_cons = match_retrieved(cons, g_cons, policy, "cons")

            ndcg_pros_l.append(ndcg_from_matches(m_pros, g_pros, k))
            ndcg_cons_l.append(ndcg_from_matches(m_cons, g_cons, k))
            meanrel_pros_l.append(mean_relevance_from_matches(m_pros, k))
            meanrel_cons_l.append(mean_relevance_from_matches(m_cons, k))
            eild_l.append(relevance_constrained_ild(
                pros + cons, m_pros + m_cons))

            dr_p = distractor_rate(pros, d_pros, k)
            dr_c = distractor_rate(cons, d_cons, k)
            if dr_p is not None:
                dr_pros_l.append(dr_p)
            if dr_c is not None:
                dr_cons_l.append(dr_c)

            cd_l.append(conflict_density(pros, cons))

        cell_summary[system] = {
            "ndcg_mean": _pair_mean(
                float(np.mean(ndcg_pros_l)) if ndcg_pros_l else None,
                float(np.mean(ndcg_cons_l)) if ndcg_cons_l else None),
            "meanrel_mean": _pair_mean(
                float(np.mean(meanrel_pros_l)) if meanrel_pros_l else None,
                float(np.mean(meanrel_cons_l)) if meanrel_cons_l else None),
            "eild_all": float(np.mean(eild_l)) if eild_l else None,
            "distractor_rate_mean": _pair_mean(
                float(np.mean(dr_pros_l)) if dr_pros_l else None,
                float(np.mean(dr_cons_l)) if dr_cons_l else None),
            "conflict_density": float(np.mean(cd_l)) if cd_l else None,
            "n_policies": len(ndcg_pros_l),
        }
    return cell_summary

# ============================================================
# JUDGE B — RUBRIC SCORING (1-5 per set, 3 axes) — DISABLED HERE
# ============================================================
# Extracted verbatim from eval_uc1.py (v2) for reference. Left commented
# out so this lean script keeps its stated design goal: no get_ollama()
# import-time side effect unless the rubric judge is actually wanted.
#
# To re-enable:
#   1. Add to imports: get_ollama, load_checkpoint, parse_json_object, mean_std
#      (from common.py) and `import time` at the top of this file.
#   2. Uncomment JUDGE_MODEL, F_RUBRIC, and `ollama = get_ollama()` below.
#   3. Uncomment the block below and call run_rubric(retrievals) from main(),
#      merging its per_system output into compute_cell's per-system dict
#      (or write it to its own file — this script has no rubric key in
#      grid_summary.json today, so aggregation code would need to change).
#
# JUDGE_MODEL = "llama3.3:70b"
# F_RUBRIC    = os.path.join(GRID_OUT_ROOT, "rubric_scores.json")
# ollama = get_ollama()
#
# RUBRIC_SYSTEM = """You are an expert debate analyst scoring ONE set of
# arguments for a policy debate.
#
# Score the set on each of the following criteria, from 1 (poor) to 5
# (excellent). Score each criterion independently and use the full range.
#
#   conflict_engagement : do the PRO and CON arguments directly engage
#       each other — attacking the same premises and dimensions — rather
#       than talking past each other?
#   redundancy_avoidance : does each argument add a distinct point, with
#       little repetition within the set?
#   balanced_writing_aid : would this set equip a citizen to write a
#       balanced, well-informed position paper covering the strongest
#       pros, strongest cons, and the underlying value tensions?
#
# Return ONLY this JSON, no markdown:
# {
#   "conflict_engagement": 1-5,
#   "redundancy_avoidance": 1-5,
#   "balanced_writing_aid": 1-5
# }"""
#
# RUBRIC_KEYS = ["conflict_engagement", "redundancy_avoidance",
#                "balanced_writing_aid"]
#
#
# def build_rubric_prompt(policy, retrieval_set):
#     p = "\n".join(f"  {i+1}. {a}"
#                   for i, a in enumerate(retrieval_set.get("retrieved_pros", [])))
#     c = "\n".join(f"  {i+1}. {a}"
#                   for i, a in enumerate(retrieval_set.get("retrieved_cons", [])))
#     return (f'Policy: "{policy}"\n\n'
#             f"PRO arguments:\n{p}\n\nCON arguments:\n{c}\n\n"
#             "Score this set on the three criteria. Return only the JSON.")
#
#
# def judge_rubric(policy, retrieval_set, retries=2):
#     prompt = build_rubric_prompt(policy, retrieval_set)
#     for attempt in range(retries + 1):
#         try:
#             r = ollama.chat(
#                 model=JUDGE_MODEL,
#                 messages=[
#                     {"role": "system", "content": RUBRIC_SYSTEM},
#                     {"role": "user",   "content": prompt},
#                 ],
#                 options={"temperature": 0.0, "num_predict": 200,
#                          "num_ctx": 8192},
#             )
#             obj = parse_json_object(r["message"]["content"],
#                                     required_keys=RUBRIC_KEYS)
#             if obj is not None:
#                 clean = {}
#                 for kk in RUBRIC_KEYS:
#                     try:
#                         v = float(obj[kk])
#                     except (ValueError, TypeError):
#                         v = None
#                     clean[kk] = v if (v is not None and 1 <= v <= 5) else None
#                 if all(clean[kk] is not None for kk in RUBRIC_KEYS):
#                     return clean
#             time.sleep(1)
#         except Exception as e:
#             print(f"      rubric error (try {attempt+1}): {e}")
#             time.sleep(2)
#     return None
#
#
# def run_rubric(retrievals):
#     print("\n=== JUDGE B: rubric scoring (1-5 per set) ===")
#     results = load_checkpoint(F_RUBRIC)
#
#     for system in SYSTEMS:
#         if system not in results:
#             results[system] = {"per_policy": {}}
#         done = set(results[system]["per_policy"].keys())
#         print(f"  {system}: {len(done)}/{len(POLICIES)}")
#
#         for policy in POLICIES:
#             if policy in done or policy not in retrievals[system]:
#                 continue
#             scores = judge_rubric(policy, retrievals[system][policy])
#             if scores is not None:
#                 results[system]["per_policy"][policy] = scores
#             save_json(results, F_RUBRIC)
#             time.sleep(SLEEP_SEC)
#
#         # Aggregate
#         per_policy = results[system]["per_policy"]
#         for kk in RUBRIC_KEYS:
#             vals = [v[kk] for v in per_policy.values() if v.get(kk) is not None]
#             m, s = mean_std(vals)
#             results[system][f"mean_{kk}"] = m
#             results[system][f"std_{kk}"]  = s
#         save_json(results, F_RUBRIC)
#
#     return results

# ============================================================
# MAIN
# ============================================================
def main():
    configs = [(p, k) for p, k in product(POOL_SIZES, TOP_KS) if k <= p]
    print("=" * 68)
    print("UC1 LEAN GRID EVAL -- 5 reported metrics only")
    print(f"  Cells      : {len(configs)}  ({POOL_SIZES} x {TOP_KS}, k<=p)")
    print(f"  Systems    : {SYSTEMS}")
    print(f"  Sweep root : {SWEEP_CONFIGS_ROOT}")
    print(f"  Out        : {GRID_OUT_ROOT}")
    print("=" * 68)

    print("\nLoading ground truth + distractor manifest...")
    ground_truth = load_ground_truth()
    distractor_manifest = load_distractor_manifest()

    grid = {}
    long_rows = []

    for pool_size, top_k in configs:
        cell_key = f"pool{pool_size}_k{top_k}"
        print(f"\n--- {cell_key} ---")
        retrievals = load_retrievals_for_cell(pool_size, top_k)
        cell_summary = compute_cell(retrievals, ground_truth, top_k,
                                     distractor_manifest)
        grid[cell_key] = cell_summary

        for system in SYSTEMS:
            r = cell_summary[system]
            def fmt(v):
                return f"{v:.3f}" if v is not None else "None"
            print(f"    {system:11s} nDCG={fmt(r['ndcg_mean'])}  "
                  f"MeanRel={fmt(r['meanrel_mean'])}  "
                  f"EILD={fmt(r['eild_all'])}  "
                  f"distr={fmt(r['distractor_rate_mean'])}  "
                  f"conf={fmt(r['conflict_density'])}")

        for system in SYSTEMS:
            long_rows.append({"pool_size": pool_size, "top_k": top_k,
                              "system": system, **cell_summary[system]})

    save_json(grid, os.path.join(GRID_OUT_ROOT, "grid_summary.json"))

    csv_path = os.path.join(GRID_OUT_ROOT, "grid_summary.csv")
    fieldnames = ["pool_size", "top_k", "system", "ndcg_mean",
                  "meanrel_mean", "eild_all", "distractor_rate_mean",
                  "conflict_density", "n_policies"]
    long_rows.sort(key=lambda r: (r["pool_size"], r["top_k"], r["system"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(long_rows)

    print(f"\nDone. {len(configs)} cells evaluated, 5 metrics each.")
    print(f"  grid_summary.json -> {os.path.join(GRID_OUT_ROOT, 'grid_summary.json')}")
    print(f"  grid_summary.csv  -> {csv_path}")


if __name__ == "__main__":
    main()