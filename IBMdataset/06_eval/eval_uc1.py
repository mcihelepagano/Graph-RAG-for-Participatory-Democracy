"""
eval_uc1.py  (v2)
=================
Stage 2 evaluation for UC1 (Conflict Map) at operating point pool=200, k=5.

CHANGES IN v2 (relative to the previous eval_uc1.py), each motivated by a
diagnosed issue or a metric-design review:

  CHANGE 6 — EXACT-TEXT-FIRST RELEVANCE MATCHING.
    Ground truth v5 judges the FULL pool, keyed by raw argument text, and
    every system retrieves verbatim texts from that pool (except B/C's
    multi-hop expansion, which can leave it). Fuzzy matching against an
    exactly-judged item only adds noise: a retrieved pool member could be
    assigned the grade of a *different* nearby argument. v2 therefore
    resolves each retrieved item by EXACT text lookup first, and falls
    back to semantic matching (cosine >= SEMANTIC_MATCH_THRESHOLD,
    single best match) ONLY for items with no exact key — i.e. genuine
    out-of-pool retrievals. Per-system match-type fractions
    (exact / semantic / none) are reported as a diagnostic so the
    in-pool assumption is verifiable, not assumed.

  CHANGE 7 — RECALL@k ADDED.
    Of all grade>=2 arguments in the judged pool, what fraction did the
    system surface in its top-k? Deterministic, no new LLM passes.
    Counts UNIQUE matched ground-truth arguments, so retrieving two
    paraphrases of the same relevant argument scores once (a deliberate
    contrast with Precision, which they would inflate). Denominator is
    pool-relative; out-of-pool retrievals cannot contribute, consistent
    with how Precision treats them. None (excluded from aggregation)
    when a stance has no grade>=2 arguments.

  CHANGE 8 — MEAN RELEVANCE@k ADDED.
    Mean matched grade of the top-k (0-3 scale). Rank-insensitive,
    directly interpretable companion to nDCG; the natural y-axis against
    ILD for the relevance-diversity frontier figure. Divides by k, so
    under-retrieval is penalized (intentional).

  CHANGE 9 — LINEAR-GAIN nDCG ROBUSTNESS CHECK.
    Primary nDCG keeps exponential gain (2^g - 1). With the MIN-rule
    consensus making grade-3s rare, exponential gain lets a handful of
    items dominate (gains 0/1/3/7 vs 0/1/2/3). v2 additionally reports
    linear-gain nDCG (gain = g) from the SAME matches; if the system
    ranking is stable under both, that is an appendix sentence — if not,
    we want to know before a reviewer does.

  CHANGE 10 — PER-POLICY CSV EXPORT.
    The thesis-critical analysis is per-policy (cross-referenced with
    EQUIVALENT-community modularity and argument-count terciles), not the
    aggregate mean. v2 writes results/eval_uc1/per_policy.csv with one
    row per (system, policy) and every deterministic metric.

  CHANGE 11 — ONE EMBEDDING PASS PER (POLICY, STANCE).
    Ground-truth embeddings are computed once per (policy, stance) and
    shared across systems and metrics (previously each metric re-embedded
    the same ground truth per system).

  CHANGE 12 — DETERMINISTIC CHECKPOINT SCHEMA VERSIONING.
    deterministic.json now carries "_schema": 2. A checkpoint from the
    old metric set is renamed to *.v1.bak and recomputed, instead of
    silently aggregating missing keys to None. (Stale checkpoints have
    bitten this pipeline before.)

  CHANGE 13 — S-RECALL@k ADDED (closes the four-metric-set gap).
    The agreed relevance set is nDCG-exp (primary) + Precision@k
    (interpretable / distractor-rejection) + MeanRel@k (graded,
    rank-free) + S-Recall@k (subtopic coverage); plain Recall@k is kept
    for continuity but demoted, since its denominator is structurally
    tiny/misleading on this hardened pool. S-Recall needs a notion of
    "subtopic" that the ground truth does not supply directly (GT only
    stores {text, grade}), so subtopics are operationalised here as
    embedding clusters over the judged pool's relevant (grade >=
    RELEVANT_GRADE) arguments per (policy, stance): agglomerative,
    average-linkage, cosine distance, threshold SUBTOPIC_DISTANCE_THRESHOLD
    (no fixed cluster count — the number of subtopics is not known a
    priori and should vary by policy). S-Recall@k is then the fraction of
    those subtopic clusters represented at least once in the top-k.
    This is a defensible but NOT the only possible operationalisation —
    flag the clustering threshold to Rafael before treating S-Recall
    numbers as final; a sensitivity check across 2-3 thresholds is cheap
    and worth doing before the report is frozen.

Carried over from v1: graded semantic nDCG (Järvelin & Kekäläinen 2002),
nDCG-consistent single-best-match Precision@k (grade >= 2), ILD
(Carbonell & Goldstein 1998), conflict density (NLI), rubric judge,
binary pairwise judge DISABLED (RUN_PAIRWISE=False; ~73-97% tie rates,
no discriminative signal), stance-aware SimpleRAG baseline shared with
UC2.

DETERMINISTIC METRICS (per stance unless noted)
  1. nDCG@k        — graded, exponential gain (primary) + linear gain
                     (robustness), exact-first matching.
  2. Precision@k   — single best match, hit iff matched grade >= 2.
  3. Recall@k      — unique grade>=2 pool arguments surfaced in top-k
                     (kept for continuity; demoted in favor of S-Recall).
  4. S-Recall@k    — fraction of embedding-cluster "subtopics" among the
                     judged pool's relevant arguments represented in top-k.
  5. MeanRel@k     — mean matched grade of top-k (0-3).
  6. ILD           — intra-list distance, reference-free (pros/cons/all).
  7. Conflict density — NLI contradiction over pro x con pairs.
  8. Match diagnostics — exact / semantic / unmatched fractions.

LLM JUDGES
  A. Binary double-sided pairwise (3 axes) — DISABLED, scaffolding kept.
  B. Rubric scoring 1-5 per set on the same 3 axes.

INPUT
  results/sweep_uc1/configs/pool200_k5/<system>.json
  data/ground_truth.json

OUTPUT
  results/eval_uc1/
    pairwise_axes.json   (only if RUN_PAIRWISE)
    rubric_scores.json
    deterministic.json
    per_policy.csv       (NEW)
    summary.json
    summary.csv

NOTE ON RE-RUNS: the deterministic metrics changed in v2, so delete (or
let CHANGE 12 auto-archive) results/eval_uc1/deterministic.json,
summary.json and summary.csv before running. rubric_scores.json is
unaffected and will be reused from checkpoint.

Usage:
    python eval_uc1.py
"""

import os
import time
import csv
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from policies import POLICIES
from common import (load_json, save_json, load_checkpoint, get_ollama,
                    embed_texts, nli_label_batch, parse_ab, parse_json_object,
                    mean_std, fmt, SEMANTIC_MATCH_THRESHOLD,
                    load_corpus, simple_rag_retrieve)


# ============================================================
# CONFIG
# ============================================================
JUDGE_MODEL = "llama3.3:70b"
POOL_SIZE   = 200
CHOSEN_K    = 5
SLEEP_SEC   = 0.2

# Grade threshold for "relevant" in Precision@k and Recall@k.
# Grades are 0-3 (MIN-rule consensus); >= 2 is the top half.
RELEVANT_GRADE = 2

# S-Recall@k (CHANGE 13): cosine-distance threshold for agglomerative
# clustering of the judged pool's relevant arguments into "subtopics".
# 0.15 cosine distance (~0.85 cosine similarity) is a reasonable starting
# point for near-paraphrase grouping but is a genuine methodological
# choice, not a settled constant — sensitivity-check before finalizing
# for the report.
SUBTOPIC_DISTANCE_THRESHOLD = 0.15

# --- TEMPORARILY DISABLED ------------------------------------------------
# The binary double-sided pairwise judge (3 axes) produces very high tie
# rates because retrieved sets are near-equivalent, so it adds little
# discriminative signal. Disabled until/unless we move to a graded
# preference judge. The rubric judge and the deterministic metrics remain
# the UC1 evidence. Set back to True to re-enable; downstream aggregation
# handles its absence gracefully.
RUN_PAIRWISE = False
# -------------------------------------------------------------------------

CONFIG_DIR   = os.path.join("results/sweep_uc1/hard_pools_nearest_L100_no_dd/configs",
                            f"pool{POOL_SIZE}_k{CHOSEN_K}")
GT_FILE      = "data/ground_truth_nearest_L100.json"
# Distractor manifest (per-policy per-stance distractor TEXTS), emitted by
# build_distractor_manifest.py from the hardened L100 export. Enables a
# direct, provenance-based distractor-rate@k (grade cannot distinguish an
# injected distractor from a native grade-0 item). Optional: if absent,
# distractor-rate metrics are reported as None and the eval still runs.
DISTRACTOR_MANIFEST = os.environ.get(
    "DISTRACTOR_MANIFEST", "data/distractor_manifest_nearest_L100.json")
# Canonical export (unified path). VERIFY this matches the cluster copy —
# an older revision of this script pointed at data/neo4j_export_full.json.
NEO4J_EXPORT = "data/neo4j_export_l100_distractor_edges_no_dd.json"

OUT_ROOT     = "results/eval_uc1/hard_poolsllpllpkok"
F_PAIRWISE   = os.path.join(OUT_ROOT, "pairwise_axes.json")
F_RUBRIC     = os.path.join(OUT_ROOT, "rubric_scores.json")
F_DETERM     = os.path.join(OUT_ROOT, "deterministic.json")
F_PERPOLICY  = os.path.join(OUT_ROOT, "per_policy.csv")
F_SUMMARY    = os.path.join(OUT_ROOT, "summary.json")
F_CSV        = os.path.join(OUT_ROOT, "summary.csv")
# SimpleRAG retrievals for UC1 live here (generated by Phase 0).
F_SIMPLERAG  = os.path.join(OUT_ROOT, "simplerag_retrieval.json")

DETERM_SCHEMA = 5  # bump whenever the deterministic metric set changes (3: + EILD, 4: + distractor_rate, 5: + s-recall)

# SimpleRAG is the vanilla-RAG baseline (no graph). It is stance-aware,
# so it produces retrieved_pros / retrieved_cons like every other system
# and is fully comparable on every UC1 metric.
SYSTEMS     = ["System A", "System B",
               "Baseline A", "Baseline B", "SimpleRAG"]
CHALLENGERS = ["System A", "Baseline A", "Baseline B", "SimpleRAG"]

os.makedirs(OUT_ROOT, exist_ok=True)
ollama = get_ollama()


# ============================================================
# PHASE 0 — SimpleRAG retrieval (vanilla RAG, no graph)
# ============================================================
def ensure_simplerag_retrieval():
    """Generate stance-aware SimpleRAG retrievals at k=CHOSEN_K if missing.

    Uses the SAME shared implementation as UC2 (common.simple_rag_retrieve):
    embed the policy query, take top-k PRO and top-k CON by cosine
    similarity from the full corpus. No graph, no MMR, no PageRank.
    """
    if os.path.exists(F_SIMPLERAG):
        return {r["policy"]: r for r in load_json(F_SIMPLERAG)}

    print(f"\n=== PHASE 0: SimpleRAG retrieval (k={CHOSEN_K}) ===")
    if not os.path.exists(NEO4J_EXPORT):
        raise FileNotFoundError(
            f"Neo4j export not found: {NEO4J_EXPORT}\n"
            "Needed to build the SimpleRAG full-corpus baseline.")
    corpus = load_corpus(NEO4J_EXPORT)
    records = []
    for policy in POLICIES:
        pros, cons = simple_rag_retrieve(policy, corpus, CHOSEN_K)
        records.append({"policy": policy, "retrieved_pros": pros,
                        "retrieved_cons": cons, "system": "SimpleRAG",
                        "method": "dense_cosine_topk_full_corpus_stance_aware"})
    save_json(records, F_SIMPLERAG)
    print(f"  Saved {len(records)} SimpleRAG retrievals -> {F_SIMPLERAG}")
    return {r["policy"]: r for r in records}


# ============================================================
# I/O
# ============================================================
def system_path(system):
    return os.path.join(CONFIG_DIR, system.lower().replace(" ", "_") + ".json")


def load_retrievals():
    """Load retrievals for all systems. Graph-based systems read from the
    UC1 sweep; SimpleRAG is generated in Phase 0."""
    out = {}
    for system in SYSTEMS:
        if system == "SimpleRAG":
            out[system] = ensure_simplerag_retrieval()
            print(f"  Loaded SimpleRAG: {len(out[system])} policies")
            continue
        path = system_path(system)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing retrieval file: {path}\n"
                f"Run sweep_uc1_k.py first (pool={POOL_SIZE}, k={CHOSEN_K})."
            )
        records = load_json(path)
        out[system] = {r["policy"]: r for r in records}
        print(f"  Loaded {system}: {len(out[system])} policies")
    return out


def load_ground_truth():
    if not os.path.exists(GT_FILE):
        raise FileNotFoundError(
            f"Ground truth not found: {GT_FILE}\n"
            "Run generate_ground_truth.py first."
        )
    gt = load_json(GT_FILE)
    print(f"  Loaded ground truth: {len(gt)} policies")
    return gt


# ============================================================
# DISTRACTOR MANIFEST + distractor-rate@k (provenance-based)
# ============================================================
def load_distractor_manifest():
    """Load the per-policy per-stance distractor-text manifest, or return
    None if it is absent. Absence is non-fatal: distractor-rate metrics are
    then reported as None (e.g. on an L0 control with no distractors, or if
    build_distractor_manifest.py has not been run)."""
    if not os.path.exists(DISTRACTOR_MANIFEST):
        print(f"  Distractor manifest not found ({DISTRACTOR_MANIFEST}); "
              f"distractor-rate@k will be None. Run "
              f"build_distractor_manifest.py to enable it.")
        return None
    man = load_json(DISTRACTOR_MANIFEST)
    n = sum(len(v.get("pros", [])) + len(v.get("cons", []))
            for v in man.values())
    print(f"  Loaded distractor manifest: {len(man)} policies, "
          f"{n} distractor texts.")
    # Pre-strip into sets per (policy, stance) for O(1) membership.
    prepared = {}
    for policy, d in man.items():
        prepared[policy] = {
            "pros": {t.strip() for t in d.get("pros", [])},
            "cons": {t.strip() for t in d.get("cons", [])},
        }
    return prepared


def distractor_rate(retrieved, distractor_set, k):
    """Fraction of the top-k retrieved items that are injected distractors.

    Provenance-based: an item counts iff its stripped text is in the
    policy/stance distractor_set (built from is_distractor-tagged nodes),
    NOT iff its grade is 0 — this separates injected distractors from
    native low-grade arguments, which no grade-keyed metric can do.

    Returns None when distractor_set is None (no manifest / no distractors
    for this stance), so the cell is excluded from aggregation rather than
    counted as 0.0.
    """
    if distractor_set is None:
        return None
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for t in top if t.strip() in distractor_set)
    return hits / len(top)


# ============================================================
# MATCHING — exact text first, semantic fallback (CHANGE 6)
# ============================================================
# Ground-truth embeddings are cached per (policy, stance) so they are
# computed at most once across all systems and metrics (CHANGE 11).
_gt_cache = {}


def _dedup_grades(graded_map):
    """{stripped text: grade}, MAX on duplicate stripped keys — the same
    tie-break generate_ground_truth.py uses. Shared by matching, Recall's
    denominator, and nDCG's ideal ranking so all three see the SAME
    ground-truth key space."""
    out = {}
    for t, g in graded_map.items():
        s = t.strip()
        g = float(g)
        if s not in out or g > out[s]:
            out[s] = g
    return out


def _gt_entry(policy, stance, graded_map):
    """Cached (texts, grades, lookup, embeddings-or-None) for a graded map.
    Embeddings are computed lazily on first semantic fallback."""
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
    """Resolve each retrieved item to a ground-truth grade.

    Returns a list of dicts, one per retrieved item:
        {"grade": float, "match": "exact"|"semantic"|"none",
         "gt_text": stripped ground-truth text or None}

    Exact (stripped) text lookup is tried first — for retrievals drawn
    from the fully-judged pool this always succeeds and is noise-free.
    Items with no exact key (out-of-pool retrievals from B/C expansion)
    fall back to single-best-match semantic resolution; below threshold
    they are unmatched (grade 0, like the unjudged sentinel: no credit).
    """
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
# METRICS FROM MATCHES (nDCG / Precision / Recall / MeanRel)
# ============================================================
def _gain_exp(g):
    return 2.0 ** g - 1.0


def _gain_lin(g):
    return g


def ndcg_from_matches(matches, graded_map, k, gain):
    """Graded nDCG@k from resolved matches. Ideal ranking = best k
    consensus grades in the judged pool, descending (full-pool judging
    makes IDCG well-defined). Järvelin & Kekäläinen (2002)."""
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


def precision_from_matches(matches, k, relevant_grade=RELEVANT_GRADE):
    """Fraction of top-k whose resolved grade is >= relevant_grade.

    Single-best-match resolution (inherited from v1): a hit requires the
    item's OWN match to be graded relevant, closing the "near ANY
    relevant argument" saturation loophole on self-similar pools."""
    if not matches:
        return 0.0
    top = matches[:k]
    hits = sum(1 for m in top
               if m["match"] != "none" and m["grade"] >= relevant_grade)
    return hits / len(top)


def recall_from_matches(matches, graded_map, k,
                        relevant_grade=RELEVANT_GRADE):
    """Unique grade>=relevant_grade pool arguments surfaced in top-k,
    over all such arguments in the judged pool (CHANGE 7).

    Unique by ground-truth text: two retrieved paraphrases resolving to
    the same relevant argument count ONCE. Returns None when the stance
    has no relevant arguments (excluded from aggregation)."""
    dedup = _dedup_grades(graded_map)
    n_relevant = sum(1 for g in dedup.values() if g >= relevant_grade)
    if n_relevant == 0:
        return None
    found = {m["gt_text"] for m in matches[:k]
             if m["gt_text"] is not None and m["grade"] >= relevant_grade}
    return len(found) / n_relevant



# ------------------------------------------------------------
# S-RECALL@k — subtopic coverage (CHANGE 13)
# ------------------------------------------------------------
# Subtopic clusters are cached per (policy, stance) alongside the ground
# truth cache, since they are a function of the same graded_map and are
# reused across every system.
_subtopic_cache = {}


def _subtopic_clusters(policy, stance, graded_map,
                       relevant_grade=RELEVANT_GRADE,
                       distance_threshold=SUBTOPIC_DISTANCE_THRESHOLD):
    """Cluster the judged pool's relevant (grade >= relevant_grade)
    arguments into subtopics via agglomerative clustering (average
    linkage, cosine distance, no fixed k — CHANGE 13).

    Returns (cluster_of: {stripped text -> cluster_id}, n_clusters).
    n_clusters == 0 when there are fewer than 2 relevant arguments in the
    pool (S-Recall is then undefined for this stance, same convention as
    Recall@k's None).
    """
    key = (policy, stance, relevant_grade, distance_threshold)
    if key in _subtopic_cache:
        return _subtopic_cache[key]

    dedup = _dedup_grades(graded_map)
    rel_texts = [t for t, g in dedup.items() if g >= relevant_grade]

    if len(rel_texts) == 0:
        result = ({}, 0)
    elif len(rel_texts) == 1:
        result = ({rel_texts[0]: 0}, 1)
    else:
        embs = embed_texts(rel_texts)
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=distance_threshold,
        ).fit(embs)
        labels = clustering.labels_
        cluster_of = {t: int(lbl) for t, lbl in zip(rel_texts, labels)}
        result = (cluster_of, int(labels.max()) + 1)

    _subtopic_cache[key] = result
    return result


def srecall_from_matches(matches, policy, stance, graded_map, k,
                         relevant_grade=RELEVANT_GRADE):
    """S-Recall@k: fraction of relevant-argument subtopic clusters (see
    _subtopic_clusters) represented at least once among the top-k matched,
    relevant retrievals. Returns None when the stance has fewer than 2
    relevant arguments in the judged pool (no subtopic structure to
    measure), matching Recall@k's exclusion convention."""
    cluster_of, n_clusters = _subtopic_clusters(
        policy, stance, graded_map, relevant_grade)
    if n_clusters < 2:
        return None
    hit_clusters = {
        cluster_of[m["gt_text"]]
        for m in matches[:k]
        if m.get("gt_text") is not None
        and m["grade"] >= relevant_grade
        and m["gt_text"] in cluster_of
    }
    return len(hit_clusters) / n_clusters


def mean_relevance_from_matches(matches, k):
    """Mean matched grade of the top-k, on the 0-3 scale (CHANGE 8).
    Divides by k: returning fewer than k items is penalized."""
    if not matches or k <= 0:
        return 0.0
    return float(sum(m["grade"] for m in matches[:k]) / k)


def match_fractions(matches, k):
    """Diagnostic: fractions of top-k resolved exactly / semantically /
    not at all. Verifies the in-pool assumption per system (CHANGE 6)."""
    top = matches[:k]
    if not top:
        return 0.0, 0.0, 0.0
    n = len(top)
    e = sum(1 for m in top if m["match"] == "exact") / n
    s = sum(1 for m in top if m["match"] == "semantic") / n
    return e, s, 1.0 - e - s


# ============================================================
# ILD (reference-free diversity)
# ============================================================
def intra_list_distance(texts):
    """ILD = 1 - mean pairwise cosine similarity. 0.0 for size < 2.
    Carbonell & Goldstein (1998)."""
    if len(texts) < 2:
        return 0.0
    embs = embed_texts(texts)
    sim  = embs @ embs.T
    iu   = np.triu_indices(len(embs), k=1)
    return 1.0 - float(np.mean(sim[iu]))


def relevance_constrained_ild(texts, matches, relevant_grade=RELEVANT_GRADE):
    """EILD — relevance-constrained intra-list distance.

    Identical to intra_list_distance but computed ONLY over the retrieved
    items whose resolved ground-truth grade is >= relevant_grade. This
    removes the hardened-pool artifact whereby a retriever (notably the
    random baseline) inflates raw ILD simply by surfacing off-topic
    distractors that sit far apart in embedding space: a grade-0 distractor
    never enters the pairwise computation, so EILD measures diversity AMONG
    RELEVANT arguments — the construct of interest for the conflict map.

    `texts` and `matches` must be aligned (same order, same length): matches
    are the per-item resolution records from match_retrieved(), each carrying
    a "grade". Returns 0.0 when fewer than 2 relevant items are retrieved.
    """
    if not texts or not matches or len(texts) != len(matches):
        return 0.0
    rel_texts = [t for t, m in zip(texts, matches)
                 if m is not None and m.get("match") != "none"
                 and m.get("grade", 0.0) >= relevant_grade]
    return intra_list_distance(rel_texts)


# ============================================================
# Conflict density (reference-free, supplementary)
# ============================================================
def conflict_density(pros, cons):
    """Mean P(contradiction) over all pro x con pairs."""
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


# ============================================================
# DETERMINISTIC PASS
# ============================================================
def compute_deterministic(retrievals, ground_truth):
    print("\n=== DETERMINISTIC METRICS ===")
    distractor_manifest = load_distractor_manifest()
    results = load_checkpoint(F_DETERM)

    # CHANGE 12 — refuse to mix metric schemas. An old-schema checkpoint
    # is archived and everything is recomputed.
    if results and results.get("_schema") != DETERM_SCHEMA:
        bak = F_DETERM + ".v1.bak"
        print(f"  Checkpoint has old/missing schema -> archiving to {bak} "
              f"and recomputing.")
        os.replace(F_DETERM, bak)
        results = {}
    results["_schema"] = DETERM_SCHEMA

    for system in SYSTEMS:
        if system in results:
            print(f"  {system}: already done")
            continue
        print(f"  {system}...")
        per_policy = {}

        for policy in POLICIES:
            if policy not in retrievals[system]:
                continue
            rec  = retrievals[system][policy]
            pros = rec.get("retrieved_pros", [])
            cons = rec.get("retrieved_cons", [])
            gt   = ground_truth.get(policy, {})
            g_pros = gt.get("graded_pros", {})
            g_cons = gt.get("graded_cons", {})

            # Provenance-based distractor sets for this policy/stance (None
            # when no manifest or this policy has no injected distractors).
            if distractor_manifest is None:
                d_pros = d_cons = None
            else:
                dm = distractor_manifest.get(policy, {})
                d_pros = dm.get("pros")   # set or None
                d_cons = dm.get("cons")

            m_pros = match_retrieved(pros, g_pros, policy, "pros")
            m_cons = match_retrieved(cons, g_cons, policy, "cons")

            ep, sp, up = match_fractions(m_pros, CHOSEN_K)
            ec, sc, uc = match_fractions(m_cons, CHOSEN_K)

            per_policy[policy] = {
                # nDCG — exponential gain (primary), linear (robustness)
                "ndcg_pros": ndcg_from_matches(
                    m_pros, g_pros, CHOSEN_K, _gain_exp),
                "ndcg_cons": ndcg_from_matches(
                    m_cons, g_cons, CHOSEN_K, _gain_exp),
                "ndcg_lin_pros": ndcg_from_matches(
                    m_pros, g_pros, CHOSEN_K, _gain_lin),
                "ndcg_lin_cons": ndcg_from_matches(
                    m_cons, g_cons, CHOSEN_K, _gain_lin),
                # Precision / Recall / MeanRel
                "prec_pros": precision_from_matches(m_pros, CHOSEN_K),
                "prec_cons": precision_from_matches(m_cons, CHOSEN_K),
                "recall_pros": recall_from_matches(
                    m_pros, g_pros, CHOSEN_K),
                "recall_cons": recall_from_matches(
                    m_cons, g_cons, CHOSEN_K),
                "srecall_pros": srecall_from_matches(
                    m_pros, policy, "pros", g_pros, CHOSEN_K),
                "srecall_cons": srecall_from_matches(
                    m_cons, policy, "cons", g_cons, CHOSEN_K),
                "meanrel_pros": mean_relevance_from_matches(
                    m_pros, CHOSEN_K),
                "meanrel_cons": mean_relevance_from_matches(
                    m_cons, CHOSEN_K),
                # Diversity + conflict
                "ild_pros":  intra_list_distance(pros),
                "ild_cons":  intra_list_distance(cons),
                "ild_all":   intra_list_distance(pros + cons),
                # EILD — relevance-constrained ILD (diversity among grade>=2
                # items only). Immune to the hardened-pool distractor
                # inflation that lets the random baseline farm raw ILD.
                "eild_pros": relevance_constrained_ild(pros, m_pros),
                "eild_cons": relevance_constrained_ild(cons, m_cons),
                "eild_all":  relevance_constrained_ild(
                    pros + cons, m_pros + m_cons),
                # Distractor-rate@k — DIRECT, provenance-based fraction of
                # top-k that are injected distractors (grade-independent).
                # None when no manifest / no distractors for the stance.
                "distractor_rate_pros": distractor_rate(pros, d_pros, CHOSEN_K),
                "distractor_rate_cons": distractor_rate(cons, d_cons, CHOSEN_K),
                "conflict_density": conflict_density(pros, cons),
                # Match diagnostics (mean over pros+cons fractions)
                "match_exact_frac":    float(np.mean([ep, ec])),
                "match_semantic_frac": float(np.mean([sp, sc])),
                "match_none_frac":     float(np.mean([up, uc])),
            }

        results[system] = {"per_policy": per_policy}
        save_json(results, F_DETERM)

    return results


# ============================================================
# JUDGE A — binary double-sided pairwise (3 axes) — DISABLED
# ============================================================
JUDGE_PROMPTS = {
    "conflict_engagement": """You are an expert debate analyst.
You will see two sets of arguments for a policy debate.

Decide which set contains arguments that more directly ENGAGE with
each other — where the PRO and CON arguments attack the same premises,
address the same dimensions, and respond to each other's claims.
Penalize sets where pros and cons talk past each other.

Reply with EXACTLY ONE LETTER: "A" or "B" on the first line.
Then one sentence justification. No other text.""",

    "redundancy_avoidance": """You are an expert debate analyst.
You will see two sets of arguments for a policy debate.

Decide which set has LESS internal redundancy. Penalize sets where
multiple arguments make essentially the same point. Reward sets where
each argument adds a distinct dimension, mechanism, or value.

Reply with EXACTLY ONE LETTER: "A" or "B" on the first line.
Then one sentence justification. No other text.""",

    "balanced_writing_aid": """You are an expert debate analyst.
A citizen needs to write a balanced position paper on this policy
using ONLY one of the argument sets below.

Decide which set would BETTER equip them to write a balanced,
well-informed contribution — covering the strongest pros, the
strongest cons, and the underlying value tensions.

Reply with EXACTLY ONE LETTER: "A" or "B" on the first line.
Then one sentence justification. No other text.""",
}


def build_pair_prompt(policy, set_a, set_b):
    def fmt_set(s):
        p = "\n".join(f"  {i+1}. {a}"
                      for i, a in enumerate(s.get("retrieved_pros", [])))
        c = "\n".join(f"  {i+1}. {a}"
                      for i, a in enumerate(s.get("retrieved_cons", [])))
        return f"PRO arguments:\n{p}\nCON arguments:\n{c}"
    return (f'Policy: "{policy}"\n\n'
            f"SET A:\n{fmt_set(set_a)}\n\n"
            f"SET B:\n{fmt_set(set_b)}\n\n"
            'Reply with ONLY "A" or "B" on the first line, then one sentence.')


def judge_once(system_prompt, user_prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            r = ollama.chat(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.0, "num_predict": 100,
                         "num_ctx": 8192},
            )
            ans = parse_ab(r["message"]["content"])
            if ans is not None:
                return ans
            time.sleep(1)
        except Exception as e:
            print(f"      judge error (try {attempt+1}): {e}")
            time.sleep(2)
    return None


def double_sided(axis_prompt, policy, set_b, set_ch):
    r1 = judge_once(axis_prompt, build_pair_prompt(policy, set_b, set_ch))
    time.sleep(SLEEP_SEC)
    r2 = judge_once(axis_prompt, build_pair_prompt(policy, set_ch, set_b))
    time.sleep(SLEEP_SEC)
    bw = cw = 0
    if r1 == "A": bw += 1
    elif r1 == "B": cw += 1
    if r2 == "B": bw += 1
    elif r2 == "A": cw += 1
    winner = "system_b" if bw > cw else ("challenger" if cw > bw else "tie")
    return {"winner": winner, "b_wins": bw, "ch_wins": cw,
            "run1": r1, "run2": r2}


def run_pairwise(retrievals):
    if not RUN_PAIRWISE:
        print("\n=== JUDGE A: binary double-sided pairwise === SKIPPED "
              "(RUN_PAIRWISE=False)")
        return {a: {} for a in JUDGE_PROMPTS}
    print("\n=== JUDGE A: binary double-sided pairwise ===")
    results = load_checkpoint(F_PAIRWISE) or {a: {} for a in JUDGE_PROMPTS}
    b_ret = retrievals["System B"]

    for axis, axis_prompt in JUDGE_PROMPTS.items():
        if axis not in results:
            results[axis] = {}
        for ch in CHALLENGERS:
            if ch not in results[axis]:
                results[axis][ch] = {"per_policy": {}}
            done = set(results[axis][ch]["per_policy"].keys())
            print(f"  {axis} — System B vs {ch}: {len(done)}/{len(POLICIES)}")

            for policy in POLICIES:
                if policy in done:
                    continue
                if policy not in b_ret or policy not in retrievals[ch]:
                    continue
                res = double_sided(axis_prompt, policy,
                                   b_ret[policy], retrievals[ch][policy])
                results[axis][ch]["per_policy"][policy] = res
                save_json(results, F_PAIRWISE)

            decisions = list(results[axis][ch]["per_policy"].values())
            if decisions:
                b = sum(1 for d in decisions if d["winner"] == "system_b")
                c = sum(1 for d in decisions if d["winner"] == "challenger")
                t = sum(1 for d in decisions if d["winner"] == "tie")
                n = len(decisions)
                results[axis][ch].update({
                    "b_wins": b, "ch_wins": c, "ties": t, "total": n,
                    "b_winrate": round(b / n, 4) if n else 0.0,
                })
            save_json(results, F_PAIRWISE)
    return results


# ============================================================
# JUDGE B — rubric scoring (1-5 per set on 3 axes)
# Edge et al. GraphRAG; Liu G-Eval.
# ============================================================
RUBRIC_SYSTEM = """You are an expert debate analyst scoring ONE set of
arguments for a policy debate.

Score the set on each of the following criteria, from 1 (poor) to 5
(excellent). Score each criterion independently and use the full range.

  conflict_engagement : do the PRO and CON arguments directly engage
      each other — attacking the same premises and dimensions — rather
      than talking past each other?
  redundancy_avoidance : does each argument add a distinct point, with
      little repetition within the set?
  balanced_writing_aid : would this set equip a citizen to write a
      balanced, well-informed position paper covering the strongest
      pros, strongest cons, and the underlying value tensions?

Return ONLY this JSON, no markdown:
{
  "conflict_engagement": 1-5,
  "redundancy_avoidance": 1-5,
  "balanced_writing_aid": 1-5
}"""

RUBRIC_KEYS = ["conflict_engagement", "redundancy_avoidance",
               "balanced_writing_aid"]


def build_rubric_prompt(policy, retrieval_set):
    p = "\n".join(f"  {i+1}. {a}"
                  for i, a in enumerate(retrieval_set.get("retrieved_pros", [])))
    c = "\n".join(f"  {i+1}. {a}"
                  for i, a in enumerate(retrieval_set.get("retrieved_cons", [])))
    return (f'Policy: "{policy}"\n\n'
            f"PRO arguments:\n{p}\n\nCON arguments:\n{c}\n\n"
            "Score this set on the three criteria. Return only the JSON.")


def judge_rubric(policy, retrieval_set, retries=2):
    prompt = build_rubric_prompt(policy, retrieval_set)
    for attempt in range(retries + 1):
        try:
            r = ollama.chat(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": RUBRIC_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.0, "num_predict": 200,
                         "num_ctx": 8192},
            )
            obj = parse_json_object(r["message"]["content"],
                                    required_keys=RUBRIC_KEYS)
            if obj is not None:
                clean = {}
                for kk in RUBRIC_KEYS:
                    try:
                        v = float(obj[kk])
                    except (ValueError, TypeError):
                        v = None
                    clean[kk] = v if (v is not None and 1 <= v <= 5) else None
                if all(clean[kk] is not None for kk in RUBRIC_KEYS):
                    return clean
            time.sleep(1)
        except Exception as e:
            print(f"      rubric error (try {attempt+1}): {e}")
            time.sleep(2)
    return None


def run_rubric(retrievals):
    print("\n=== JUDGE B: rubric scoring (1-5 per set) ===")
    results = load_checkpoint(F_RUBRIC)

    for system in SYSTEMS:
        if system not in results:
            results[system] = {"per_policy": {}}
        done = set(results[system]["per_policy"].keys())
        print(f"  {system}: {len(done)}/{len(POLICIES)}")

        for policy in POLICIES:
            if policy in done or policy not in retrievals[system]:
                continue
            scores = judge_rubric(policy, retrievals[system][policy])
            if scores is not None:
                results[system]["per_policy"][policy] = scores
            save_json(results, F_RUBRIC)
            time.sleep(SLEEP_SEC)

        # Aggregate
        per_policy = results[system]["per_policy"]
        for kk in RUBRIC_KEYS:
            vals = [v[kk] for v in per_policy.values() if v.get(kk) is not None]
            m, s = mean_std(vals)
            results[system][f"mean_{kk}"] = m
            results[system][f"std_{kk}"]  = s
        save_json(results, F_RUBRIC)

    return results


# ============================================================
# AGGREGATE + REPORT
# ============================================================
DETERM_KEYS = [
    "ndcg_pros", "ndcg_cons", "ndcg_lin_pros", "ndcg_lin_cons",
    "prec_pros", "prec_cons", "recall_pros", "recall_cons",
    "srecall_pros", "srecall_cons",
    "meanrel_pros", "meanrel_cons",
    "ild_pros", "ild_cons", "ild_all",
    "eild_pros", "eild_cons", "eild_all",
    "distractor_rate_pros", "distractor_rate_cons", "conflict_density",
    "match_exact_frac", "match_semantic_frac", "match_none_frac",
]


def aggregate_deterministic(determ):
    agg = {}
    for system in SYSTEMS:
        pp = determ.get(system, {}).get("per_policy", {})
        agg[system] = {}
        for kk in DETERM_KEYS:
            # Recall can be None for a policy/stance with no grade>=2
            # arguments — those policies are excluded from the mean.
            vals = [v[kk] for v in pp.values()
                    if kk in v and v[kk] is not None]
            agg[system][kk] = float(np.mean(vals)) if vals else None
        agg[system]["n_policies"] = len(pp)
    return agg


def _pair_mean(a, b):
    if a is None or b is None:
        return None
    return float(np.mean([a, b]))


def write_per_policy_csv(determ):
    """CHANGE 10 — one row per (system, policy), every deterministic
    metric. This is the input to the modularity / tercile analysis."""
    rows = []
    for system in SYSTEMS:
        pp = determ.get(system, {}).get("per_policy", {})
        for policy, m in pp.items():
            rows.append({"system": system, "policy": policy,
                         **{kk: m.get(kk) for kk in DETERM_KEYS}})
    if not rows:
        return
    keys = ["system", "policy"] + DETERM_KEYS
    with open(F_PERPOLICY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {F_PERPOLICY} ({len(rows)} rows)")


def build_summary(agg, pairwise, rubric):
    print("\n=== BUILDING SUMMARY ===")
    out = {
        "operating_point": f"pool={POOL_SIZE}, k={CHOSEN_K}",
        "judge_model":     JUDGE_MODEL,
        "relevance_matching": (
            "exact text first; semantic fallback "
            f"(cosine >= {SEMANTIC_MATCH_THRESHOLD}, single best match) "
            "for out-of-pool retrievals only"),
        "relevant_grade_threshold": RELEVANT_GRADE,
        "ndcg_gain": "exponential 2^g-1 (primary); linear g (robustness)",
        "method": ("graded nDCG@k (exp+lin) + Precision@k + Recall@k "
                   "+ S-Recall@k + MeanRel@k + ILD + conflict density "
                   "+ match diagnostics + rubric judge"
                   + (" + binary pairwise" if RUN_PAIRWISE else "")),
        "per_system":       {},
        "pairwise_by_axis": {},
    }

    for system in SYSTEMS:
        d = agg.get(system, {})
        r = rubric.get(system, {})
        out["per_system"][system] = {
            "ndcg_pros": d.get("ndcg_pros"), "ndcg_cons": d.get("ndcg_cons"),
            "ndcg_mean": _pair_mean(d.get("ndcg_pros"), d.get("ndcg_cons")),
            "ndcg_lin_pros": d.get("ndcg_lin_pros"),
            "ndcg_lin_cons": d.get("ndcg_lin_cons"),
            "ndcg_lin_mean": _pair_mean(d.get("ndcg_lin_pros"),
                                        d.get("ndcg_lin_cons")),
            "prec_pros": d.get("prec_pros"), "prec_cons": d.get("prec_cons"),
            "recall_pros": d.get("recall_pros"),
            "recall_cons": d.get("recall_cons"),
            "srecall_pros": d.get("srecall_pros"),
            "srecall_cons": d.get("srecall_cons"),
            "srecall_mean": _pair_mean(d.get("srecall_pros"),
                                       d.get("srecall_cons")),
            "meanrel_pros": d.get("meanrel_pros"),
            "meanrel_cons": d.get("meanrel_cons"),
            "meanrel_mean": _pair_mean(d.get("meanrel_pros"),
                                       d.get("meanrel_cons")),
            "ild_all":   d.get("ild_all"),
            "eild_pros": d.get("eild_pros"),
            "eild_cons": d.get("eild_cons"),
            "eild_all":  d.get("eild_all"),
            "distractor_rate_pros": d.get("distractor_rate_pros"),
            "distractor_rate_cons": d.get("distractor_rate_cons"),
            "distractor_rate_mean": _pair_mean(
                d.get("distractor_rate_pros"), d.get("distractor_rate_cons")),
            "conflict_density": d.get("conflict_density"),
            "match_exact_frac":    d.get("match_exact_frac"),
            "match_semantic_frac": d.get("match_semantic_frac"),
            "match_none_frac":     d.get("match_none_frac"),
            "rubric_conflict_engagement":  r.get("mean_conflict_engagement"),
            "rubric_redundancy_avoidance": r.get("mean_redundancy_avoidance"),
            "rubric_balanced_writing_aid": r.get("mean_balanced_writing_aid"),
            "n_policies": d.get("n_policies"),
        }

    if RUN_PAIRWISE:
        for axis in JUDGE_PROMPTS:
            out["pairwise_by_axis"][axis] = {}
            for ch in CHALLENGERS:
                pw = pairwise.get(axis, {}).get(ch, {})
                out["pairwise_by_axis"][axis][ch] = {
                    "b_wins": pw.get("b_wins"), "ch_wins": pw.get("ch_wins"),
                    "ties": pw.get("ties"), "total": pw.get("total"),
                    "b_winrate": pw.get("b_winrate"),
                }

    save_json(out, F_SUMMARY)
    return out


def write_csv(summary):
    rows = []
    for system in SYSTEMS:
        rows.append({"section": "deterministic", "system": system,
                     **summary["per_system"][system]})
    for axis, per_ch in summary["pairwise_by_axis"].items():
        for ch, m in per_ch.items():
            rows.append({"section": f"pairwise_{axis}", "system": ch, **m})
    # Union of all keys
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(F_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"  wrote {F_CSV}")


def print_report(summary):
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    def show(rows, headers):
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print(headers); [print(r) for r in rows]

    print("\n\n" + "=" * 78)
    print(f"UC1 EVALUATION — {summary['operating_point']}")
    print("=" * 78)
    print(f"Judge        : {summary['judge_model']}")
    print(f"Relevance    : {summary['relevance_matching']}")
    print(f"Relevant if  : grade >= {summary['relevant_grade_threshold']}")

    print("\n-- Relevance metrics (higher = better) --")
    headers = ["System", "nDCG", "nDCG(lin)", "MeanRel", "P@k", "R@k", "S-R@k"]
    rows = []
    for system in SYSTEMS:
        m = summary["per_system"][system]
        rows.append([
            system, fmt(m["ndcg_mean"]), fmt(m["ndcg_lin_mean"]),
            fmt(m["meanrel_mean"]),
            fmt(_pair_mean(m["prec_pros"], m["prec_cons"])),
            fmt(_pair_mean(m["recall_pros"], m["recall_cons"])),
            fmt(m.get("srecall_mean")),
        ])
    show(rows, headers)

    print("\n-- Diversity / conflict / matching diagnostics --")
    headers = ["System", "ILD all", "EILD all", "Distr.rate",
               "Conflict", "exact", "semantic", "none"]
    rows = []
    for system in SYSTEMS:
        m = summary["per_system"][system]
        rows.append([system, fmt(m["ild_all"]), fmt(m.get("eild_all")),
                     fmt(m.get("distractor_rate_mean")),
                     fmt(m["conflict_density"]),
                     fmt(m["match_exact_frac"]),
                     fmt(m["match_semantic_frac"]),
                     fmt(m["match_none_frac"])])
    show(rows, headers)

    print("\n-- Rubric judge (1-5 per set, higher = better) --")
    headers = ["System", "Conflict eng.", "Redundancy avoid.", "Writing aid"]
    rows = []
    for system in SYSTEMS:
        m = summary["per_system"][system]
        rows.append([system, fmt(m["rubric_conflict_engagement"]),
                     fmt(m["rubric_redundancy_avoidance"]),
                     fmt(m["rubric_balanced_writing_aid"])])
    show(rows, headers)

    if not RUN_PAIRWISE:
        print("\n-- Binary pairwise -- SKIPPED (RUN_PAIRWISE=False)")
    for axis, per_ch in summary["pairwise_by_axis"].items():
        print(f"\n-- Binary pairwise: {axis} (System B win rate) --")
        headers = ["Challenger", "B wins", "Ch wins", "Ties", "B winrate"]
        rows = []
        for ch in CHALLENGERS:
            m = per_ch[ch]
            rows.append([ch, m.get("b_wins") or 0, m.get("ch_wins") or 0,
                         m.get("ties") or 0,
                         (f"{m['b_winrate']*100:.1f}%"
                          if m.get("b_winrate") is not None else "-")])
        show(rows, headers)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print(f"UC1 EVALUATION v2 — pool={POOL_SIZE}, k={CHOSEN_K}")
    print("=" * 65)

    print("\nLoading retrievals...")
    retrievals = load_retrievals()
    print("\nLoading ground truth...")
    ground_truth = load_ground_truth()

    determ   = compute_deterministic(retrievals, ground_truth)
    pairwise = run_pairwise(retrievals)
    rubric   = run_rubric(retrievals)

    agg     = aggregate_deterministic(determ)
    write_per_policy_csv(determ)
    summary = build_summary(agg, pairwise, rubric)
    write_csv(summary)
    print_report(summary)

    print(f"\nResults saved to: {OUT_ROOT}/")


if __name__ == "__main__":
    main()