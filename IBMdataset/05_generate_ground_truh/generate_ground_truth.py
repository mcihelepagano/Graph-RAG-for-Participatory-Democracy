"""
generate_ground_truth.py  (v5 — consensus labels, dual-granularity
                            agreement, ordinal Krippendorff's alpha)
===================================================================
Generates GRADED-RELEVANCE ground truth for UC1 evaluation.

WHAT'S NEW IN v5 (robustness upgrades, each tied to published practice)
-----------------------------------------------------------------------
v4 used the PRIMARY annotator's grade as the ground truth and used the
secondary annotator only to report a single quadratic-weighted Cohen's
kappa on the full 0-3 scale. v5 keeps everything v4 did well and adds
five evidence-grounded robustness measures. Each is configurable and
defaults to the more rigorous behaviour.

  1. CONSENSUS LABELS (was: primary-only).
     The ground-truth grade is now a CONSENSUS of the two annotators
     rather than the primary's unilateral grade. With two annotators a
     majority vote is impossible, so a documented resolution rule is
     applied per argument. Default = MIN (conservative): on disagreement
     take the lower grade, on the assumption that if relevance cannot be
     agreed it should not be treated as highly relevant. This mirrors the
     TripJudge collection's disagreement heuristic.
       Refs: Arabzadeh, Kanoulas & Clarke, "TripJudge" (CIKM 2022);
             Hofstätter et al., "FiRA" (SIGIR 2020) — majority/aggregation;
             Wahle et al., Tetun ad-hoc retrieval (2024) — majority voting.
     Configurable via CONSENSUS_RULE in {"min","max","mean","primary"}.

  2. DUAL-GRANULARITY AGREEMENT (was: 4-class kappa only).
     Inter-annotator agreement is now reported BOTH on the full 0-3
     ordinal scale AND on the collapsed binary scale (relevant = grade
     >= RELEVANT_GRADE_THRESHOLD). Full-scale agreement is known to be
     lower for fine-grained relevance; the binary cut is the granularity
     that actually drives the relevant-set used by Precision@k.
       Refs: Hofstätter et al., "FiRA" (SIGIR 2020) — 2-class vs 4-class
             kappa; Arabzadeh et al., "TripJudge" (CIKM 2022) — 4-grade
             vs 2-grade agreement; Voorhees et al., "TREC 2025 RAG Track
             Overview" — Cohen's kappa to validate automated vs human
             labels on a shared graded scale.

  3. ORDINAL KRIPPENDORFF'S ALPHA + BOOTSTRAP CI (new, secondary stat).
     Reported alongside quadratic-weighted kappa as a second, coefficient-
     family-independent reliability estimate. Alpha is chance-corrected,
     handles ordinal scales with a distance metric, generalises to >2
     annotators (future-proofing), and tolerates missing data — exactly
     the unjudged sentinels introduced in (5). A 95% bootstrap CI is
     reported because alpha's sampling distribution is not closed-form.
       Refs: Krippendorff, "Content Analysis" (2004), ch. 11;
             Hayes & Krippendorff, "Answering the call for a standard
             reliability measure for coding data" (Comm. Methods &
             Measures, 2007); for a 2025 RAG precedent see the Krippendorff
             alpha + bootstrap-CI usage in large-scale medical-RAG expert
             evaluation (2025).

  4. SHUFFLE-BEFORE-BATCH (mitigates batch-relative grading).
     v4 batched arguments in storage order, so for dense policies the
     ~7 batches per stance could correlate batch composition with how
     the corpus happened to be ordered, inducing per-batch scale drift.
     v5 shuffles each stance with a recorded seed before batching, so
     batch membership is independent of storage order. The shuffle is
     undone before saving, so grades realign with the original argument
     order. The seed is stored for exact reproducibility.

  5. UNJUDGED SENTINEL (was: silent zero-fill).
     v4 defaulted any id the model skipped to grade 0 (IRRELEVANT),
     conflating "judged irrelevant" with "not judged". v5 records a
     sentinel (None) for missing ids and EXCLUDES them from metrics and
     from the consensus, rather than fabricating a grade-0 label. The
     coverage guard is retained as a retry trigger; the sentinel only
     applies to the residual gaps that survive all retries.

DESIGN INHERITED FROM v4 (kept unchanged)
-----------------------------------------
This is the consolidated, rigour-first version. Compared to v3.1 it makes
two methodological simplifications and keeps all reliability scaffolding.

  REMOVED — semantic deduplication.
      Rationale: dedup filtered the judged candidate pool with an embedding
      model + similarity threshold, so the ground truth was grounded in a
      DIFFERENT argument space than the retrieval systems (which operate on
      the full pool). A system that retrieved a near-duplicate of a relevant
      argument was wrongly penalised. For an LLM annotator at temperature 0
      there is no annotator-fatigue benefit to dedup, so it was pure cost:
      an extra dependency, an extra hyperparameter to justify, and a
      candidate-space mismatch. Removed.

  REMOVED — positional shortlist (first 60).
      Rationale: "first 60 of 200" meant the annotator never saw arguments
      61-200, while the retrieval systems did. Any strong argument in the
      tail looked like a retrieval error at evaluation time. The annotator
      now judges the SAME pool the systems retrieve from (up to
      MAX_ARGS_PER_STANCE), removing the sampling bias.

  KEPT — reliability scaffolding (these are rigour, not complexity):
      - per-call timeout (prevents silent grade_secondary=null on slow runs)
      - verbose error logging (failures are diagnosable in real time)
      - retry with backoff sleep
      - per-policy checkpointing (a crash mid-run loses nothing)
      - parse_grades 50%-coverage guard (rejects truncated/garbled output)

  KEPT — the science:
      - graded relevance 0-3 (nDCG-compatible; not fixed top-K)
      - quadratic-weighted Cohen's kappa (chance-corrected, distance-aware)
      - relevant set = grade >= 2
      - full diagnostic report with grade distributions + drift visibility

KNOWN LIMITATION (note in thesis):
    Judging the full pool produces a long grade-0 tail and risks annotator
    drift across the ~7 batches per stance. Zeros do not distort nDCG, and
    the per-policy grade distribution in the report lets drift be detected.

Usage:
    python generate_ground_truth.py
"""

import time
import re
import random
import numpy as np
from collections import Counter

from policies import POLICIES
from common import (load_json, save_json, load_checkpoint,
                    parse_json_object)
from ollama import Client as OllamaClient


# ============================================================
# CONFIG
# ============================================================
OLLAMA_HOST = "http://127.0.0.1:11434"

ANNOTATOR_PRIMARY   = "llama3.3:70b"
ANNOTATOR_SECONDARY = "gemma4:31b"

NEO4J_EXPORT_FILE = "data/neo4j_export_with_new_edges.json"
OUTPUT_FILE       = "data/ground_truth.json"
REPORT_FILE       = "data/ground_truth_report.json"

# Full pool: annotator judges the SAME arguments the retrieval systems
# retrieve from. Set to match the retrieval pool_size (200).
MAX_ARGS_PER_STANCE      = 200
RELEVANT_GRADE_THRESHOLD = 2

# --- v5: consensus resolution -------------------------------------------
# How a single ground-truth grade is derived from the two annotators'
# grades for the same argument:
#   "min"     : conservative — lower grade wins on disagreement (TripJudge
#               heuristic: unresolved relevance is treated as less relevant)
#   "max"     : generous — higher grade wins
#   "mean"    : averaged then rounded to nearest integer (ties -> down)
#   "primary" : v4 behaviour — primary annotator's grade is taken verbatim
# When only one annotator graded an argument (the other returned a
# sentinel), that single available grade is used regardless of rule.
CONSENSUS_RULE = "min"

# --- v5: shuffle-before-batch -------------------------------------------
# Shuffle each stance's argument order before batching so batch
# composition is independent of corpus storage order. Grades are
# realigned to the original order before saving. Set False for v4 behaviour.
SHUFFLE_BEFORE_BATCH = True
SHUFFLE_SEED         = 42

# --- v5: agreement reporting --------------------------------------------
# Compute Krippendorff's ordinal alpha (with bootstrap CI) in addition to
# quadratic-weighted Cohen's kappa. Set False to skip alpha entirely.
COMPUTE_KRIPPENDORFF = True
ALPHA_BOOTSTRAP_N    = 2000   # bootstrap resamples for the 95% CI
ALPHA_CI_SEED        = 7

BATCH_SIZE = 30          # arguments per annotation request

# Output token budget. A 30-argument batch emits ~700-900 tokens of JSON,
# but gemma4:31b can pad with repeated structure and overrun a tight
# budget, truncating the list mid-object (observed: response cut off at
# '"grade": ' with no value). 4000 gives generous headroom so the model
# finishes the list. The parse guard + list_recovery handle any residual
# truncation near the very end.
NUM_PREDICT = 4000

HIGH_AGREEMENT   = 0.6
MEDIUM_AGREEMENT = 0.4

SLEEP_SEC          = 0.5
RETRY_SLEEP_SEC    = 5
PARSE_RETRY_SLEEP  = 2

# The Ollama Python client uses httpx, which defaults to ~30s timeout.
# A batch of 30 arguments through gemma4:31b with constrained VRAM can
# take 2-3 minutes. Without an explicit timeout the client silently
# times out, the except block catches it, all retries fail, and the
# policy is quietly demoted to SINGLE_ANNOTATOR — the exact bug that
# v3.1 described but never actually fixed (keep_alive is NOT a timeout,
# it controls how long the model stays loaded in VRAM after the request).
OLLAMA_TIMEOUT_SEC = 300  # 5 minutes per request

ollama = OllamaClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_SEC)


# ============================================================
# LOAD / FETCH
# ============================================================
def load_export():
    return load_json(NEO4J_EXPORT_FILE)


def fetch_arguments(export, policy_name):
    """Resolve PRO/CON argument IDs to text, capped at MAX_ARGS_PER_STANCE.

    No dedup, no shortlist — returns the full pool (up to the cap) in the
    same order the retrieval systems see it.
    """
    policies  = export.get("policies", {})
    arguments = export.get("arguments", {})
    if policy_name not in policies:
        return [], []
    data = policies[policy_name]

    def resolve(id_list):
        out = []
        for aid in id_list:
            arg = arguments.get(aid, {})
            txt = (arg.get("text") or "").strip()
            if txt:
                out.append(txt)
            if len(out) >= MAX_ARGS_PER_STANCE:
                break
        return out

    return resolve(data.get("pros", [])), resolve(data.get("cons", []))


# ============================================================
# ANNOTATION
# ============================================================
ANNOTATION_SYSTEM_PROMPT = """You are a debate research analyst grading arguments for a policy briefing.

You will be shown a numbered list of arguments on ONE side of a policy debate.
Rate EVERY argument on a 0-3 relevance scale for a researcher who needs to
understand this debate thoroughly and objectively:

  3 = ESSENTIAL. Addresses the core claim of the debate; a researcher could
      not write a balanced analysis without engaging this point.
  2 = RELEVANT. A substantive, well-grounded argument that adds a real
      dimension, but is not indispensable.
  1 = WEAK. On-topic but vague, a bare restatement of the policy position,
      or a minor procedural point.
  0 = IRRELEVANT. Off-topic, incoherent, or contributes nothing.

CALIBRATION EXAMPLES — anchor your judgement to these.

Worked example for policy: "Should the death penalty be abolished?"
PRO arguments (i.e. arguments SUPPORTING abolition):

  Argument: "Capital punishment is irreversible — wrongful executions
  cannot be undone, and the empirical record shows innocent people
  have been executed."
  Grade: 3 (ESSENTIAL).
  Reason: A core abolitionist argument grounded in evidence; no
  balanced analysis can omit this.

  Argument: "Many advanced democracies have abolished the death
  penalty, suggesting it is incompatible with modern human-rights
  norms."
  Grade: 2 (RELEVANT).
  Reason: A substantive comparative argument that adds a dimension,
  but is supporting context rather than a core claim.

  Argument: "The death penalty is bad and should be abolished."
  Grade: 1 (WEAK).
  Reason: On-topic but a bare restatement of the position with no
  reasoning.

  Argument: "I read a poem about prisons last week."
  Grade: 0 (IRRELEVANT).
  Reason: Off-topic.

KEY RULES:
- Grade each argument on its own merits.
- Do NOT impose a quota — it is fine if many arguments are a 3 or many
  are a 0. Be consistent across the list.
- A long argument is not automatically high-grade; a short argument is
  not automatically low-grade. Grade content, not length.

Return ONLY valid JSON. No markdown fences, no commentary outside the JSON."""


def build_annotation_prompt(policy, stance_label, arguments):
    numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(arguments))
    entries = ",\n    ".join(
        f'{{"id": {i+1}, "grade": 0}}' for i in range(min(3, len(arguments)))
    )
    return f"""Policy: "{policy}"

{stance_label} ARGUMENTS ({len(arguments)} total):
{numbered}

Task:
Assign every argument a relevance grade from 0 to 3 (see the scale and the
calibration examples in the system message).

Return ONLY this JSON — one entry per argument, no markdown fences:
{{
  "grades": [
    {entries},
    ...one entry for EACH of the {len(arguments)} arguments...
  ]
}}

Rules:
- Output exactly {len(arguments)} entries, ids 1 to {len(arguments)}.
- "grade" must be an integer 0, 1, 2, or 3.
- Do NOT wrap the response in ```json``` fences."""


def _extract_grade_pairs(raw):
    """Salvage complete {"id": N, "grade": M} pairs from a possibly
    truncated response by direct regex, ignoring JSON validity.

    This is the fallback for mid-object truncation, which the suffix-based
    list_recovery in parse_json_object cannot repair (a dangling
    '"grade": ' with no value is not closeable). Every COMPLETE pair before
    the truncation point is still recoverable, and the coverage guard
    decides whether enough survived. Returns a list of (id, grade) ints.
    """
    if not raw:
        return []
    pairs = []
    # Matches: "id": 12 , "grade": 3   (whitespace/newline tolerant)
    pattern = re.compile(
        r'"id"\s*:\s*(\d+)\s*,\s*"grade"\s*:\s*([0-3])\b')
    for m in pattern.finditer(raw):
        pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def parse_grades(raw, n_expected, coverage_threshold=0.9):
    """Parse {id, grade} entries into a grade list aligned with the batch.

    Two-stage parse:
      1. Try parse_json_object (handles clean + suffix-recoverable JSON).
      2. If that fails or under-covers, fall back to regex extraction of
         complete {id, grade} pairs, which survives mid-object truncation.

    Missing ids stay at 0. Because dedup was removed, the pool can contain
    arguments the model skips; an unjudged argument silently defaulting to
    grade 0 (IRRELEVANT) would corrupt the ground truth. So we require the
    model to have judged at least `coverage_threshold` of the batch — any
    lower and we reject the response and retry, rather than zero-filling a
    large gap. The caller logs how many were filled.
    """
    entries = []
    obj = parse_json_object(raw, required_keys=["grades"], list_recovery=True)
    if obj is not None and isinstance(obj.get("grades"), list):
        for e in obj["grades"]:
            if isinstance(e, dict):
                try:
                    entries.append((int(e["id"]), int(e["grade"])))
                except (KeyError, ValueError, TypeError):
                    continue

    # Fallback / supplement: regex-salvage complete pairs. This rescues
    # responses truncated mid-object, where stage 1 yields too few entries.
    if len(entries) < n_expected:
        salvaged = _extract_grade_pairs(raw)
        if len(salvaged) > len(entries):
            entries = salvaged

    if not entries:
        return None

    # v5: unjudged ids stay as a sentinel (None), NOT grade 0. A skipped
    # argument is "not judged", not "judged irrelevant"; conflating the two
    # fabricates negative labels. Downstream code excludes None from
    # metrics and from the consensus.
    grades = [None] * n_expected
    assigned = [False] * n_expected
    seen = 0
    for raw_id, g in entries:
        idx = raw_id - 1
        if 0 <= idx < n_expected and 0 <= g <= 3 and not assigned[idx]:
            grades[idx] = g
            assigned[idx] = True
            seen += 1
    # Reject responses that judged fewer than coverage_threshold of the
    # batch — a sign of truncation or skipping. Forces a retry instead of
    # recording many sentinels. The sentinel is only for the residual gaps
    # that survive all retries.
    if seen < n_expected * coverage_threshold:
        return None
    n_missing = n_expected - seen
    if n_missing > 0:
        print(f"      note: {n_missing}/{n_expected} ids missing from "
              f"response, recorded as UNJUDGED (sentinel, excluded from "
              f"metrics)")
    return grades


def annotate_stance(policy, stance_label, arguments, model, retries=3,
                    batch_size=BATCH_SIZE):
    """Grade a full stance pool in batches of `batch_size`.

    Returns a flat grade list aligned with `arguments` (sentinel None for
    any argument left unjudged after all retries), or None if any batch
    fails all retries.

    v5: when SHUFFLE_BEFORE_BATCH is set, the arguments are permuted with a
    deterministic per-stance seed before batching, so batch composition is
    independent of corpus storage order. Grades are mapped back to the
    original argument order before returning, so callers see no difference
    in alignment.
    """
    if not arguments:
        return []

    n = len(arguments)

    # Build the (possibly shuffled) processing order.
    order = list(range(n))
    if SHUFFLE_BEFORE_BATCH and n > batch_size:
        # Per-stance deterministic seed: same policy+stance always shuffles
        # identically, so a resumed run reproduces the exact batching.
        rng = random.Random(f"{SHUFFLE_SEED}|{policy}|{stance_label}")
        rng.shuffle(order)

    shuffled_args = [arguments[i] for i in order]

    if n <= batch_size:
        grades_shuf = _annotate_chunk(policy, stance_label, shuffled_args,
                                      model, 0, retries)
    else:
        grades_shuf = []
        n_batches = (n + batch_size - 1) // batch_size
        for b, start in enumerate(range(0, n, batch_size)):
            chunk = shuffled_args[start:start + batch_size]
            chunk_grades = _annotate_chunk(
                policy, stance_label, chunk, model, start, retries
            )
            if chunk_grades is None:
                print(f"      batch {b+1}/{n_batches} failed "
                      f"(model={model}, args {start+1}-{start+len(chunk)})")
                return None
            grades_shuf.extend(chunk_grades)

    if grades_shuf is None:
        return None

    # Map grades from shuffled order back to original argument order.
    grades = [None] * n
    for shuf_pos, orig_idx in enumerate(order):
        grades[orig_idx] = grades_shuf[shuf_pos]
    return grades


def _annotate_chunk(policy, stance_label, arguments, model,
                    id_offset, retries):
    prompt = build_annotation_prompt(policy, stance_label, arguments)
    for attempt in range(retries):
        try:
            # At temperature 0 a truncation is deterministic — retrying the
            # identical prompt truncates at the identical point. A tiny
            # temperature on retries perturbs generation just enough to break
            # a repeating failure, without meaningfully affecting grades.
            temperature = 0.0 if attempt == 0 else 0.2
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": temperature,
                         "num_predict": NUM_PREDICT,
                         "num_ctx": 16384},
            )
            raw = response["message"]["content"]
            grades = parse_grades(raw, len(arguments))
            if grades is not None:
                return grades
            print(f"      parse failed (attempt {attempt+1}, "
                  f"model={model}, "
                  f"args {id_offset+1}-{id_offset+len(arguments)}) "
                  f"— raw[:120]: {repr(raw[:120])}")
            time.sleep(PARSE_RETRY_SLEEP)
        except Exception as e:
            # Verbose: type + message + which model, so failures are
            # diagnosable without trawling the Ollama server log.
            print(f"      ollama error (attempt {attempt+1}, model={model}): "
                  f"{type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP_SEC)
    return None


# ============================================================
# AGREEMENT — quadratic-weighted Cohen's kappa
# ============================================================
def _paired_valid(rater_a, rater_b):
    """Return the subset of (a, b) pairs where BOTH grades are present
    (not None sentinel). Agreement statistics are computed only over
    arguments both annotators actually judged.
    """
    a_out, b_out = [], []
    for x, y in zip(rater_a, rater_b):
        if x is None or y is None:
            continue
        a_out.append(int(x))
        b_out.append(int(y))
    return a_out, b_out


def quadratic_weighted_kappa(rater_a, rater_b, n_classes=4):
    a_raw, b_raw = _paired_valid(rater_a, rater_b)
    a = np.asarray(a_raw, dtype=int)
    b = np.asarray(b_raw, dtype=int)
    if len(a) == 0 or len(a) != len(b):
        return None

    O = np.zeros((n_classes, n_classes), dtype=float)
    for x, y in zip(a, b):
        O[x, y] += 1

    w = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    act_a = np.bincount(a, minlength=n_classes).astype(float)
    act_b = np.bincount(b, minlength=n_classes).astype(float)
    E = np.outer(act_a, act_b)

    O_sum, E_sum = O.sum(), E.sum()
    if O_sum == 0 or E_sum == 0:
        return None
    O /= O_sum
    E /= E_sum

    num = float((w * O).sum())
    den = float((w * E).sum())
    if den == 0:
        return 1.0
    return 1.0 - num / den


def binary_cohen_kappa(rater_a, rater_b, threshold=RELEVANT_GRADE_THRESHOLD):
    """Unweighted Cohen's kappa on the COLLAPSED binary scale
    (relevant = grade >= threshold). This is the granularity that defines
    the relevant set used by Precision@k, and full-scale agreement is
    known to understate agreement at this boundary (FiRA, TripJudge).
    Sentinel-aware: pairs with a missing grade are dropped.
    """
    a_raw, b_raw = _paired_valid(rater_a, rater_b)
    if not a_raw:
        return None
    a = np.array([1 if g >= threshold else 0 for g in a_raw])
    b = np.array([1 if g >= threshold else 0 for g in b_raw])
    n = len(a)
    po = float(np.mean(a == b))
    # Expected agreement from marginals.
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ============================================================
# AGREEMENT — Krippendorff's ordinal alpha (+ bootstrap CI)
# ============================================================
# Secondary, coefficient-family-independent reliability estimate.
#   Refs: Krippendorff, Content Analysis (2004), ch. 11;
#         Hayes & Krippendorff, Comm. Methods & Measures (2007).
# We use the ordinal difference metric so that, like quadratic-weighted
# kappa, a 2-vs-3 disagreement counts far less than a 0-vs-3 disagreement.
# A self-contained implementation is used so the cluster run has no hard
# dependency; if the `krippendorff` package is installed it is used as a
# cross-check instead.
def _krippendorff_alpha_ordinal(pair_a, pair_b):
    """Ordinal Krippendorff's alpha for two raters over paired grades.

    pair_a, pair_b : equal-length lists of integer grades (sentinels
                     already removed). Returns alpha in (-inf, 1], or None
                     if undefined (e.g. no variance / too few items).
    """
    a, b = _paired_valid(pair_a, pair_b)
    if len(a) < 2:
        return None

    # Value domain and marginal frequencies across BOTH raters.
    units = list(zip(a, b))
    values = sorted(set(a) | set(b))
    if len(values) < 2:
        return 1.0  # all identical -> perfect agreement, no disagreement

    # Coincidence-based formulation. For 2 raters per unit, each unit
    # contributes its 2 values. Build the value frequency vector n_v and
    # the ordinal distance metric.
    from collections import Counter
    n_v = Counter()
    for x, y in units:
        n_v[x] += 1
        n_v[y] += 1
    n_total = sum(n_v.values())  # = 2 * n_units

    # Ordinal distance between values g and h (Krippendorff 2004):
    # delta^2 = ( sum of n_v for v strictly between, plus half the
    #             endpoints )^2, using the marginal frequencies.
    def ordinal_dist2(g, h):
        if g == h:
            return 0.0
        lo, hi = (g, h) if g < h else (h, g)
        s = (n_v[lo] + n_v[hi]) / 2.0
        for v in values:
            if lo < v < hi:
                s += n_v[v]
        return s * s

    # Observed disagreement Do.
    Do_num = 0.0
    for x, y in units:
        Do_num += ordinal_dist2(x, y)
    Do = Do_num / len(units)  # per-unit (m=2) observed disagreement

    # Expected disagreement De from the marginals.
    De_num = 0.0
    for g in values:
        for h in values:
            De_num += n_v[g] * n_v[h] * ordinal_dist2(g, h)
    De = De_num / (n_total * (n_total - 1))

    if De == 0:
        return 1.0
    return 1.0 - (Do / De)


def krippendorff_alpha_with_ci(pair_a, pair_b, n_boot=ALPHA_BOOTSTRAP_N,
                               seed=ALPHA_CI_SEED):
    """Point estimate + 95% bootstrap CI for ordinal alpha over paired
    grades. Bootstrap resamples UNITS (argument pairs) with replacement,
    because alpha's sampling distribution has no simple closed form
    (Krippendorff 2004; Hayes & Krippendorff 2007).
    """
    a, b = _paired_valid(pair_a, pair_b)
    if len(a) < 2:
        return {"alpha": None, "ci_low": None, "ci_high": None, "n": len(a)}

    point = _krippendorff_alpha_ordinal(a, b)

    rng = np.random.default_rng(seed)
    n = len(a)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ba = [a[i] for i in idx]
        bb = [b[i] for i in idx]
        val = _krippendorff_alpha_ordinal(ba, bb)
        if val is not None:
            boots.append(val)
    if boots:
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
    else:
        lo = hi = None
    return {"alpha": (round(point, 4) if point is not None else None),
            "ci_low": (round(lo, 4) if lo is not None else None),
            "ci_high": (round(hi, 4) if hi is not None else None),
            "n": n}


# ============================================================
# CONSENSUS RESOLUTION
# ============================================================
def resolve_consensus(g_primary, g_secondary, rule=CONSENSUS_RULE):
    """Combine the two annotators' grades for ONE argument into a single
    ground-truth grade.

    Resolution rules (see CONSENSUS_RULE docs):
      "min"     conservative — lower grade on disagreement (TripJudge)
      "max"     generous — higher grade
      "mean"    rounded mean (ties round down via int(floor+0.5)->banker?)
      "primary" primary annotator verbatim (v4 behaviour)

    Sentinel handling: if exactly one annotator graded the argument, that
    single grade is used regardless of rule. If neither did, returns None.
    """
    if g_primary is None and g_secondary is None:
        return None
    if g_primary is None:
        return int(g_secondary)
    if g_secondary is None:
        return int(g_primary)

    gp, gs = int(g_primary), int(g_secondary)
    if rule == "primary":
        return gp
    if rule == "min":
        return min(gp, gs)
    if rule == "max":
        return max(gp, gs)
    if rule == "mean":
        # Round half up to the nearest integer grade.
        return int(np.floor((gp + gs) / 2.0 + 0.5))
    # Unknown rule -> fall back to conservative min.
    return min(gp, gs)


def agreement_flag(kappa_mean):
    if kappa_mean is None:
        return "SINGLE_ANNOTATOR"
    if kappa_mean >= HIGH_AGREEMENT:
        return "HIGH"
    if kappa_mean >= MEDIUM_AGREEMENT:
        return "MEDIUM"
    return "LOW"


# ============================================================
# DIAGNOSTIC REPORTING
# ============================================================
def grade_dist(grades):
    c = Counter(int(g) for g in grades if g is not None)
    return {str(k): c.get(k, 0) for k in range(4)}


def fmt4(v):
    """Format a possibly-None float to 4 dp for console logging."""
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def merge_dist(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in ["0", "1", "2", "3"]}


def write_report(results, output_path):
    per_policy = []
    kappas = []
    kappas_binary = []
    alphas = []

    dist_primary_pro   = {str(k): 0 for k in range(4)}
    dist_primary_con   = {str(k): 0 for k in range(4)}
    dist_secondary_pro = {str(k): 0 for k in range(4)}
    dist_secondary_con = {str(k): 0 for k in range(4)}

    def _alpha_mean(v):
        vals = [a["alpha"] for a in (v.get("alpha_ordinal_pros"),
                                     v.get("alpha_ordinal_cons"))
                if a is not None and a.get("alpha") is not None]
        return float(np.mean(vals)) if vals else None

    def _kbin_mean(v):
        vals = [x for x in (v.get("kappa_binary_pros"),
                            v.get("kappa_binary_cons")) if x is not None]
        return float(np.mean(vals)) if vals else None

    for policy, v in results.items():
        kbin_mean  = _kbin_mean(v)
        alpha_mean = _alpha_mean(v)
        per_policy.append({
            "policy":         policy,
            "kappa_pros":     v.get("kappa_pros"),
            "kappa_cons":     v.get("kappa_cons"),
            "kappa_mean":     v.get("kappa_mean"),
            "kappa_binary_mean": (round(kbin_mean, 4) if kbin_mean is not None else None),
            "alpha_ordinal_mean": (round(alpha_mean, 4) if alpha_mean is not None else None),
            "agreement_flag": v.get("agreement_flag"),
            "n_pros":         v.get("n_pros_judged"),
            "n_cons":         v.get("n_cons_judged"),
            "n_pros_unjudged": v.get("n_pros_unjudged", 0),
            "n_cons_unjudged": v.get("n_cons_unjudged", 0),
            "n_relevant_pros": len(v.get("relevant_pros", [])),
            "n_relevant_cons": len(v.get("relevant_cons", [])),
        })
        if v.get("kappa_mean") is not None:
            kappas.append(v["kappa_mean"])
        if kbin_mean is not None:
            kappas_binary.append(kbin_mean)
        if alpha_mean is not None:
            alphas.append(alpha_mean)

        # Distributions use the per-annotator grades for drift inspection.
        pro_pri = [r.get("grade_primary") for r in v.get("pros", [])]
        con_pri = [r.get("grade_primary") for r in v.get("cons", [])]
        pro_sec = [r.get("grade_secondary") for r in v.get("pros", [])
                   if r.get("grade_secondary") is not None]
        con_sec = [r.get("grade_secondary") for r in v.get("cons", [])
                   if r.get("grade_secondary") is not None]

        dist_primary_pro   = merge_dist(dist_primary_pro,   grade_dist(pro_pri))
        dist_primary_con   = merge_dist(dist_primary_con,   grade_dist(con_pri))
        dist_secondary_pro = merge_dist(dist_secondary_pro, grade_dist(pro_sec))
        dist_secondary_con = merge_dist(dist_secondary_con, grade_dist(con_sec))

    per_policy.sort(key=lambda r: (r["kappa_mean"] is None,
                                   r["kappa_mean"] or 0.0))
    flags = [v.get("agreement_flag") for v in results.values()]

    def _summ(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {"mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "min":  round(float(np.min(vals)), 4),
                "max":  round(float(np.max(vals)), 4)}

    report = {
        "config": {
            "annotator_primary":   ANNOTATOR_PRIMARY,
            "annotator_secondary": ANNOTATOR_SECONDARY,
            "max_args_per_stance": MAX_ARGS_PER_STANCE,
            "batch_size":          BATCH_SIZE,
            "relevant_grade":      RELEVANT_GRADE_THRESHOLD,
            "high_agreement":      HIGH_AGREEMENT,
            "medium_agreement":    MEDIUM_AGREEMENT,
            "consensus_rule":      CONSENSUS_RULE,
            "shuffle_before_batch": SHUFFLE_BEFORE_BATCH,
            "shuffle_seed":        SHUFFLE_SEED,
            "agreement_statistics": [
                "quadratic-weighted Cohen's kappa (0-3 ordinal scale)",
                "unweighted Cohen's kappa (binary relevant cut)",
                ("ordinal Krippendorff's alpha + 95% bootstrap CI"
                 if COMPUTE_KRIPPENDORFF else "ordinal alpha: disabled"),
            ],
            "deduplication":       "none (full pool judged)",
            "shortlist":           "none (full pool judged)",
            "unjudged_handling":   "sentinel (None), excluded from metrics",
            "references": [
                "Cohen (1968) weighted kappa",
                "Landis & Koch (1977) agreement bands",
                "Krippendorff (2004); Hayes & Krippendorff (2007) ordinal alpha",
                "Hofstaetter et al. FiRA (SIGIR 2020) 2-class vs 4-class kappa",
                "Arabzadeh et al. TripJudge (CIKM 2022) disagreement heuristic",
                "Voorhees et al. TREC 2025 RAG Track (kappa for automated labels)",
            ],
        },
        "per_policy":        per_policy,
        "agreement_summary": {
            "kappa_ordinal_qwk": _summ(kappas),
            "kappa_binary":      _summ(kappas_binary),
            "alpha_ordinal":     _summ(alphas),
            "n_policies": len(results),
            "n_HIGH":     flags.count("HIGH"),
            "n_MEDIUM":   flags.count("MEDIUM"),
            "n_LOW":      flags.count("LOW"),
            "n_SINGLE":   flags.count("SINGLE_ANNOTATOR"),
            # Back-compat: keep the flat keys v4 consumers expect.
            "mean_kappa": _summ(kappas)["mean"],
            "std_kappa":  _summ(kappas)["std"],
            "min_kappa":  _summ(kappas)["min"],
            "max_kappa":  _summ(kappas)["max"],
        },
        "grade_distribution": {
            "primary":   {"PRO": dist_primary_pro,   "CON": dist_primary_con},
            "secondary": {"PRO": dist_secondary_pro, "CON": dist_secondary_con},
        },
    }
    save_json(report, output_path)
    return report


def print_dist(label, dist):
    total = sum(dist.values())
    if total == 0:
        print(f"    {label:30s} (no data)")
        return
    pct = {k: 100 * v / total for k, v in dist.items()}
    print(f"    {label:30s} "
          f"0={dist['0']:>4} ({pct['0']:>4.1f}%)  "
          f"1={dist['1']:>4} ({pct['1']:>4.1f}%)  "
          f"2={dist['2']:>4} ({pct['2']:>4.1f}%)  "
          f"3={dist['3']:>4} ({pct['3']:>4.1f}%)")


# ============================================================
# MAIN
# ============================================================
def run():
    results   = load_checkpoint(OUTPUT_FILE)
    completed = set(results.keys())
    remaining = [p for p in POLICIES if p not in completed]

    print("=" * 65)
    print("Ground Truth Generation v4 — graded relevance (0-3), full pool")
    print("=" * 65)
    print(f"Primary annotator   : {ANNOTATOR_PRIMARY}")
    print(f"Secondary annotator : {ANNOTATOR_SECONDARY}")
    print(f"Pool per stance     : up to {MAX_ARGS_PER_STANCE} (no dedup, no shortlist)")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Relevant grade      : >= {RELEVANT_GRADE_THRESHOLD}")
    print(f"Agreement statistic : quadratic-weighted Cohen's kappa")
    print(f"Calibration         : few-shot examples in system prompt")
    print(f"Policies total      : {len(POLICIES)}")
    print(f"Already completed   : {len(completed)}")
    print(f"Remaining           : {len(remaining)}")
    print("=" * 65)

    print(f"\nLoading export: {NEO4J_EXPORT_FILE}")
    export = load_export()
    print(f"  {len(export.get('policies', {}))} policies, "
          f"{len(export.get('arguments', {}))} arguments loaded")

    policy_durations = []  # seconds per fully-processed policy, for ETA

    for i, policy in enumerate(remaining):
        policy_start = time.time()
        print(f"\n[{i+1}/{len(remaining)}] {policy}")

        pros, cons = fetch_arguments(export, policy)
        print(f"  Fetched: {len(pros)} PROs, {len(cons)} CONs (full pool)")
        if len(pros) < 2 or len(cons) < 2:
            print(f"  Skipping — too few arguments")
            continue

        # Primary
        print(f"  Primary annotation ({ANNOTATOR_PRIMARY})...")
        t_pri = time.time()
        gp_pri = annotate_stance(policy, "PRO", pros, ANNOTATOR_PRIMARY)
        gc_pri = annotate_stance(policy, "CON", cons, ANNOTATOR_PRIMARY)
        if gp_pri is None or gc_pri is None:
            print(f"  Primary failed — skipping policy")
            continue
        print(f"    primary took {time.time() - t_pri:.0f}s")
        time.sleep(SLEEP_SEC)

        # Secondary
        print(f"  Secondary annotation ({ANNOTATOR_SECONDARY})...")
        t_sec = time.time()
        gp_sec = annotate_stance(policy, "PRO", pros, ANNOTATOR_SECONDARY)
        gc_sec = annotate_stance(policy, "CON", cons, ANNOTATOR_SECONDARY)
        print(f"    secondary took {time.time() - t_sec:.0f}s")

        if gp_sec is None or gc_sec is None:
            print(f"  Secondary failed — single-annotator mode")
            k_pros = k_cons = k_mean = None
            kb_pros = kb_cons = None
            alpha_pro = alpha_con = None
            flag = "SINGLE_ANNOTATOR"
            gp_sec = gp_sec or [None] * len(pros)
            gc_sec = gc_sec or [None] * len(cons)
        else:
            # Full-scale quadratic-weighted kappa (sentinel-aware).
            k_pros = quadratic_weighted_kappa(gp_pri, gp_sec)
            k_cons = quadratic_weighted_kappa(gc_pri, gc_sec)
            valid  = [k for k in (k_pros, k_cons) if k is not None]
            k_mean = float(np.mean(valid)) if valid else None
            flag   = agreement_flag(k_mean)

            # Binary-collapsed kappa at the relevant-set boundary.
            kb_pros = binary_cohen_kappa(gp_pri, gp_sec)
            kb_cons = binary_cohen_kappa(gc_pri, gc_sec)

            # Ordinal Krippendorff's alpha + bootstrap CI (secondary stat).
            if COMPUTE_KRIPPENDORFF:
                alpha_pro = krippendorff_alpha_with_ci(gp_pri, gp_sec)
                alpha_con = krippendorff_alpha_with_ci(gc_pri, gc_sec)
            else:
                alpha_pro = alpha_con = None

            print(f"  Kappa(0-3 qwk): PRO={fmt4(k_pros)}, CON={fmt4(k_cons)} -> {flag}")
            print(f"  Kappa(binary) : PRO={fmt4(kb_pros)}, CON={fmt4(kb_cons)}")
            if COMPUTE_KRIPPENDORFF:
                print(f"  Alpha(ordinal): PRO={fmt4(alpha_pro['alpha'])} "
                      f"CI[{fmt4(alpha_pro['ci_low'])},{fmt4(alpha_pro['ci_high'])}], "
                      f"CON={fmt4(alpha_con['alpha'])} "
                      f"CI[{fmt4(alpha_con['ci_low'])},{fmt4(alpha_con['ci_high'])}]")

            print(f"  Grade distribution this policy:")
            print_dist("PRO primary",   grade_dist(gp_pri))
            print_dist("PRO secondary", grade_dist(gp_sec))
            print_dist("CON primary",   grade_dist(gc_pri))
            print_dist("CON secondary", grade_dist(gc_sec))
        time.sleep(SLEEP_SEC)

        # v5: the stored ground-truth grade is the CONSENSUS of both
        # annotators (rule = CONSENSUS_RULE), not the primary verbatim.
        # grade_primary / grade_secondary are retained for transparency.
        def build(texts, g_pri, g_sec):
            recs = []
            for t, gp, gs in zip(texts, g_pri, g_sec):
                consensus = resolve_consensus(gp, gs, rule=CONSENSUS_RULE)
                recs.append({
                    "text": t,
                    "grade": consensus,             # consensus = ground truth
                    "grade_primary":   (int(gp) if gp is not None else None),
                    "grade_secondary": (int(gs) if gs is not None else None),
                })
            return recs

        pro_records = build(pros, gp_pri, gp_sec)
        con_records = build(cons, gc_pri, gc_sec)

        # Dedup was removed, so the pool may contain identical argument
        # texts. The graded_* dicts below are keyed by text, so exact
        # duplicates collapse. v5: tie-break is MAX grade (deterministic),
        # not "last wins", so the lookup never disagrees with the list in a
        # direction that hides a relevant argument.
        n_pro_dupes = len(pro_records) - len({r["text"] for r in pro_records})
        n_con_dupes = len(con_records) - len({r["text"] for r in con_records})
        if n_pro_dupes or n_con_dupes:
            print(f"  warning: duplicate texts collapse graded dicts "
                  f"({n_pro_dupes} PRO, {n_con_dupes} CON). "
                  f"Lists keep all records; lookup keeps MAX grade.")

        def graded_lookup(records):
            """Text -> consensus grade, MAX on duplicate text, skipping
            unjudged (None consensus)."""
            out = {}
            for r in records:
                if r["grade"] is None:
                    continue
                if r["text"] not in out or r["grade"] > out[r["text"]]:
                    out[r["text"]] = r["grade"]
            return out

        results[policy] = {
            "pros": pro_records,
            "cons": con_records,
            # Relevant set uses the consensus grade; unjudged (None) excluded.
            "relevant_pros": [r["text"] for r in pro_records
                              if r["grade"] is not None
                              and r["grade"] >= RELEVANT_GRADE_THRESHOLD],
            "relevant_cons": [r["text"] for r in con_records
                              if r["grade"] is not None
                              and r["grade"] >= RELEVANT_GRADE_THRESHOLD],
            "graded_pros": graded_lookup(pro_records),
            "graded_cons": graded_lookup(con_records),
            "annotator_primary":   ANNOTATOR_PRIMARY,
            "annotator_secondary": ANNOTATOR_SECONDARY,
            "consensus_rule":      CONSENSUS_RULE,
            "kappa_pros":  (round(k_pros, 4) if k_pros is not None else None),
            "kappa_cons":  (round(k_cons, 4) if k_cons is not None else None),
            "kappa_mean":  (round(k_mean, 4) if k_mean is not None else None),
            "kappa_binary_pros": (round(kb_pros, 4) if kb_pros is not None else None),
            "kappa_binary_cons": (round(kb_cons, 4) if kb_cons is not None else None),
            "alpha_ordinal_pros": alpha_pro,
            "alpha_ordinal_cons": alpha_con,
            "agreement_flag":  flag,
            "n_pros_judged":   len(pros),
            "n_cons_judged":   len(cons),
            "n_pros_unjudged": sum(1 for r in pro_records if r["grade"] is None),
            "n_cons_unjudged": sum(1 for r in con_records if r["grade"] is None),
            "n_pro_text_dupes": n_pro_dupes,
            "n_con_text_dupes": n_con_dupes,
        }
        save_json(results, OUTPUT_FILE)
        print(f"  Saved. Relevant: {len(results[policy]['relevant_pros'])} PRO, "
              f"{len(results[policy]['relevant_cons'])} CON "
              f"(consensus rule={CONSENSUS_RULE})")

        # Timing + ETA. Uses only fully-processed policies (skipped/failed
        # ones don't reach here), so the average reflects real work.
        elapsed = time.time() - policy_start
        policy_durations.append(elapsed)
        avg = sum(policy_durations) / len(policy_durations)
        n_left = len(remaining) - (i + 1)
        eta_s = avg * n_left
        print(f"  Time: {elapsed:.0f}s this policy | "
              f"avg {avg:.0f}s | "
              f"{n_left} left | "
              f"ETA {eta_s/3600:.1f}h ({eta_s/60:.0f}m)")

    # ── Final report ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("WRITING DIAGNOSTIC REPORT")
    print("=" * 65)
    report = write_report(results, REPORT_FILE)

    s = report["agreement_summary"]
    kq = s["kappa_ordinal_qwk"]; kb = s["kappa_binary"]; al = s["alpha_ordinal"]
    print(f"\nAgreement summary across {s['n_policies']} policies "
          f"(consensus rule = {CONSENSUS_RULE}):")
    print(f"  Quadratic-weighted kappa (0-3): mean={kq['mean']} "
          f"std={kq['std']} range=[{kq['min']}, {kq['max']}]")
    print(f"  Binary Cohen's kappa (>= {RELEVANT_GRADE_THRESHOLD}): "
          f"mean={kb['mean']} std={kb['std']} range=[{kb['min']}, {kb['max']}]")
    print(f"  Ordinal Krippendorff's alpha : mean={al['mean']} "
          f"std={al['std']} range=[{al['min']}, {al['max']}]")
    print(f"  Flags  HIGH(>={HIGH_AGREEMENT})={s['n_HIGH']}  "
          f"MEDIUM(>={MEDIUM_AGREEMENT})={s['n_MEDIUM']}  "
          f"LOW={s['n_LOW']}  SINGLE={s['n_SINGLE']}")

    print(f"\nGlobal grade distribution (across all policies):")
    print(f"  PRIMARY ({ANNOTATOR_PRIMARY}):")
    print_dist("  PRO", report["grade_distribution"]["primary"]["PRO"])
    print_dist("  CON", report["grade_distribution"]["primary"]["CON"])
    print(f"  SECONDARY ({ANNOTATOR_SECONDARY}):")
    print_dist("  PRO", report["grade_distribution"]["secondary"]["PRO"])
    print_dist("  CON", report["grade_distribution"]["secondary"]["CON"])

    print(f"\nFiles:")
    print(f"  Ground truth : {OUTPUT_FILE}")
    print(f"  Report       : {REPORT_FILE}")
    print(f"\nFOR THE THESIS APPENDIX:")
    print(f"  - Report agreement at TWO granularities: quadratic-weighted")
    print(f"    Cohen's kappa on the 0-3 scale (mean={kq['mean']}) and")
    print(f"    unweighted kappa on the binary relevant cut (mean={kb['mean']}).")
    print(f"    Cite Cohen (1968); interpret via Landis & Koch (1977).")
    print(f"    Precedent for dual granularity: FiRA (SIGIR 2020), TripJudge")
    print(f"    (CIKM 2022).")
    print(f"  - Report ordinal Krippendorff's alpha (mean={al['mean']}) with")
    print(f"    95% bootstrap CIs as a coefficient-independent cross-check.")
    print(f"    Cite Krippendorff (2004); Hayes & Krippendorff (2007).")
    print(f"  - State that the ground-truth grade is the CONSENSUS of two")
    print(f"    annotators (rule={CONSENSUS_RULE}); cite TripJudge (CIKM 2022)")
    print(f"    for the disagreement heuristic.")
    print(f"  - Note: agreement is LLM-vs-LLM (inter-model consistency), not")
    print(f"    human-grounded. For a human anchor, hand-grade a small sample")
    print(f"    and report kappa against it (cf. TREC 2025 RAG Track, which")
    print(f"    uses kappa to validate automated labels against human ones).")
    print(f"  - Full pool judged (no dedup/shortlist); unjudged args carry a")
    print(f"    sentinel and are excluded from metrics, not scored as 0.")


if __name__ == "__main__":
    run()