"""
generate_relations.py
=====================
Generates CONTRADICTS and EQUIVALENT argument relations from scratch,
within-policy, using a three-stage pipeline:

  STAGE 1 — BLOCKING (cheap, embedding cosine).
    Reduces the pair space to plausible candidates before any expensive
    classification. Asymmetric floors, justified by the pair-count probe:
      CONTRADICTS : cross-stance PRO x CON, cosine >= 0.50  (moderate band)
      EQUIVALENT  : within-stance PRO-PRO + CON-CON, cosine >= 0.75 (paraphrase)
    Embeddings are read from the export, so blocking is pure numpy.

  STAGE 2 — NLI CLASSIFICATION (cross-encoder/nli-deberta-v3-base, GPU).
    Runs on blocked candidates only. Label order (verified against the
    model's id2label config): 0=contradiction, 1=entailment, 2=neutral.
      CONTRADICTS : BIDIRECTIONAL contradiction — min(P(contra|A->B),
                    P(contra|B->A)) >= TAU_CONTRA. NLI is directional, so
                    each unordered pair is scored BOTH ways and the weaker
                    (min) direction must clear the threshold. This is the
                    fix for the stance-opposition saturation problem: a
                    genuine rebuttal contradicts in both readings, whereas a
                    mere opposing-stance pair often does not.
      EQUIVALENT  : BIDIRECTIONAL entailment — P(entail | A->B) >= TAU_ENTAIL
                    AND P(entail | B->A) >= TAU_ENTAIL. NLI is directional,
                    so each unordered pair is scored BOTH ways.

  STAGE 3 — OPTIONAL LLM ADJUDICATION (llama3.3:70b via Ollama).
    OFF by default (USE_LLM_ADJUDICATION = False). When on, only the
    UNCERTAIN band is escalated — pairs where NLI is ambiguous
    (score in [LOW, HIGH)) — to an argumentative judge. Confident NLI
    verdicts pass straight through. This is the adaptive-confidence-
    threshold idea: cheap model decides the easy cases, expensive model
    only adjudicates the hard middle.

  SYMMETRY + CONFLICT RESOLUTION.
    Both relations are symmetric -> store each unordered pair once.
    A pair cannot be both CONTRADICTS and EQUIVALENT -> if both fire, keep
    the higher-confidence verdict.

SCOPE.
    Within-policy only. Every edge carries scope="within_policy". The
    schema supports "within_metapolicy" for a future expansion, but that
    is NOT generated here (no consumer in the current systems).

OUTPUT.
    data/relations_generated.json
    {
      "config": {...},
      "contradicts": [ {source, target, policy, confidence, method, scope}, ...],
      "equivalent":  [ {source, target, policy, confidence, method, scope}, ...],
      "stats": {...}
    }
    Edges reference argument IDs (the keys of export["arguments"]).
    Run with --merge to also write them into export["edges"] of a COPY
    of the export (never mutates the original in place).

VALIDATION.
    Use --sample N to dump N random candidates per relation with their
    texts + scores to data/relations_sample_for_labeling.json, for hand
    labelling and precision/recall measurement (recommended for the thesis).

Usage:
    python generate_relations.py                 # NLI only, write edges
    python generate_relations.py --llm           # enable LLM adjudication
    python generate_relations.py --sample 150    # also dump a labeling set
    python generate_relations.py --merge out.json  # merge edges into export copy
"""

import os
import sys
import json
import time
import argparse
import numpy as np

from common import (load_json, save_json, load_checkpoint,
                    nli_label_batch, get_ollama, parse_ab)


# ============================================================
# CONFIG
# ============================================================
NEO4J_EXPORT_FILE = "data/neo4j_export_71full.json"
OUTPUT_FILE       = "data/relations_generated2.json"
CANDIDATES_CACHE  = "data/relations_candidates2.json"  # stage-1 checkpoint

# Stage-1 blocking floors (from the pair-count probe)
FLOOR_CONTRA = 0.70   # cross-stance cosine floor for CONTRADICTS candidates
                      # (calibrated 2026: 0.50 gave ~33% saturation
                      #  stance-opposition graph; 0.70 + bidirectional min
                      #  lands ~11% — defensible rebuttal range. Cosine floor
                      #  is the dominant lever; contra prob is near-binary.)
FLOOR_EQUIV  = 0.75   # within-stance cosine floor for EQUIVALENT candidates

# Stage-2 NLI thresholds
TAU_CONTRA = 0.90     # min(contra fwd, contra bwd) to accept CONTRADICTS
                      # (bidirectional; near-binary distribution means any
                      #  0.6-0.95 cuts similarly — 0.90 chosen for headroom)
TAU_ENTAIL = 0.60     # P(entailment) in EACH direction to accept EQUIVALENT

# Stage-3 uncertain band (only used if LLM adjudication is enabled):
# NLI scores in [LOW, HIGH) are escalated to the LLM.
UNCERTAIN_LOW  = 0.40
UNCERTAIN_HIGH = 0.70

USE_LLM_ADJUDICATION = False   # default off; --llm turns it on
LLM_MODEL = "llama3.3:70b"

NLI_BATCH = 64
SCOPE = "within_policy"


# ============================================================
# LOAD
# ============================================================
def load_export():
    return load_json(NEO4J_EXPORT_FILE)


def build_arg_index(export):
    """Return {arg_id: text} and {arg_id: normalised np.array} for all
    arguments that carry both text and embedding."""
    arguments = export.get("arguments", {})
    texts, embs = {}, {}
    for aid, a in arguments.items():
        emb = a.get("embedding")
        txt = (a.get("text") or "").strip()
        if emb and txt:
            v = np.asarray(emb, dtype=float)
            n = np.linalg.norm(v)
            if n > 0:
                texts[aid] = txt
                embs[aid] = v / n
    return texts, embs


# ============================================================
# STAGE 1 — BLOCKING
# ============================================================
def block_candidates(export, texts, embs):
    """Return two lists of unordered candidate (id_a, id_b) pairs:
      contra_pairs  : cross-stance, cosine >= FLOOR_CONTRA, with policy tag
      equiv_pairs   : within-stance, cosine >= FLOOR_EQUIV, with policy tag
    Each entry is (policy, id_a, id_b, cosine).
    """
    policies = export.get("policies", {})
    contra_pairs, equiv_pairs = [], []

    for policy, data in policies.items():
        pro_ids = [a for a in data.get("pros", []) if a in embs]
        con_ids = [a for a in data.get("cons", []) if a in embs]

        # --- CONTRADICTS candidates: cross-stance PRO x CON ---
        if pro_ids and con_ids:
            P = np.stack([embs[a] for a in pro_ids])
            C = np.stack([embs[a] for a in con_ids])
            sims = P @ C.T
            pi, ci = np.where(sims >= FLOOR_CONTRA)
            for p, c in zip(pi, ci):
                contra_pairs.append(
                    (policy, pro_ids[p], con_ids[c], float(sims[p, c])))

        # --- EQUIVALENT candidates: within-stance (upper triangle) ---
        for stance_ids in (pro_ids, con_ids):
            if len(stance_ids) < 2:
                continue
            M = np.stack([embs[a] for a in stance_ids])
            sims = M @ M.T
            iu = np.triu_indices(len(stance_ids), k=1)
            keep = np.where(sims[iu] >= FLOOR_EQUIV)[0]
            rows, cols = iu[0][keep], iu[1][keep]
            for r, c in zip(rows, cols):
                equiv_pairs.append(
                    (policy, stance_ids[r], stance_ids[c], float(sims[r, c])))

    return contra_pairs, equiv_pairs


def block_metapolicy_contradicts(export, texts, embs):
    """Cross-policy CONTRADICTS candidates for the Coherence Audit (UC4).

    Groups policies by their 'metapolicy' field and, for each pair of
    DIFFERENT policies under the SAME metapolicy, blocks argument pairs at
    FLOOR_CONTRA. Unlike the within-policy tier, ALL stance combinations are
    considered: a coherence conflict can arise from any pairing (e.g. a
    citizen's PRO on policy X contradicting their PRO on policy Y). This is
    the semantically correct space for 'inconsistent positions across
    related domains'.

    Each entry is (metapolicy, id_a, id_b, cosine). The metapolicy name
    occupies the tag slot; edges are marked scope='within_metapolicy' and
    carry both source policies downstream via the argument records.
    """
    policies = export.get("policies", {})

    # Group policy names by metapolicy
    by_meta = {}
    for pname, data in policies.items():
        meta = data.get("metapolicy")
        if meta is None:
            continue
        by_meta.setdefault(meta, []).append(pname)

    # Precompute each policy's full argument id list (PRO + CON, embedded)
    def all_args(pname):
        d = policies[pname]
        return [a for a in (d.get("pros", []) + d.get("cons", []))
                if a in embs]

    pairs = []
    n_groups_used = 0
    for meta, pnames in by_meta.items():
        if len(pnames) < 2:
            continue  # need at least two policies to have cross-policy edges
        n_groups_used += 1
        # All unordered policy pairs within this metapolicy
        for i in range(len(pnames)):
            for j in range(i + 1, len(pnames)):
                a_ids = all_args(pnames[i])
                b_ids = all_args(pnames[j])
                if not a_ids or not b_ids:
                    continue
                A = np.stack([embs[a] for a in a_ids])
                B = np.stack([embs[b] for b in b_ids])
                sims = A @ B.T
                ai, bi = np.where(sims >= FLOOR_CONTRA)
                for x, y in zip(ai, bi):
                    pairs.append((meta, a_ids[x], b_ids[y],
                                  float(sims[x, y])))
    print(f"    metapolicy groups with >=2 policies: {n_groups_used}")
    return pairs


# ============================================================
# STAGE 2 — NLI CLASSIFICATION
# ============================================================
def classify_contradicts(pairs, texts):
    """For each (policy, a, b, cos) score contradiction via NLI in BOTH
    directions and keep the symmetric (min) contradiction probability.

    WHY BIDIRECTIONAL + MARGIN (the fix):
      Single-direction P(contradiction) is ~1.0 for almost any pair of
      opposing-stance arguments on the same topic, so a raw >=tau threshold
      admits a third of all pro x con pairs (a stance-opposition graph, not
      a rebuttal graph). Requiring MUTUAL contradiction (min of A->B and
      B->A) restores most of the selectivity, because a genuine rebuttal
      contradicts in both readings whereas a mere topical-opposition pair
      often does not. This mirrors the bidirectional discipline that already
      makes EQUIVALENT behave.

    We only ever read index 0 of the NLI tuple (P(contradiction)), which is
    unambiguous per the model's verified id2label (0=contradiction). We do
    NOT depend on the entail/neutral index order here, so the fix is robust
    regardless of how the common.py wrapper orders positions 1 and 2.
      nli_contra      = min(contra_fwd, contra_bwd)   # mutual conflict
      contra_min_pair = which direction was the weaker (for inspection)
    The margin test is applied at threshold time against TAU_CONTRA and an
    additional MIN_CONTRA_MARGIN over the non-contradiction mass (1-contra).
    """
    if not pairs:
        return []
    # Forward A->B
    fp = [texts[a] for (_, a, b, _) in pairs]
    fh = [texts[b] for (_, a, b, _) in pairs]
    # Backward B->A
    bp = [texts[b] for (_, a, b, _) in pairs]
    bh = [texts[a] for (_, a, b, _) in pairs]

    f_contra, b_contra = [], []
    for i in range(0, len(pairs), NLI_BATCH):
        fpr = nli_label_batch(fp[i:i+NLI_BATCH], fh[i:i+NLI_BATCH],
                              batch_size=NLI_BATCH)
        bpr = nli_label_batch(bp[i:i+NLI_BATCH], bh[i:i+NLI_BATCH],
                              batch_size=NLI_BATCH)
        f_contra.extend(p[0] for p in fpr)   # index 0 = contradiction (verified)
        b_contra.extend(p[0] for p in bpr)
        if (i // NLI_BATCH) % 20 == 0:
            print(f"    CONTRADICTS NLI: {min(i+NLI_BATCH, len(pairs))}/{len(pairs)}")

    out = []
    for (policy, a, b, cos), cf, cb in zip(pairs, f_contra, b_contra):
        contra_min = float(min(cf, cb))
        out.append({"policy": policy, "source": a, "target": b,
                    "cosine": cos,
                    "nli_contra": contra_min,            # symmetric (min)
                    "nli_contra_fwd": float(cf),
                    "nli_contra_bwd": float(cb)})
    return out


def classify_equivalent(pairs, texts):
    """Bidirectional entailment. Score A->B and B->A; keep min(entail)
    as the symmetric equivalence confidence."""
    if not pairs:
        return []
    # Forward A->B
    fp = [texts[a] for (_, a, b, _) in pairs]
    fh = [texts[b] for (_, a, b, _) in pairs]
    # Backward B->A
    bp = [texts[b] for (_, a, b, _) in pairs]
    bh = [texts[a] for (_, a, b, _) in pairs]

    fwd, bwd = [], []
    for i in range(0, len(pairs), NLI_BATCH):
        fprobs = nli_label_batch(fp[i:i+NLI_BATCH], fh[i:i+NLI_BATCH],
                                 batch_size=NLI_BATCH)
        bprobs = nli_label_batch(bp[i:i+NLI_BATCH], bh[i:i+NLI_BATCH],
                                 batch_size=NLI_BATCH)
        fwd.extend(p[1] for p in fprobs)  # entailment index = 1
        bwd.extend(p[1] for p in bprobs)
        if (i // NLI_BATCH) % 20 == 0:
            print(f"    EQUIVALENT NLI: {min(i+NLI_BATCH, len(pairs))}/{len(pairs)}")

    out = []
    for (policy, a, b, cos), ef, eb in zip(pairs, fwd, bwd):
        out.append({"policy": policy, "source": a, "target": b,
                    "cosine": cos,
                    "nli_entail_fwd": float(ef), "nli_entail_bwd": float(eb),
                    "nli_entail_min": float(min(ef, eb))})
    return out


# ============================================================
# STAGE 3 — OPTIONAL LLM ADJUDICATION
# ============================================================
CONTRA_JUDGE_SYSTEM = """You are an expert in argumentation. You will see two
arguments about the same policy. Decide whether they are in DIRECT CONFLICT —
i.e. they cannot both be accepted because one attacks or negates the claim or
reasoning of the other.

Answer with EXACTLY ONE LETTER on the first line:
  A = they are in direct conflict (CONTRADICTS)
  B = they are not in direct conflict
Then one short sentence. No other text."""

EQUIV_JUDGE_SYSTEM = """You are an expert in argumentation. You will see two
arguments about the same policy. Decide whether they make essentially the
SAME point — the same claim and reasoning, just phrased differently
(paraphrase / equivalent).

Answer with EXACTLY ONE LETTER on the first line:
  A = they are equivalent (same point)
  B = they are not equivalent
Then one short sentence. No other text."""


def llm_judge(ollama, system_prompt, text_a, text_b, retries=2):
    user = f"Argument 1:\n{text_a}\n\nArgument 2:\n{text_b}\n\nAnswer A or B."
    for attempt in range(retries + 1):
        try:
            r = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user}],
                options={"temperature": 0.0, "num_predict": 60,
                         "num_ctx": 4096})
            ans = parse_ab(r["message"]["content"])
            if ans is not None:
                return ans == "A"   # True => relation holds
        except Exception as e:
            print(f"      llm judge error (try {attempt+1}): "
                  f"{type(e).__name__}: {e}")
            time.sleep(2)
    return None


def adjudicate(scored, texts, kind, ollama, scope=SCOPE, arg_policy=None):
    """Resolve each scored candidate into accept/reject.

    kind  = "contra" or "equiv". Confident NLI verdicts decide directly;
    the uncertain band [UNCERTAIN_LOW, UNCERTAIN_HIGH) is escalated to the
    LLM when adjudication is enabled. Returns accepted edges.

    scope = "within_policy" or "within_metapolicy". For the metapolicy tier,
    the candidate's tag slot holds the metapolicy name; pass arg_policy
    ({arg_id: policy_name}) so each edge also records the two source
    policies (needed by the Coherence Audit to report which domains clash).
    """
    accepted = []
    sysprompt = CONTRA_JUDGE_SYSTEM if kind == "contra" else EQUIV_JUDGE_SYSTEM
    n_escalated = 0

    for c in scored:
        if kind == "contra":
            score = c["nli_contra"]
            tau = TAU_CONTRA
        else:
            score = c["nli_entail_min"]
            tau = TAU_ENTAIL

        method = "nli"
        accept = score >= tau

        if USE_LLM_ADJUDICATION and UNCERTAIN_LOW <= score < UNCERTAIN_HIGH:
            verdict = llm_judge(ollama, sysprompt,
                                texts[c["source"]], texts[c["target"]])
            n_escalated += 1
            if verdict is not None:
                accept = verdict
                method = "nli+llm"

        if accept:
            edge = {"source": c["source"], "target": c["target"],
                    "scope": scope,
                    "method": method, "cosine": round(c["cosine"], 4)}
            if scope == "within_metapolicy":
                edge["metapolicy"] = c["policy"]   # tag slot holds metapolicy
                if arg_policy is not None:
                    edge["source_policy"] = arg_policy.get(c["source"])
                    edge["target_policy"] = arg_policy.get(c["target"])
            else:
                edge["policy"] = c["policy"]
            if kind == "contra":
                edge["confidence"] = round(c["nli_contra"], 4)  # = min(fwd,bwd)
                if "nli_contra_fwd" in c:
                    edge["contra_fwd"] = round(c["nli_contra_fwd"], 4)
                    edge["contra_bwd"] = round(c["nli_contra_bwd"], 4)
            else:
                edge["confidence"] = round(c["nli_entail_min"], 4)
                edge["entail_fwd"] = round(c["nli_entail_fwd"], 4)
                edge["entail_bwd"] = round(c["nli_entail_bwd"], 4)
            accepted.append(edge)

    if USE_LLM_ADJUDICATION:
        print(f"    {kind}: escalated {n_escalated} uncertain pairs to LLM")
    return accepted


# ============================================================
# SYMMETRY + CONFLICT RESOLUTION
# ============================================================
def pair_key(a, b):
    return tuple(sorted((a, b)))


def resolve(contra_edges, equiv_edges):
    """Deduplicate symmetric pairs and resolve CONTRADICTS/EQUIVALENT
    conflicts by keeping the higher-confidence verdict."""
    # Deduplicate within each relation (blocking already used unordered
    # pairs, but be safe).
    def dedup(edges):
        best = {}
        for e in edges:
            k = pair_key(e["source"], e["target"])
            if k not in best or e["confidence"] > best[k]["confidence"]:
                best[k] = e
        return best

    contra = dedup(contra_edges)
    equiv  = dedup(equiv_edges)

    # Cross-relation conflict: a pair cannot be both.
    conflicts = set(contra.keys()) & set(equiv.keys())
    for k in conflicts:
        if contra[k]["confidence"] >= equiv[k]["confidence"]:
            del equiv[k]
        else:
            del contra[k]

    return list(contra.values()), list(equiv.values()), len(conflicts)


# ============================================================
# VALIDATION SAMPLE
# ============================================================
def dump_labeling_sample(contra_scored, equiv_scored, texts, n):
    import random
    random.seed(42)

    def make(scored, kind):
        rows = []
        pick = random.sample(scored, min(n, len(scored)))
        for c in pick:
            row = {"policy": c["policy"],
                   "text_a": texts[c["source"]],
                   "text_b": texts[c["target"]],
                   "cosine": round(c["cosine"], 4),
                   "label": ""}   # human fills: 1 = relation holds, 0 = not
            if kind == "contra":
                row["nli_contra"] = round(c["nli_contra"], 4)
            else:
                row["nli_entail_min"] = round(c["nli_entail_min"], 4)
            rows.append(row)
        return rows

    sample = {"instructions": ("Fill 'label' with 1 if the relation holds, "
                               "0 if not. Then compute precision/recall vs "
                               "the accepted edges at the chosen thresholds."),
              "contradicts_candidates": make(contra_scored, "contra"),
              "equivalent_candidates":  make(equiv_scored, "equiv")}
    path = "data/relations_sample_for_labeling.json"
    save_json(sample, path)
    print(f"\n  Wrote labeling sample ({n} per relation) -> {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    global USE_LLM_ADJUDICATION

    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="enable LLM adjudication on the uncertain band")
    ap.add_argument("--coherence", action="store_true",
                    help="also generate within-metapolicy cross-policy "
                         "CONTRADICTS for the Coherence Audit (UC4)")
    ap.add_argument("--sample", type=int, default=0,
                    help="dump N candidates per relation for hand labeling")
    ap.add_argument("--merge", type=str, default="",
                    help="also write a copy of the export with edges merged "
                         "into export['edges'] at this path")
    args = ap.parse_args()
    USE_LLM_ADJUDICATION = args.llm

    print("=" * 70)
    print("RELATION GENERATION — CONTRADICTS + EQUIVALENT")
    print("=" * 70)
    print(f"Blocking floors : contra cosine>={FLOOR_CONTRA}, "
          f"equiv cosine>={FLOOR_EQUIV}")
    print(f"NLI thresholds  : contra P>={TAU_CONTRA}, "
          f"equiv entail(both)>={TAU_ENTAIL}")
    print(f"LLM adjudication: {'ON' if USE_LLM_ADJUDICATION else 'OFF'}")
    print(f"Coherence tier  : {'ON' if args.coherence else 'OFF'} "
          f"(within-metapolicy cross-policy CONTRADICTS)")
    print("=" * 70)

    export = load_export()
    texts, embs = build_arg_index(export)
    print(f"\nIndexed {len(texts)} arguments with text+embedding")

    # --- Stage 1: blocking (cached) ---
    if os.path.exists(CANDIDATES_CACHE):
        print(f"Loading cached candidates from {CANDIDATES_CACHE}")
        cache = load_json(CANDIDATES_CACHE)
        contra_pairs = [tuple(x) for x in cache["contra_pairs"]]
        equiv_pairs  = [tuple(x) for x in cache["equiv_pairs"]]
    else:
        print("\nStage 1: blocking...")
        contra_pairs, equiv_pairs = block_candidates(export, texts, embs)
        save_json({"contra_pairs": contra_pairs, "equiv_pairs": equiv_pairs},
                  CANDIDATES_CACHE)
    print(f"  CONTRADICTS candidates: {len(contra_pairs):,}")
    print(f"  EQUIVALENT  candidates: {len(equiv_pairs):,}")

    # --- Stage 2: NLI ---
    print("\nStage 2: NLI classification...")
    contra_scored = classify_contradicts(contra_pairs, texts)
    equiv_scored  = classify_equivalent(equiv_pairs, texts)

    # Optional labeling sample (uses scored candidates, before thresholding)
    if args.sample > 0:
        dump_labeling_sample(contra_scored, equiv_scored, texts, args.sample)

    # --- Stage 3: adjudication / thresholding ---
    print("\nStage 3: adjudication + thresholding...")
    ollama = get_ollama() if USE_LLM_ADJUDICATION else None
    contra_edges = adjudicate(contra_scored, texts, "contra", ollama)
    equiv_edges  = adjudicate(equiv_scored, texts, "equiv", ollama)
    print(f"  Accepted (pre-resolve): {len(contra_edges):,} CONTRADICTS, "
          f"{len(equiv_edges):,} EQUIVALENT")

    # --- Symmetry + conflict resolution ---
    contra_final, equiv_final, n_conflicts = resolve(contra_edges, equiv_edges)
    print(f"  After resolve: {len(contra_final):,} CONTRADICTS, "
          f"{len(equiv_final):,} EQUIVALENT "
          f"({n_conflicts} cross-relation conflicts resolved)")

    # --- Coherence Audit tier (UC4): within-metapolicy cross-policy ---
    coherence_final = []
    if args.coherence:
        print("\nCoherence tier: within-metapolicy cross-policy CONTRADICTS...")
        # Map each argument id to its policy (for source/target_policy tags)
        arg_policy = {}
        for pname, data in export.get("policies", {}).items():
            for aid in (data.get("pros", []) + data.get("cons", [])):
                arg_policy[aid] = pname

        meta_pairs = block_metapolicy_contradicts(export, texts, embs)
        print(f"    candidates: {len(meta_pairs):,}")
        if meta_pairs:
            meta_scored = classify_contradicts(meta_pairs, texts)
            if ollama is None and USE_LLM_ADJUDICATION:
                ollama = get_ollama()
            meta_edges = adjudicate(meta_scored, texts, "contra", ollama,
                                    scope="within_metapolicy",
                                    arg_policy=arg_policy)
            # Deduplicate symmetric pairs (no cross-relation conflict here)
            best = {}
            for e in meta_edges:
                k = pair_key(e["source"], e["target"])
                if k not in best or e["confidence"] > best[k]["confidence"]:
                    best[k] = e
            coherence_final = list(best.values())
        print(f"    accepted: {len(coherence_final):,} coherence CONTRADICTS")

    # --- Write ---
    out = {
        "config": {
            "floor_contra": FLOOR_CONTRA, "floor_equiv": FLOOR_EQUIV,
            "tau_contra": TAU_CONTRA, "tau_entail": TAU_ENTAIL,
            "llm_adjudication": USE_LLM_ADJUDICATION,
            "uncertain_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
            "coherence_tier": args.coherence,
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "equivalent_definition": "bidirectional entailment (min of both directions)",
        },
        "contradicts": contra_final,
        "equivalent":  equiv_final,
        "coherence_contradicts": coherence_final,
        "stats": {
            "n_arguments_indexed": len(texts),
            "n_contra_candidates": len(contra_pairs),
            "n_equiv_candidates":  len(equiv_pairs),
            "n_contradicts": len(contra_final),
            "n_equivalent":  len(equiv_final),
            "n_coherence_contradicts": len(coherence_final),
            "n_conflicts_resolved": n_conflicts,
        },
    }
    save_json(out, OUTPUT_FILE)
    print(f"\nSaved -> {OUTPUT_FILE}")

    # --- Optional merge into an export copy ---
    if args.merge:
        export["edges"] = export.get("edges", {})
        export["edges"]["contradicts"] = contra_final
        export["edges"]["equivalent"]  = equiv_final
        if args.coherence:
            export["edges"]["coherence_contradicts"] = coherence_final
        save_json(export, args.merge)
        print(f"Merged edges into export copy -> {args.merge}")

    print("\nNext: hand-label data/relations_sample_for_labeling.json "
          "(if generated) and report precision/recall for the thesis.")


if __name__ == "__main__":
    main()