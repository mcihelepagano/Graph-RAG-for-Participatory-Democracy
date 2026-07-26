"""
generate_relations_l100_distractors_no_dd_edges.py
====================================================
CORRECTED robustness-run variant: excludes DISTRACTOR<->DISTRACTOR pairs
from CONTRADICTS/EQUIVALENT generation entirely.

WHY THIS EXISTS (the bug this fixes).
    generate_relations_l100_distractors.py ran blocking over the FULL
    pro/con lists of the L100 export — native and distractor arguments
    alike, with no distinction between them. Inspection of System B's
    actual top-5 retrievals (via inspect_distractor_retrievals.py)
    showed this produced a specific, non-realistic artifact: distractor
    injection pulls in BOTH stances of a sibling policy's debate
    together (e.g. both the pro AND con of "We should ban telemarketing"
    get injected into the "autonomous cars" pool). Relation generation
    then correctly finds that these two transplanted arguments
    genuinely contradict EACH OTHER (conf ~0.99-1.0) — because they do,
    as the ORIGINAL telemarketing debate — but that contradiction has
    nothing to do with the target policy, and could never arise in a
    real deployed system (real arguments filed under a policy are
    actually about that policy; there's no mechanism to transplant a
    coherent unrelated debate wholesale into a real corpus). 40,763 of
    385,872 CONTRADICTS edges (10.6%) were this distractor-distractor
    type, vs. only 553 (0.1%) genuine native<->distractor edges — and
    per-example inspection showed System B's retrieved distractors were
    almost always connected via the FORMER, not the latter.

    So the prior robustness run answered a different question than the
    one it was designed to ask. The intended question was: "if
    distractors had realistic connectivity (rather than being
    artificially isolated), would System B still avoid them?" What
    actually got tested was: "if distractors form self-contained,
    mutually-reinforcing cliques from wholesale-transplanted unrelated
    debates, does that inflate their PageRank enough to fool System B?"
    — a real question, but a different and less representative one, and
    conflating the two overstates how much this says about System B's
    real-world graph-based-relevance vulnerability.

THIS SCRIPT.
    Identical three-stage pipeline (blocking -> bidirectional NLI ->
    threshold), identical thresholds (FLOOR_CONTRA=0.70, TAU_CONTRA=0.90,
    FLOOR_EQUIV=0.75, TAU_ENTAIL=0.60) to both prior runs. The ONLY
    change: Stage 1 blocking now checks each candidate pair against the
    export's own `is_distractor` flag (present directly on each argument
    record — no manifest text-matching needed) and SKIPS any pair where
    BOTH endpoints are distractors, for BOTH relations (CONTRADICTS
    cross-stance AND EQUIVALENT within-stance), before any NLI cost is
    spent on it. Native<->distractor and native<->native pairs are
    unaffected — this isolates exactly the genuine-connectivity question,
    without also cheapening the NLI compute spent on pairs that get
    discarded either way.

    Skipped-pair counts are reported explicitly (both at blocking time
    and in the output stats block) so the comparison against the
    all-pairs run is auditable, not just asserted.

INPUT  : data/hard_pools/neo4j_export_distractors_nearest_L100.json
OUTPUT : data/relations_generated_l100_distractors_no_dd_edges.json
CACHE  : data/relations_candidates_l100_distractors_no_dd_edges.json
MERGE  : data/neo4j_export_l100_distractor_edges_no_dd.json (default)

AFTER THIS SCRIPT — same downstream sequence as before, pointed at the
new merged export:
    python refresh_pagerank_and_reset.py \\
        --export data/neo4j_export_l100_distractor_edges_no_dd.json \\
        --expected-contradicts <see stats.n_contradicts printed below> \\
        --skip-cleanup --apply
    (then re-run sweep_system_b_l100_distractor_edges.py and
     eval_system_b_l100_distractor_rate.py, pointed at the new export —
     both scripts will need their NEO4J_EXPORT_PATH / RETRIEVAL_FILE
     constants updated to the "_no_dd" paths, or copy them to new
     "_no_dd" variants the same way this file was derived.)

Usage:
    python generate_relations_l100_distractors_no_dd_edges.py
    python generate_relations_l100_distractors_no_dd_edges.py --sample 150
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
# CONFIG — fixed to the L100 distractor-injected robustness run,
# distractor-distractor pairs excluded
# ============================================================
NEO4J_EXPORT_FILE = "data/hard_pools/neo4j_export_distractors_nearest_L100.json"
OUTPUT_FILE       = "data/relations_generated_l100_distractors_no_dd_edges.json"
CANDIDATES_CACHE  = "data/relations_candidates_l100_distractors_no_dd_edges.json"
DEFAULT_MERGE_OUT = "data/neo4j_export_l100_distractor_edges_no_dd.json"

FLOOR_CONTRA = 0.70
FLOOR_EQUIV  = 0.75
TAU_CONTRA = 0.90
TAU_ENTAIL = 0.60

UNCERTAIN_LOW  = 0.40
UNCERTAIN_HIGH = 0.70

USE_LLM_ADJUDICATION = False
LLM_MODEL = "llama3.3:70b"

NLI_BATCH = 64
SCOPE = "within_policy"


# ============================================================
# LOAD
# ============================================================
def load_export():
    return load_json(NEO4J_EXPORT_FILE)


def build_arg_index(export):
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


def build_distractor_flags(export):
    """{arg_id: bool} directly from the export's own is_distractor field
    (confirmed present on every argument record — no manifest
    text-matching needed here, unlike the earlier audit scripts which
    predated knowing this field existed)."""
    return {aid: bool(a.get("is_distractor", False))
            for aid, a in export.get("arguments", {}).items()}


# ============================================================
# STAGE 1 — BLOCKING (distractor-distractor pairs excluded)
# ============================================================
def block_candidates(export, texts, embs, is_distractor):
    """Same as the base script, EXCEPT: any candidate pair where BOTH
    endpoints are distractors is skipped before it's even added to the
    candidate list (cheapest possible place to cut it — no wasted NLI
    calls on pairs we're discarding regardless). Counts are tracked and
    reported so the exclusion is auditable, not silent."""
    policies = export.get("policies", {})
    contra_pairs, equiv_pairs = [], []
    n_contra_dd_skipped = 0
    n_equiv_dd_skipped = 0

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
                a_id, b_id = pro_ids[p], con_ids[c]
                if is_distractor.get(a_id, False) and is_distractor.get(b_id, False):
                    n_contra_dd_skipped += 1
                    continue
                contra_pairs.append((policy, a_id, b_id, float(sims[p, c])))

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
                a_id, b_id = stance_ids[r], stance_ids[c]
                if is_distractor.get(a_id, False) and is_distractor.get(b_id, False):
                    n_equiv_dd_skipped += 1
                    continue
                equiv_pairs.append((policy, a_id, b_id, float(sims[r, c])))

    print(f"  Skipped distractor<->distractor candidates: "
          f"{n_contra_dd_skipped:,} CONTRADICTS, "
          f"{n_equiv_dd_skipped:,} EQUIVALENT (excluded before NLI)")
    return contra_pairs, equiv_pairs, n_contra_dd_skipped, n_equiv_dd_skipped


# ============================================================
# STAGE 2 — NLI CLASSIFICATION (unchanged from base script)
# ============================================================
def classify_contradicts(pairs, texts):
    if not pairs:
        return []
    fp = [texts[a] for (_, a, b, _) in pairs]
    fh = [texts[b] for (_, a, b, _) in pairs]
    bp = [texts[b] for (_, a, b, _) in pairs]
    bh = [texts[a] for (_, a, b, _) in pairs]

    f_contra, b_contra = [], []
    for i in range(0, len(pairs), NLI_BATCH):
        fpr = nli_label_batch(fp[i:i+NLI_BATCH], fh[i:i+NLI_BATCH],
                              batch_size=NLI_BATCH)
        bpr = nli_label_batch(bp[i:i+NLI_BATCH], bh[i:i+NLI_BATCH],
                              batch_size=NLI_BATCH)
        f_contra.extend(p[0] for p in fpr)
        b_contra.extend(p[0] for p in bpr)
        if (i // NLI_BATCH) % 20 == 0:
            print(f"    CONTRADICTS NLI: {min(i+NLI_BATCH, len(pairs))}/{len(pairs)}")

    out = []
    for (policy, a, b, cos), cf, cb in zip(pairs, f_contra, b_contra):
        contra_min = float(min(cf, cb))
        out.append({"policy": policy, "source": a, "target": b,
                    "cosine": cos, "nli_contra": contra_min,
                    "nli_contra_fwd": float(cf), "nli_contra_bwd": float(cb)})
    return out


def classify_equivalent(pairs, texts):
    if not pairs:
        return []
    fp = [texts[a] for (_, a, b, _) in pairs]
    fh = [texts[b] for (_, a, b, _) in pairs]
    bp = [texts[b] for (_, a, b, _) in pairs]
    bh = [texts[a] for (_, a, b, _) in pairs]

    fwd, bwd = [], []
    for i in range(0, len(pairs), NLI_BATCH):
        fprobs = nli_label_batch(fp[i:i+NLI_BATCH], fh[i:i+NLI_BATCH],
                                 batch_size=NLI_BATCH)
        bprobs = nli_label_batch(bp[i:i+NLI_BATCH], bh[i:i+NLI_BATCH],
                                 batch_size=NLI_BATCH)
        fwd.extend(p[1] for p in fprobs)
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
# STAGE 3 — OPTIONAL LLM ADJUDICATION (unchanged, off by default)
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
                return ans == "A"
        except Exception as e:
            print(f"      llm judge error (try {attempt+1}): "
                  f"{type(e).__name__}: {e}")
            time.sleep(2)
    return None


def adjudicate(scored, texts, kind, ollama):
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
                    "scope": SCOPE, "method": method,
                    "cosine": round(c["cosine"], 4), "policy": c["policy"]}
            if kind == "contra":
                edge["confidence"] = round(c["nli_contra"], 4)
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
# SYMMETRY + CONFLICT RESOLUTION (unchanged)
# ============================================================
def pair_key(a, b):
    return tuple(sorted((a, b)))


def resolve(contra_edges, equiv_edges):
    def dedup(edges):
        best = {}
        for e in edges:
            k = pair_key(e["source"], e["target"])
            if k not in best or e["confidence"] > best[k]["confidence"]:
                best[k] = e
        return best

    contra = dedup(contra_edges)
    equiv  = dedup(equiv_edges)

    conflicts = set(contra.keys()) & set(equiv.keys())
    for k in conflicts:
        if contra[k]["confidence"] >= equiv[k]["confidence"]:
            del equiv[k]
        else:
            del contra[k]

    return list(contra.values()), list(equiv.values()), len(conflicts)


# ============================================================
# VALIDATION SAMPLE (unchanged)
# ============================================================
def dump_labeling_sample(contra_scored, equiv_scored, texts, n):
    import random
    random.seed(42)

    def make(scored, kind):
        rows = []
        pick = random.sample(scored, min(n, len(scored)))
        for c in pick:
            row = {"policy": c["policy"], "text_a": texts[c["source"]],
                   "text_b": texts[c["target"]],
                   "cosine": round(c["cosine"], 4), "label": ""}
            if kind == "contra":
                row["nli_contra"] = round(c["nli_contra"], 4)
            else:
                row["nli_entail_min"] = round(c["nli_entail_min"], 4)
            rows.append(row)
        return rows

    sample = {"instructions": ("Fill 'label' with 1 if the relation holds, "
                               "0 if not."),
              "contradicts_candidates": make(contra_scored, "contra"),
              "equivalent_candidates":  make(equiv_scored, "equiv")}
    path = "data/relations_sample_for_labeling_no_dd.json"
    save_json(sample, path)
    print(f"\n  Wrote labeling sample ({n} per relation) -> {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    global USE_LLM_ADJUDICATION

    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--merge", type=str, default=DEFAULT_MERGE_OUT)
    args = ap.parse_args()
    USE_LLM_ADJUDICATION = args.llm

    print("=" * 70)
    print("RELATION GENERATION — L100 DISTRACTOR-DISTRACTOR EDGES EXCLUDED")
    print("=" * 70)
    print(f"Export in       : {NEO4J_EXPORT_FILE}")
    print(f"Output          : {OUTPUT_FILE}")
    print(f"Candidates cache: {CANDIDATES_CACHE}")
    print(f"Merge target    : {args.merge}")
    print(f"Blocking floors : contra cosine>={FLOOR_CONTRA}, "
          f"equiv cosine>={FLOOR_EQUIV}")
    print(f"NLI thresholds  : contra P>={TAU_CONTRA}, "
          f"equiv entail(both)>={TAU_ENTAIL}")
    print("EXCLUSION       : distractor<->distractor pairs dropped at "
          "blocking, for BOTH relations")
    print("=" * 70)

    export = load_export()
    texts, embs = build_arg_index(export)
    is_distractor = build_distractor_flags(export)
    n_distractors = sum(is_distractor.values())
    print(f"\nIndexed {len(texts)} arguments with text+embedding "
          f"({n_distractors} flagged is_distractor=True)")

    if os.path.exists(CANDIDATES_CACHE):
        print(f"Loading cached candidates from {CANDIDATES_CACHE}")
        cache = load_json(CANDIDATES_CACHE)
        contra_pairs = [tuple(x) for x in cache["contra_pairs"]]
        equiv_pairs  = [tuple(x) for x in cache["equiv_pairs"]]
        n_contra_dd_skipped = cache.get("n_contra_dd_skipped", 0)
        n_equiv_dd_skipped = cache.get("n_equiv_dd_skipped", 0)
    else:
        print("\nStage 1: blocking (distractor-distractor pairs excluded)...")
        (contra_pairs, equiv_pairs,
         n_contra_dd_skipped, n_equiv_dd_skipped) = block_candidates(
            export, texts, embs, is_distractor)
        save_json({"contra_pairs": contra_pairs, "equiv_pairs": equiv_pairs,
                  "n_contra_dd_skipped": n_contra_dd_skipped,
                  "n_equiv_dd_skipped": n_equiv_dd_skipped},
                  CANDIDATES_CACHE)
    print(f"  CONTRADICTS candidates: {len(contra_pairs):,}")
    print(f"  EQUIVALENT  candidates: {len(equiv_pairs):,}")

    print("\nStage 2: NLI classification...")
    contra_scored = classify_contradicts(contra_pairs, texts)
    equiv_scored  = classify_equivalent(equiv_pairs, texts)

    if args.sample > 0:
        dump_labeling_sample(contra_scored, equiv_scored, texts, args.sample)

    print("\nStage 3: adjudication + thresholding...")
    ollama = get_ollama() if USE_LLM_ADJUDICATION else None
    contra_edges = adjudicate(contra_scored, texts, "contra", ollama)
    equiv_edges  = adjudicate(equiv_scored, texts, "equiv", ollama)
    print(f"  Accepted (pre-resolve): {len(contra_edges):,} CONTRADICTS, "
          f"{len(equiv_edges):,} EQUIVALENT")

    contra_final, equiv_final, n_conflicts = resolve(contra_edges, equiv_edges)
    print(f"  After resolve: {len(contra_final):,} CONTRADICTS, "
          f"{len(equiv_final):,} EQUIVALENT "
          f"({n_conflicts} cross-relation conflicts resolved)")

    out = {
        "config": {
            "floor_contra": FLOOR_CONTRA, "floor_equiv": FLOOR_EQUIV,
            "tau_contra": TAU_CONTRA, "tau_entail": TAU_ENTAIL,
            "llm_adjudication": USE_LLM_ADJUDICATION,
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "equivalent_definition": "bidirectional entailment (min of both directions)",
            "source_export": NEO4J_EXPORT_FILE,
            "distractor_distractor_pairs_excluded": True,
        },
        "contradicts": contra_final,
        "equivalent":  equiv_final,
        "stats": {
            "n_arguments_indexed": len(texts),
            "n_distractors_indexed": n_distractors,
            "n_contra_candidates": len(contra_pairs),
            "n_equiv_candidates":  len(equiv_pairs),
            "n_contra_dd_pairs_excluded": n_contra_dd_skipped,
            "n_equiv_dd_pairs_excluded": n_equiv_dd_skipped,
            "n_contradicts": len(contra_final),
            "n_equivalent":  len(equiv_final),
            "n_conflicts_resolved": n_conflicts,
        },
    }
    save_json(out, OUTPUT_FILE)
    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"For comparison against the unrestricted run: "
          f"{len(contra_final):,} CONTRADICTS here vs 385,872 there "
          f"({n_contra_dd_skipped:,} distractor-distractor pairs were "
          f"excluded before NLI even ran on them).")

    if args.merge:
        export["edges"] = export.get("edges", {})
        export["edges"]["contradicts"] = contra_final
        export["edges"]["equivalent"]  = equiv_final
        save_json(export, args.merge)
        print(f"Merged edges into export copy -> {args.merge}")

    print(f"\nNext: refresh_pagerank_and_reset.py --export {args.merge} "
          f"--expected-contradicts {len(contra_final)} --skip-cleanup --apply")


if __name__ == "__main__":
    main()