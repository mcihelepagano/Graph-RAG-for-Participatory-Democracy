"""
sweep_uc2_k.py  (PATCHED)
=========================
Focused k-sensitivity sweep for UC2 (debate summarization).
Pool size is FIXED at 200 (chosen from the UC1 sweep), only top_k varies.

PATCH APPLIED (vs the original):
  - The hard-coded POLICIES list (which had drifted — it contained
    "We should end mandatory retirement" and lacked "We should adopt a
    multi-party system") is REPLACED by `from policies import POLICIES`,
    the single shared 30-policy list. Every script now uses the
    identical policy set.
  - Nothing else changed. All checkpointing logic is untouched.

Pipeline per (system, k):
  1. Retrieval at top_k=k from a pool of 200
  2. Qwen3:8b structured 5-field summary
  3. Deterministic metrics (no LLM judge — that's stage 2)

Metrics computed:
  A. Coverage             — fraction of retrieved args entailed by summary (NLI)
  B. Balance distortion   — |summary pro/con ratio - source pro/con ratio|
  C. Conciseness          — sentences per 100 words (claim density)
  D. Inter-system divergence — how different are summaries across systems
                                at this k? Headline saturation signal:
                                  HIGH at low k -> retrieval differences survive
                                  LOW  at high k -> summarizer washes out retrieval

Heavy per-policy checkpointing — safe to interrupt and resume. Every
retrieval and every summary is written to disk as soon as it is
produced, so if the cluster job is killed the next run resumes exactly
where it stopped.

Sweeps one distractor-pool configuration per run, selected via the
POOL_CONFIG constant (canonical / nearest_L100 / with_distractor_edges
/ no_dd) — same mechanism as sweep_uc1_k.py. Sweep logic is identical
across all four; only the export file and output root differ.

OUTPUT TREE:
  results/sweep_uc2/<config_name>/          (<config_name> omitted for "canonical")
    retrieval/<system>_k<k>.json     raw retrievals
    summaries/<system>_k<k>.json     Qwen3 summaries
    metrics_deterministic.json       fast metrics
    sweep_summary.csv                flat CSV
    plots/*.png

Usage on cluster:
    sbatch run_sweep_uc2.sh
or directly:
    python sweep_uc2_k.py             # sweeps whatever POOL_CONFIG is set to
    # to sweep another configuration, edit POOL_CONFIG and rerun
"""

import os
import re
import json
import random
import time
import csv
import numpy as np
import torch
# Neo4j replaced by JSON export adapter for cluster portability.
from neo4j_export_adapter import (
    load_export,
    fetch_constrained_pool as _fetch_constrained_pool,
    fetch_system_b_pool as _fetch_system_b_pool,
)
from ollama import Client as OllamaClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # headless backend — safe for cluster / no display
import matplotlib.pyplot as plt

# PATCH: single shared policy list (replaces the old hard-coded list).
from policies import POLICIES


# ============================================================
# CONFIG
# ============================================================
# Which distractor-pool configuration to sweep. Sweep logic is identical
# across all of them — only the export file and output root differ.
# Change this one line and rerun to produce each configuration.
POOL_CONFIG = "nearest_L100"   # one of: "canonical", "nearest_L100",
                               #         "with_distractor_edges", "no_dd"

POOL_CONFIGS = {
    # Plain canonical export, no injected distractors.
    "canonical": {
        "export":   "data/neo4j_export_with_new_edges.json",
        "out_root": "results/sweep_uc2",
    },
    # Hardened pool, distractors as isolated nodes (no distractor edges).
    # Primary result — see thesis methodology.
    "nearest_L100": {
        "export":   "data/hard_pools/neo4j_export_distractors_nearest_L100.json",
        "out_root": "results/sweep_uc2/hard_pools_nearest_L100",
    },
    # Hardened pool, unrestricted distractor<->distractor CONTRADICTS edges.
    # PLACEHOLDER path — confirm the actual filename before running.
    "with_distractor_edges": {
        "export":   "data/hard_pools/neo4j_export_distractors_with_distractor_edges.json",
        "out_root": "results/sweep_uc2/hard_pools_with_distractor_edges",
    },
    # Hardened pool, corrected graph with zero distractor<->distractor edges.
    # PLACEHOLDER path — confirm the actual filename before running.
    "no_dd": {
        "export":   "data/hard_pools/neo4j_export_distractors_no_dd.json",
        "out_root": "results/sweep_uc2/hard_pools_no_dd",
    },
}

NEO4J_EXPORT_PATH = POOL_CONFIGS[POOL_CONFIG]["export"]

OLLAMA_HOST  = "http://127.0.0.1:11434"
SUMMARIZER   = "qwen3:8b"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"

# Sweep grid — POOL fixed at 200 (UC1 winner), top_k varies
POOL_SIZE     = 200
TOP_KS        = [5, 10, 15, 20, 30]
RANDOM_SEED   = 42

ENTAIL_THRESH = 0.5

# I/O
OUT_ROOT      = POOL_CONFIGS[POOL_CONFIG]["out_root"]
DIR_RETRIEVAL = os.path.join(OUT_ROOT, "retrieval")
DIR_SUMMARIES = os.path.join(OUT_ROOT, "summaries")
DIR_PLOTS     = os.path.join(OUT_ROOT, "plots")
F_DETERM      = os.path.join(OUT_ROOT, "metrics_deterministic.json")
F_CSV         = os.path.join(OUT_ROOT, "sweep_summary.csv")

SYSTEMS = ["System A", "System B", "Baseline A", "Baseline B"]


# ============================================================
# SETUP
# ============================================================
for d in [OUT_ROOT, DIR_RETRIEVAL, DIR_SUMMARIES, DIR_PLOTS]:
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading NLI model on {device}...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME).to(device).eval()
print(f"  NLI id2label: {nli_model.config.id2label}")
assert [nli_model.config.id2label[i].lower() for i in range(3)] == \
    ["contradiction", "entailment", "neutral"], \
    f"Unexpected NLI label order: {nli_model.config.id2label}"

ollama = OllamaClient(host=OLLAMA_HOST)


def get_driver():
    """Loads the JSON export; returns a GraphHandle with the same .close() API."""
    return load_export(NEO4J_EXPORT_PATH)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def calculate_mmr(texts, vecs, query_embedding=None, relevance_scores=None,
                  top_k=20, lambda_param=0.5):
    if not texts:
        return []
    if len(texts) <= top_k:
        return texts

    embeddings = np.array(vecs)

    if relevance_scores is not None:
        scores = np.array(relevance_scores, dtype=float)
        mn, mx = scores.min(), scores.max()
        query_sim = (scores - mn) / (mx - mn) if mx > mn else np.ones(len(scores))
    elif query_embedding is not None:
        q = np.array(query_embedding).reshape(1, -1)
        query_sim = cosine_similarity(embeddings, q).flatten()
    else:
        centroid  = embeddings.mean(axis=0, keepdims=True)
        query_sim = cosine_similarity(embeddings, centroid).flatten()

    selected   = [int(np.argmax(query_sim))]
    unselected = [i for i in range(len(texts)) if i != selected[0]]

    while len(selected) < top_k and unselected:
        best_score, best_idx = -np.inf, -1
        sel_embs = embeddings[selected]
        for i in unselected:
            diversity = float(np.max(
                cosine_similarity(embeddings[i].reshape(1, -1), sel_embs)
            ))
            score = lambda_param * query_sim[i] - (1 - lambda_param) * diversity
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        unselected.remove(best_idx)

    return [texts[i] for i in selected]


def cosine_topk(texts, vecs, query_vec, k):
    if not texts:
        return []
    sims = cosine_similarity(np.array(vecs), query_vec.reshape(1, -1)).flatten()
    top  = np.argsort(sims)[-k:][::-1]
    return [texts[i] for i in top]


def pool_centroid(vecs):
    """Mean embedding of a candidate pool — the query surrogate used by
    Baseline A. Matches System A's own centroid fallback in calculate_mmr
    (see the `else` branch there), so the only difference between System A
    and Baseline A is MMR's diversity term, not the query representation."""
    return np.array(vecs).mean(axis=0, keepdims=True)


def fetch_constrained_pool(driver, policy_name):
    """Pool of POOL_SIZE PRO + POOL_SIZE CON, plain. Delegates to adapter."""
    return _fetch_constrained_pool(driver, policy_name, pool_size=POOL_SIZE)


def fetch_system_b_pool(driver, policy_name):
    """Pool of POOL_SIZE per stance, with PageRank scores.
    Includes multi-hop via CONTRADICTS edges (matches UC1 System B exactly).
    Multi-hop cap = POOL_SIZE // 2 (matches the 20:10 ratio of the original
    Cypher query, scaled to the larger pool)."""
    hop_size = max(5, POOL_SIZE // 2)
    return _fetch_system_b_pool(driver, policy_name,
                                pool_size=POOL_SIZE, hop_size=hop_size)


# ============================================================
# PHASE 1 — RETRIEVAL
# ============================================================
def retrieval_path(system, k):
    safe = system.lower().replace(" ", "_")
    return os.path.join(DIR_RETRIEVAL, f"{safe}_k{k}.json")


def run_retrieval_for_k(driver, k):
    """Run all 4 systems at top_k=k. Per-policy checkpointed."""
    print(f"\n  Retrieval at top_k={k}")
    all_results = {sys: [] for sys in SYSTEMS}

    done_per_sys = {}
    for sys in SYSTEMS:
        path = retrieval_path(sys, k)
        if os.path.exists(path):
            existing = load_json(path)
            all_results[sys] = existing
            done_per_sys[sys] = {r["policy"] for r in existing}
        else:
            done_per_sys[sys] = set()

    for policy in tqdm(POLICIES, desc=f"  k={k} retrieval"):
        if all(policy in done_per_sys[sys] for sys in SYSTEMS):
            continue

        pa_t, pa_v, ca_t, ca_v = fetch_constrained_pool(driver, policy)
        pb_t, pb_v, pb_pr, cb_t, cb_v, cb_pr = fetch_system_b_pool(driver, policy)

        if policy not in done_per_sys["System A"]:
            pros = calculate_mmr(pa_t, pa_v, top_k=k)
            cons = calculate_mmr(ca_t, ca_v, top_k=k)
            all_results["System A"].append({
                "policy": policy, "retrieved_pros": pros, "retrieved_cons": cons,
            })
            save_json(all_results["System A"], retrieval_path("System A", k))

        if policy not in done_per_sys["System B"]:
            pros = calculate_mmr(pb_t, pb_v, relevance_scores=pb_pr, top_k=k)
            cons = calculate_mmr(cb_t, cb_v, relevance_scores=cb_pr, top_k=k)
            all_results["System B"].append({
                "policy": policy, "retrieved_pros": pros, "retrieved_cons": cons,
            })
            save_json(all_results["System B"], retrieval_path("System B", k))

        if policy not in done_per_sys["Baseline A"]:
            # Query surrogate = per-stance pool centroid, matching System
            # A's own centroid fallback — isolates MMR's diversity term as
            # the only difference between System A and Baseline A.
            pros = cosine_topk(pa_t, pa_v, pool_centroid(pa_v), k)
            cons = cosine_topk(ca_t, ca_v, pool_centroid(ca_v), k)
            all_results["Baseline A"].append({
                "policy": policy, "retrieved_pros": pros, "retrieved_cons": cons,
            })
            save_json(all_results["Baseline A"], retrieval_path("Baseline A", k))

        if policy not in done_per_sys["Baseline B"]:
            random.seed(f"{RANDOM_SEED}|{policy}")  # str seed: reproducible across processes (hash() is salted)
            pros = random.sample(pa_t, min(k, len(pa_t)))
            cons = random.sample(ca_t, min(k, len(ca_t)))
            all_results["Baseline B"].append({
                "policy": policy, "retrieved_pros": pros, "retrieved_cons": cons,
            })
            save_json(all_results["Baseline B"], retrieval_path("Baseline B", k))

    return all_results


# ============================================================
# PHASE 2 — SUMMARIZATION (Qwen3:8b structured)
# ============================================================
QWEN_SYSTEM = """You are an expert debate summarizer.
Analyze the provided policy topic and the retrieved PRO and CON arguments.
Generate a structured debate summary strictly as a JSON object.
You must use exactly these 5 keys:
{
  "STRONGEST_PRO": "...",
  "STRONGEST_CON": "...",
  "KEY_TENSION": "...",
  "MISSING_PERSPECTIVES": "...",
  "OVERALL_BALANCE": "..."
}
Be specific. KEY_TENSION must name the contradicting CLAIMS, not just the topic.
For example, prefer 'PROs argue X causes harm Y; CONs argue evidence for Y is weak.'
over 'There is tension between freedom and safety.'"""


def summary_path(system, k):
    safe = system.lower().replace(" ", "_")
    return os.path.join(DIR_SUMMARIES, f"{safe}_k{k}.json")


def parse_summary_json(raw):
    if not raw:
        return None
    raw = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    required = {"STRONGEST_PRO", "STRONGEST_CON", "KEY_TENSION",
                "MISSING_PERSPECTIVES", "OVERALL_BALANCE"}
    if not required.issubset(obj.keys()):
        return None
    return obj


def render_summary(fields):
    return (f"STRONGEST PRO: {fields['STRONGEST_PRO']}\n"
            f"STRONGEST CON: {fields['STRONGEST_CON']}\n"
            f"KEY TENSION: {fields['KEY_TENSION']}\n"
            f"MISSING PERSPECTIVES: {fields['MISSING_PERSPECTIVES']}\n"
            f"OVERALL BALANCE: {fields['OVERALL_BALANCE']}")


def generate_summary(policy, pros, cons, retries=2):
    user = (f"Policy Topic: {policy}\n\n"
            f"PRO ARGUMENTS:\n" + "\n".join(f"- {a}" for a in pros) +
            f"\n\nCON ARGUMENTS:\n" + "\n".join(f"- {a}" for a in cons))
    for attempt in range(retries + 1):
        try:
            r = ollama.chat(
                model=SUMMARIZER,
                messages=[{"role": "system", "content": QWEN_SYSTEM},
                          {"role": "user",   "content": user}],
                format="json",
                options={"temperature": 0.0},
            )
            obj = parse_summary_json(r["message"]["content"])
            if obj is not None:
                return obj
            time.sleep(1)
        except Exception as e:
            print(f"      summary error (try {attempt+1}): {e}")
            time.sleep(2)
    return None


def run_summarization_for_k(retrieval, k):
    print(f"\n  Summarization at top_k={k}")
    all_summaries = {}

    for system in SYSTEMS:
        path = summary_path(system, k)
        existing = load_json(path) if os.path.exists(path) else {}
        all_summaries[system] = existing
        records = retrieval[system]
        policy_map = {r["policy"]: r for r in records}

        for policy in tqdm(POLICIES, desc=f"  {system} k={k}"):
            if policy in existing:
                continue
            if policy not in policy_map:
                continue
            pros = policy_map[policy].get("retrieved_pros", [])
            cons = policy_map[policy].get("retrieved_cons", [])
            if not pros and not cons:
                continue

            fields = generate_summary(policy, pros, cons)
            if fields is None:
                print(f"      failed: {policy[:50]}")
                continue

            existing[policy] = {
                "fields":         fields,
                "summary_text":   render_summary(fields),
                "retrieved_pros": pros,
                "retrieved_cons": cons,
            }
            save_json(existing, path)

    return all_summaries


# ============================================================
# PHASE 3 — DETERMINISTIC METRICS
# ============================================================
def nli_label_batch(premises, hypotheses, batch_size=32):
    if not premises:
        return []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            bp = premises[i:i+batch_size]
            bh = hypotheses[i:i+batch_size]
            enc = nli_tokenizer(bp, bh, padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(device)
            logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
    # cross-encoder/nli-deberta-v3-base order = (contra, entail, neutral)
    return [(p[0], p[1], p[2]) for p in all_probs]


def compute_coverage(summary_text, args):
    if not args:
        return None
    probs = nli_label_batch([summary_text]*len(args), args)
    entail = [pr[1] for pr in probs]
    n_cov  = sum(1 for e in entail if e > ENTAIL_THRESH)
    return n_cov / len(args)


def estimate_summary_pro_ratio(summary_text, pros, cons, threshold=0.4):
    if not pros and not cons:
        return None
    args = pros + cons
    labels = ["pro"]*len(pros) + ["con"]*len(cons)
    probs = nli_label_batch([summary_text]*len(args), args)
    entail = [pr[1] for pr in probs]
    cov_pros = sum(1 for e, l in zip(entail, labels) if l == "pro" and e > threshold)
    cov_cons = sum(1 for e, l in zip(entail, labels) if l == "con" and e > threshold)
    if cov_pros + cov_cons == 0:
        return len(pros) / (len(pros) + len(cons))
    return cov_pros / (cov_pros + cov_cons)


def compute_inter_system_divergence(summaries_by_system, policy):
    """Mean pairwise cosine DISTANCE across systems' summaries for a policy.
    HIGHER = retrieval differences survive into summary (good — systems differ).
    LOWER  = summarizer saturated (bad — retrieval differences washed out).
    """
    texts = []
    for system in SYSTEMS:
        if policy in summaries_by_system.get(system, {}):
            texts.append(summaries_by_system[system][policy]["summary_text"])
    if len(texts) < 2:
        return None
    embs = embed_model.encode(texts, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False)
    sim = embs @ embs.T
    iu = np.triu_indices(len(embs), k=1)
    return 1.0 - float(np.mean(sim[iu]))


def compute_conciseness(summary_text):
    """Sentences per 100 words. Higher = denser claims per word."""
    if not summary_text:
        return None
    word_count = len(summary_text.split())
    sentence_count = max(1, len(re.findall(r"[.!?]+", summary_text)))
    if word_count == 0:
        return None
    return sentence_count / word_count * 100


def run_deterministic_metrics(summaries_per_k):
    """summaries_per_k: {k: {system: {policy: record}}}."""
    print("\n=== PHASE 3: Deterministic metrics ===")

    if os.path.exists(F_DETERM):
        results = load_json(F_DETERM)
    else:
        results = {}

    for k, summaries in summaries_per_k.items():
        k_str = str(k)
        if k_str not in results:
            results[k_str] = {}

        # Inter-system divergence — system-agnostic per-policy
        if "inter_system_divergence" not in results[k_str]:
            print(f"  Inter-system divergence at k={k}")
            divs = []
            for policy in POLICIES:
                d = compute_inter_system_divergence(summaries, policy)
                if d is not None:
                    divs.append(d)
            results[k_str]["inter_system_divergence"] = {
                "mean":       float(np.mean(divs)) if divs else None,
                "std":        float(np.std(divs))  if divs else None,
                "per_policy": divs,
            }
            save_json(results, F_DETERM)

        # Per-system metrics
        for system in SYSTEMS:
            sys_done = results[k_str].get(system, {}).get("done", False)
            if sys_done:
                continue

            print(f"  {system} k={k}: coverage + balance + conciseness")
            cov_scores       = []
            balance_distort  = []
            concise_scores   = []

            for policy in tqdm(POLICIES, desc=f"  {system} k={k}"):
                if policy not in summaries[system]:
                    continue
                rec = summaries[system][policy]
                summary_text = rec["summary_text"]
                pros = rec["retrieved_pros"]
                cons = rec["retrieved_cons"]
                args = pros + cons
                if not args:
                    continue

                cov = compute_coverage(summary_text, args)
                if cov is not None:
                    cov_scores.append(cov)

                src_ratio = len(pros) / len(args) if args else 0.5
                sum_ratio = estimate_summary_pro_ratio(summary_text, pros, cons)
                if sum_ratio is not None:
                    balance_distort.append(abs(sum_ratio - src_ratio))

                conc = compute_conciseness(summary_text)
                if conc is not None:
                    concise_scores.append(conc)

            results[k_str][system] = {
                "mean_coverage":           float(np.mean(cov_scores))      if cov_scores      else None,
                "mean_balance_distortion": float(np.mean(balance_distort)) if balance_distort else None,
                "mean_conciseness":        float(np.mean(concise_scores))  if concise_scores  else None,
                "n_policies":              len(cov_scores),
                "done":                    True,
            }
            save_json(results, F_DETERM)

    return results


# ============================================================
# PHASE 4 — REPORTING
# ============================================================
def write_csv(determ):
    rows = []
    for k_str, k_metrics in determ.items():
        k = int(k_str)
        for system in SYSTEMS:
            if system not in k_metrics:
                continue
            m = k_metrics[system]
            rows.append({
                "top_k":                   k,
                "system":                  system,
                "coverage":                m.get("mean_coverage"),
                "balance_distortion":      m.get("mean_balance_distortion"),
                "conciseness":             m.get("mean_conciseness"),
                "inter_system_divergence": k_metrics.get(
                    "inter_system_divergence", {}).get("mean"),
            })
    rows.sort(key=lambda r: (r["top_k"], r["system"]))
    with open(F_CSV, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"  wrote {F_CSV}")


def plot_curve(determ, metric_key, title, ylabel, filename,
               lower_is_better=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"System A": "tab:blue", "System B": "tab:green",
              "Baseline A": "tab:orange", "Baseline B": "tab:red"}

    for system in SYSTEMS:
        ks, vals = [], []
        for k in TOP_KS:
            v = determ.get(str(k), {}).get(system, {}).get(metric_key)
            if v is not None:
                ks.append(k)
                vals.append(v)
        if vals:
            ax.plot(ks, vals, marker="o", color=colors[system], label=system)

    direction = " (down better)" if lower_is_better else " (up better)"
    ax.set_xlabel("top_k")
    ax.set_ylabel(ylabel + direction)
    ax.set_title(title + f"  (pool={POOL_SIZE})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(DIR_PLOTS, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  saved {out_path}")


def plot_inter_system_divergence(determ):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks, means, stds = [], [], []
    for k in TOP_KS:
        d = determ.get(str(k), {}).get("inter_system_divergence", {})
        if d.get("mean") is not None:
            ks.append(k)
            means.append(d["mean"])
            stds.append(d.get("std") or 0)
    if ks:
        ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, color="tab:purple")
    ax.set_xlabel("top_k")
    ax.set_ylabel("Mean inter-system summary divergence")
    ax.set_title(f"Summary saturation curve  (pool={POOL_SIZE})\n"
                 "High = retrieval differences survive; Low = summarizer saturated")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(DIR_PLOTS, "inter_system_divergence.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  saved {out_path}")


def report(determ):
    print("\n=== PHASE 4: Reporting ===")
    write_csv(determ)
    plot_curve(determ, "mean_coverage",
               "Coverage of retrieved args by summary",
               "Mean coverage",
               "coverage_vs_k.png", lower_is_better=False)
    plot_curve(determ, "mean_balance_distortion",
               "Balance distortion (summary vs source pro/con ratio)",
               "Mean abs delta pro/con ratio",
               "balance_distortion_vs_k.png", lower_is_better=True)
    plot_curve(determ, "mean_conciseness",
               "Conciseness (sentences per 100 words)",
               "Sentences / 100 words",
               "conciseness_vs_k.png", lower_is_better=False)
    plot_inter_system_divergence(determ)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("UC2 K-SENSITIVITY SWEEP (focused, pool fixed at 200)")
    print(f"  top_k grid : {TOP_KS}")
    print(f"  pool size  : {POOL_SIZE}")
    print(f"  systems    : {SYSTEMS}")
    print(f"  policies   : {len(POLICIES)} (shared list)")
    print("=" * 65)

    # Phase 1: Retrieval
    print("\n=== PHASE 1: Retrieval ===")
    driver = get_driver()
    retrieval_per_k = {}
    try:
        for k in TOP_KS:
            retrieval_per_k[k] = run_retrieval_for_k(driver, k)
    finally:
        driver.close()

    # Phase 2: Summarization
    print("\n=== PHASE 2: Summarization ===")
    for k in TOP_KS:
        run_summarization_for_k(retrieval_per_k[k], k)

    # Reload all summaries from disk
    print("\n  Reloading summaries from disk...")
    summaries_per_k = {}
    for k in TOP_KS:
        summaries_per_k[k] = {}
        for system in SYSTEMS:
            path = summary_path(system, k)
            if os.path.exists(path):
                summaries_per_k[k][system] = load_json(path)
            else:
                summaries_per_k[k][system] = {}

    # Phase 3: Metrics
    determ = run_deterministic_metrics(summaries_per_k)

    # Phase 4: Report
    report(determ)

    print("\n=== DONE ===")
    print(f"  Output root: {OUT_ROOT}")
    print(f"  CSV:        {F_CSV}")
    print(f"  Plots:      {DIR_PLOTS}")


if __name__ == "__main__":
    main()