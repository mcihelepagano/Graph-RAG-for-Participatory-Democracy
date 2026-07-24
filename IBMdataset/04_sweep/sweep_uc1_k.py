"""
sweep_uc1_k.py  (FINAL — JSON export, per-policy checkpointing)
===============================================================
Sensitivity sweep for UC1 retrieval: varies candidate pool size and
top-k across a grid, evaluates every configuration with deterministic
metrics (no LLM judge — that is stage 2 / eval_uc1.py).

KEY PROPERTIES
--------------
  - Uses the JSON export adapter (no live Neo4j dependency) — same as
    sweep_uc2_k.py. Cluster-safe.
  - Writes raw per-policy retrieval JSON files to
      results/sweep_uc1/configs/pool<P>_k<K>/<system>.json
    These are REQUIRED by eval_uc1.py. The previous cluster version
    omitted this write — that is why the configs/ directory was empty.
  - PER-POLICY checkpointing on the retrieval files: if the job is
    killed mid-config, the next run continues from the last completed
    policy. The per-config metrics in sweep_summary.json are only
    written once ALL policies for that config are done.
  - Uses `from policies import POLICIES` — shared 30-policy list.
  - Headless matplotlib (Agg backend).
  - Sweeps one distractor-pool configuration per run, selected via the
    POOL_CONFIG constant (canonical / nearest_L100 / with_distractor_edges
    / no_dd). Sweep logic is identical across all four — only the export
    file and output root differ. Rerun once per configuration.

OUTPUT TREE
-----------
  results/sweep_uc1/<config_name>/          (<config_name> omitted for "canonical")
    configs/pool<P>_k<K>/
      system_a.json          ← required by eval_uc1.py
      system_b.json
      baseline_a.json
      baseline_b.json
    sweep_summary.json       aggregated metrics
    sweep_summary.csv
    plots/*.png

Usage:
    python sweep_uc1_k.py                 # sweeps whatever POOL_CONFIG is set to
    # to sweep another configuration, edit POOL_CONFIG and rerun
"""

import os
import json
import random
import numpy as np
from itertools import product
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neo4j_export_adapter import (
    load_export,
    fetch_constrained_pool as _fetch_constrained_pool,
    fetch_system_b_pool    as _fetch_system_b_pool,
)
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
        "out_root": "results/sweep_uc1",
    },
    # Hardened pool, distractors as isolated nodes (no distractor edges).
    # Primary result — see thesis methodology.
    "nearest_L100": {
        "export":   "data/hard_pools/neo4j_export_distractors_nearest_L100.json",
        "out_root": "results/sweep_uc1/hard_pools_nearest_L100",
    },
    # Hardened pool, unrestricted distractor<->distractor CONTRADICTS edges.
    # PLACEHOLDER path — confirm the actual filename before running.
    "with_distractor_edges": {
        "export":   "data/hard_pools/neo4j_export_distractors_with_distractor_edges.json",
        "out_root": "results/sweep_uc1/hard_pools_with_distractor_edges",
    },
    # Hardened pool, corrected graph with zero distractor<->distractor edges.
    # PLACEHOLDER path — confirm the actual filename before running.
    "no_dd": {
        "export":   "data/hard_pools/neo4j_export_distractors_no_dd.json",
        "out_root": "results/sweep_uc1/hard_pools_no_dd",
    },
}

NEO4J_EXPORT_PATH = POOL_CONFIGS[POOL_CONFIG]["export"]

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"

POOL_SIZES  = [20, 50, 100, 200]
TOP_KS      = [5, 10, 15, 20, 30]
RANDOM_SEED = 42

OUT_ROOT    = POOL_CONFIGS[POOL_CONFIG]["out_root"]
OUT_CONFIGS = os.path.join(OUT_ROOT, "configs")
OUT_PLOTS   = os.path.join(OUT_ROOT, "plots")
OUT_SUMMARY = os.path.join(OUT_ROOT, "sweep_summary.json")
OUT_CSV     = os.path.join(OUT_ROOT, "sweep_summary.csv")

# Systems written to disk, one retrieval file per (config, system).
SYSTEMS_WITH_FILES = ["System A", "System B", "Baseline A", "Baseline B"]


# ============================================================
# SETUP
# ============================================================
for d in [OUT_ROOT, OUT_CONFIGS, OUT_PLOTS]:
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


def get_driver():
    return load_export(NEO4J_EXPORT_PATH)


# ============================================================
# I/O HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def retrieval_path(pool_size, top_k, system):
    """Path to the per-system retrieval file for a given config.

    eval_uc1.py reads from:
      results/sweep_uc1/configs/pool<P>_k<K>/<system>.json
    """
    safe = system.lower().replace(" ", "_")
    cfg_dir = os.path.join(OUT_CONFIGS, f"pool{pool_size}_k{top_k}")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, f"{safe}.json")


# ============================================================
# MMR + COSINE HELPERS
# ============================================================
def calculate_mmr(texts, vecs, query_embedding=None,
                  relevance_scores=None, top_k=3, lambda_param=0.5):
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
    sims = cosine_similarity(np.array(vecs),
                             query_vec.reshape(1, -1)).flatten()
    top  = np.argsort(sims)[-k:][::-1]
    return [texts[i] for i in top]


def pool_centroid(vecs):
    """Mean embedding of a candidate pool — the query surrogate used by
    Baseline A. Matches System A's own centroid fallback in calculate_mmr
    (see the `else` branch there), so the only difference between System A
    and Baseline A is MMR's diversity term, not the query representation."""
    return np.array(vecs).mean(axis=0, keepdims=True)


# ============================================================
# POOL FETCHERS (delegated to JSON export adapter)
# ============================================================
def fetch_constrained_pool(driver, policy_name, pool_size):
    return _fetch_constrained_pool(driver, policy_name, pool_size=pool_size)


def fetch_system_b_pool(driver, policy_name, pool_size):
    hop_size = max(5, pool_size // 2)
    return _fetch_system_b_pool(driver, policy_name,
                                pool_size=pool_size, hop_size=hop_size)


# ============================================================
# RETRIEVAL — per-policy checkpointed
# ============================================================
def run_retrieval_for_config(driver, pool_size, top_k):
    """Retrieve for all systems at (pool_size, top_k).

    PER-POLICY CHECKPOINTING:
      Each system's retrieval is kept as a list in memory and flushed
      to its JSON file after every policy. On restart the file is
      read back and policies already present are skipped. This means
      an interrupted run resumes from the last completed policy, not
      from the start of the config.

    Returns {system: list_of_policy_records}.
    """
    # Load existing partial results from disk
    results = {}
    done    = {}
    for system in SYSTEMS_WITH_FILES:
        path = retrieval_path(pool_size, top_k, system)
        if os.path.exists(path):
            results[system] = load_json(path)
            done[system]    = {r["policy"] for r in results[system]}
        else:
            results[system] = []
            done[system]    = set()

    remaining = [p for p in POLICIES
                 if any(p not in done[s] for s in SYSTEMS_WITH_FILES)]

    if not remaining:
        print(f"    pool{pool_size}_k{top_k}: all {len(POLICIES)} policies "
              f"already retrieved — skipping")
        return results

    print(f"    pool{pool_size}_k{top_k}: retrieving "
          f"{len(remaining)} remaining policies ...")

    for policy in tqdm(remaining, desc=f"    pool{pool_size}_k{top_k}"):
        pa_t, pa_v, ca_t, ca_v = fetch_constrained_pool(
            driver, policy, pool_size)
        pb_t, pb_v, pb_pr, cb_t, cb_v, cb_pr = fetch_system_b_pool(
            driver, policy, pool_size)

        if policy not in done["System A"]:
            results["System A"].append({
                "policy": policy,
                "retrieved_pros": calculate_mmr(pa_t, pa_v, top_k=top_k),
                "retrieved_cons": calculate_mmr(ca_t, ca_v, top_k=top_k),
            })
            save_json(results["System A"],
                      retrieval_path(pool_size, top_k, "System A"))

        if policy not in done["System B"]:
            results["System B"].append({
                "policy": policy,
                "retrieved_pros": calculate_mmr(
                    pb_t, pb_v, relevance_scores=pb_pr, top_k=top_k),
                "retrieved_cons": calculate_mmr(
                    cb_t, cb_v, relevance_scores=cb_pr, top_k=top_k),
            })
            save_json(results["System B"],
                      retrieval_path(pool_size, top_k, "System B"))

        if policy not in done["Baseline A"]:
            # Query surrogate = per-stance pool centroid, matching System
            # A's own centroid fallback — isolates MMR's diversity term as
            # the only difference between System A and Baseline A.
            results["Baseline A"].append({
                "policy": policy,
                "retrieved_pros": cosine_topk(
                    pa_t, pa_v, pool_centroid(pa_v), top_k),
                "retrieved_cons": cosine_topk(
                    ca_t, ca_v, pool_centroid(ca_v), top_k),
            })
            save_json(results["Baseline A"],
                      retrieval_path(pool_size, top_k, "Baseline A"))

        if policy not in done["Baseline B"]:
            random.seed(f"{RANDOM_SEED}|{policy}")  # str seed: reproducible across processes (hash() is salted)
            results["Baseline B"].append({
                "policy": policy,
                "retrieved_pros": random.sample(pa_t, min(top_k, len(pa_t))),
                "retrieved_cons": random.sample(ca_t, min(top_k, len(ca_t))),
            })
            save_json(results["Baseline B"],
                      retrieval_path(pool_size, top_k, "Baseline B"))

        # Update done sets so subsequent policies in the same run
        # don't try to re-add entries already appended this iteration.
        for s in SYSTEMS_WITH_FILES:
            done[s].add(policy)

    return results


# ============================================================
# METRICS
# ============================================================
def mean_pairwise_cosine(texts):
    if len(texts) < 2:
        return 0.0
    embs = embed_model.encode(texts, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False)
    sim  = embs @ embs.T
    iu   = np.triu_indices(len(embs), k=1)
    return float(np.mean(sim[iu])) if len(iu[0]) > 0 else 0.0


def nli_label_batch(premises, hypotheses, batch_size=32):
    if not premises:
        return []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            bp = premises[i:i+batch_size]
            bh = hypotheses[i:i+batch_size]
            enc = nli_tokenizer(bp, bh, padding=True, truncation=True,
                                max_length=256,
                                return_tensors="pt").to(device)
            logits = nli_model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
    # cross-encoder/nli-deberta-v3-base order = (contra, entail, neutral)
    return [(p[0], p[1], p[2]) for p in all_probs]


def compute_metrics_for_config(systems_results):
    """systems_results: {system: list_of_policy_records}."""
    metrics = {}
    for system, records in systems_results.items():
        red_all, red_pros, red_cons = [], [], []
        cd_density, cd_count = [], []

        for r in records:
            pros = r.get("retrieved_pros", [])
            cons = r.get("retrieved_cons", [])
            red_all.append(mean_pairwise_cosine(pros + cons))
            if pros or cons:
                red_pros.append(mean_pairwise_cosine(pros))
                red_cons.append(mean_pairwise_cosine(cons))
                if pros and cons:
                    premises, hypotheses = [], []
                    for p in pros:
                        for c in cons:
                            premises.append(p)
                            hypotheses.append(c)
                    probs   = nli_label_batch(premises, hypotheses)
                    contras = [pr[0] for pr in probs]
                    cd_density.append(float(np.mean(contras)))
                    cd_count.append(
                        int(sum(1 for c in contras if c > 0.5)))
                else:
                    cd_density.append(0.0)
                    cd_count.append(0)

        metrics[system] = {
            "mean_redundancy_all":   float(np.mean(red_all))   if red_all   else float("nan"),
            "mean_redundancy_pros":  float(np.mean(red_pros))  if red_pros  else float("nan"),
            "mean_redundancy_cons":  float(np.mean(red_cons))  if red_cons  else float("nan"),
            "mean_conflict_density": float(np.mean(cd_density)) if cd_density else float("nan"),
            "mean_conflict_count":   float(np.mean(cd_count))   if cd_count   else float("nan"),
            "n_policies":            len(records),
        }
    return metrics


# ============================================================
# PLOTTING
# ============================================================
def plot_metric_curves(summary, metric_key, title, ylabel,
                       lower_is_better, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors     = {"System A": "tab:blue", "System B": "tab:green",
                  "Baseline A": "tab:orange", "Baseline B": "tab:red"}
    linestyles = ["-", "--", ":", "-."]

    for system in SYSTEMS_WITH_FILES:
        for ls_idx, pool_size in enumerate(POOL_SIZES):
            ks, vals = [], []
            for k in TOP_KS:
                key = f"pool{pool_size}_k{k}"
                if key in summary and system in summary[key]:
                    v = summary[key][system].get(metric_key)
                    if v is not None and not (isinstance(v, float)
                                              and np.isnan(v)):
                        ks.append(k)
                        vals.append(v)
            if vals:
                ax.plot(ks, vals, marker="o",
                        linestyle=linestyles[ls_idx % len(linestyles)],
                        color=colors[system],
                        label=f"{system} (pool={pool_size})", alpha=0.8)

    direction = "down better" if lower_is_better else "up better"
    ax.set_xlabel("top_k (final retrieved per stance)")
    ax.set_ylabel(f"{ylabel}  ({direction})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(OUT_PLOTS, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  saved {out_path}")


# ============================================================
# CSV EXPORT
# ============================================================
def write_csv(summary):
    rows = []
    for config_key, sys_metrics in summary.items():
        parts     = config_key.replace("pool", "").split("_k")
        pool_size = int(parts[0])
        top_k     = int(parts[1])
        for system, m in sys_metrics.items():
            rows.append({
                "pool_size":        pool_size,
                "top_k":            top_k,
                "system":           system,
                "redundancy_all":   m.get("mean_redundancy_all"),
                "redundancy_pros":  m.get("mean_redundancy_pros"),
                "redundancy_cons":  m.get("mean_redundancy_cons"),
                "conflict_density": m.get("mean_conflict_density"),
                "conflict_count":   m.get("mean_conflict_count"),
                "n_policies":       m.get("n_policies"),
            })
    rows.sort(key=lambda r: (r["pool_size"], r["top_k"], r["system"]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"  saved {OUT_CSV}")


# ============================================================
# MAIN
# ============================================================
def main():
    configs = [(p, k) for p, k in product(POOL_SIZES, TOP_KS) if k <= p]
    print("=" * 65)
    print("UC1 K-SENSITIVITY SWEEP (JSON export, per-policy checkpointed)")
    print(f"  Pool sizes : {POOL_SIZES}")
    print(f"  Top-K vals : {TOP_KS}")
    print(f"  Configs    : {len(configs)}")
    print(f"  Policies   : {len(POLICIES)} (shared list)")
    print("=" * 65)

    # Load existing metrics summary (checkpoint at config granularity).
    # Note: a config is only added to summary AFTER all its retrieval
    # files are complete, so a half-finished config will be re-evaluated.
    summary = {}
    if os.path.exists(OUT_SUMMARY):
        with open(OUT_SUMMARY, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"\nResumed: {len(summary)}/{len(configs)} configs done")

    driver = get_driver()
    try:
        for pool_size, top_k in tqdm(configs, desc="Configs"):
            config_key = f"pool{pool_size}_k{top_k}"

            # Check whether retrieval files are complete for this config.
            # Even if config_key is in summary, the retrieval files must
            # exist for eval_uc1.py to use them later.
            files_complete = all(
                os.path.exists(retrieval_path(pool_size, top_k, s))
                and len(load_json(
                    retrieval_path(pool_size, top_k, s))) == len(POLICIES)
                for s in SYSTEMS_WITH_FILES
            )

            if config_key in summary and files_complete:
                continue

            print(f"\n--- Config: pool={pool_size}, k={top_k} ---")

            # Phase 1: retrieval (per-policy checkpointed)
            systems_results = run_retrieval_for_config(driver, pool_size, top_k)

            # Phase 2: deterministic metrics
            print("  Computing metrics...")
            metrics = compute_metrics_for_config(systems_results)
            summary[config_key] = metrics

            # Persist metrics summary
            with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            for sys_name, m in metrics.items():
                print(f"    {sys_name:12s}  "
                      f"red_all={m['mean_redundancy_all']:.3f}  "
                      f"conflict={m['mean_conflict_density']:.3f}")

    finally:
        driver.close()

    # ---- Final exports -------------------------------------------
    write_csv(summary)

    print("\nGenerating plots...")
    plot_metric_curves(
        summary, "mean_redundancy_all",
        "Redundancy vs top_k (within retrieved set)",
        "Mean intra-set cosine similarity",
        lower_is_better=True, filename="redundancy_all.png")
    plot_metric_curves(
        summary, "mean_redundancy_pros",
        "Redundancy among PROs vs top_k",
        "Mean intra-set cosine similarity (pros)",
        lower_is_better=True, filename="redundancy_pros.png")
    plot_metric_curves(
        summary, "mean_redundancy_cons",
        "Redundancy among CONs vs top_k",
        "Mean intra-set cosine similarity (cons)",
        lower_is_better=True, filename="redundancy_cons.png")
    plot_metric_curves(
        summary, "mean_conflict_density",
        "Conflict density vs top_k (NLI contradiction over pro-con pairs)",
        "Mean P(contradiction)",
        lower_is_better=False, filename="conflict_density.png")
    plot_metric_curves(
        summary, "mean_conflict_count",
        "Mean #contradicting pro-con pairs per policy",
        "# pairs with P(contradiction) > 0.5",
        lower_is_better=False, filename="conflict_count.png")

    print(f"\nDone.")
    print(f"  Metrics  : {OUT_SUMMARY}")
    print(f"  Configs  : {OUT_CONFIGS}/pool<P>_k<K>/<system>.json")
    print(f"  Plots    : {OUT_PLOTS}")
    print(f"\nThe pool200_k5 config is what eval_uc1.py needs.")
    print(f"Check it exists: ls {OUT_CONFIGS}/pool200_k5/")


if __name__ == "__main__":
    main()