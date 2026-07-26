"""
common.py
=========
Shared utilities for the thesis evaluation pipeline.

This module absorbs code that was previously copy-pasted across
generate_ground_truth_v2.py, generate_uc2_reference_summaries.py,
stage2_eval_uc1.py, stage2_eval_uc2.py and stage2_eval_uc2_paragraph.py:

  - JSON load/save with checkpoint support
  - Lazy-loaded embedding model (sentence-transformers)
  - Lazy-loaded NLI model (cross-encoder)
  - Robust JSON parsing (markdown fences, <think> tags, truncation)
  - A/B parsing for pairwise judges
  - The Ollama client
  - NEW: semantic_match() — the key fix for the exact-string-match bug

WHY semantic_match() MATTERS
----------------------------
The old eval used `text.strip().lower() in relevant_set` — exact
string match. The argument corpus is full of near-duplicate
paraphrases ("...save lives and make organs more easily obtainable"
vs "...boost the availability of transplant organs and save lives").
Exact match scores a retrieved paraphrase of a ground-truth argument
as a complete miss. That is the bug behind Baseline A == 0.0 and
behind the artificially-low Jaccard agreement.

semantic_match() compares two texts by embedding cosine similarity
and counts them as "the same argument" above a threshold. Every
relevance computation in the new pipeline uses this instead of `in`.
"""

import os
import re
import json
import numpy as np


# ============================================================
# CONFIG — shared constants
# ============================================================
OLLAMA_HOST      = "http://127.0.0.1:11434"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL_NAME   = "cross-encoder/nli-deberta-v3-base"

# Two texts count as "the same argument" if their embedding cosine
# similarity is >= this threshold. 0.75 is conservative: paraphrases
# of the same claim typically score 0.80-0.95 with all-mpnet-base-v2,
# while genuinely different arguments score well below 0.75.
# Sensitivity to this value should be reported in the thesis appendix.
SEMANTIC_MATCH_THRESHOLD = 0.75

ENTAIL_THRESH = 0.5  # NLI entailment probability cut-off for coverage


# ============================================================
# OLLAMA CLIENT
# ============================================================
def get_ollama():
    """Return an Ollama client. Imported lazily so non-LLM scripts
    (or environments without ollama installed) still load this module."""
    from ollama import Client as OllamaClient
    return OllamaClient(host=OLLAMA_HOST)


# ============================================================
# JSON I/O
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint(path):
    """Load a checkpoint file, returning {} if missing or corrupt."""
    if os.path.exists(path):
        try:
            return load_json(path)
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================
# EMBEDDING MODEL (lazy singleton)
# ============================================================
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model ({EMBED_MODEL_NAME})...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_texts(texts):
    """Embed a list of texts, L2-normalised, as a numpy array."""
    if not texts:
        return np.zeros((0, 768))
    model = get_embed_model()
    return model.encode(texts, convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False)


# ============================================================
# NLI MODEL (lazy singleton)
# ============================================================
_nli_tokenizer = None
_nli_model     = None
_nli_device    = None


def get_nli():
    global _nli_tokenizer, _nli_model, _nli_device
    if _nli_model is None:
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        _nli_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading NLI model ({NLI_MODEL_NAME}) on {_nli_device}...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME).to(_nli_device).eval()
    return _nli_tokenizer, _nli_model, _nli_device


def nli_label_batch(premises, hypotheses, batch_size=32):
    """Return list of (P_contra, P_entail, P_neutral) per (premise, hypothesis).

    Note the output order is (contra, entail, neutral) — re-ordered from
    the model's native (contra, entail, neutral) head layout of
    cross-encoder/nli-deberta-v3-base, which is (contra, entail, neutral).
    """
    if not premises:
        return []
    import torch
    tokenizer, model, device = get_nli()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            bp = premises[i:i+batch_size]
            bh = hypotheses[i:i+batch_size]
            enc = tokenizer(bp, bh, padding=True, truncation=True,
                            max_length=256, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
    # native head order = (contra, entail, neutral)
    return [(p[0], p[1], p[2]) for p in all_probs]


# ============================================================
# SEMANTIC MATCHING — the core fix
# ============================================================
def semantic_match_matrix(retrieved, reference):
    """Cosine similarity matrix between two lists of texts.

    Returns an array of shape (len(retrieved), len(reference)).
    Used by all relevance metrics so that a retrieved paraphrase of
    a ground-truth argument is correctly credited.
    """
    if not retrieved or not reference:
        return np.zeros((len(retrieved), len(reference)))
    r_emb = embed_texts(retrieved)
    g_emb = embed_texts(reference)
    return r_emb @ g_emb.T


def is_relevant_semantic(text, reference_texts, reference_embs=None,
                         threshold=SEMANTIC_MATCH_THRESHOLD):
    """True if `text` semantically matches ANY argument in reference_texts.

    Replaces the old `text.strip().lower() in relevant_set` exact match.
    Pass precomputed reference_embs to avoid re-embedding in a loop.
    """
    if not reference_texts:
        return False
    t_emb = embed_texts([text])[0]
    if reference_embs is None:
        reference_embs = embed_texts(reference_texts)
    sims = reference_embs @ t_emb
    return bool(np.max(sims) >= threshold)


def best_match_index(text, reference_embs, threshold=SEMANTIC_MATCH_THRESHOLD):
    """Return the index of the best-matching reference argument, or -1 if
    no reference is within threshold. Used for graded-relevance lookup."""
    if reference_embs is None or len(reference_embs) == 0:
        return -1
    t_emb = embed_texts([text])[0]
    sims = reference_embs @ t_emb
    j = int(np.argmax(sims))
    return j if sims[j] >= threshold else -1


# ============================================================
# ROBUST JSON / A-B PARSING (shared by every LLM-calling script)
# ============================================================
def strip_model_noise(raw):
    """Remove <think> tags and markdown code fences from an LLM response."""
    if not raw:
        return ""
    raw = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw)
    return raw.strip()


def parse_json_object(raw, required_keys=None, list_recovery=False):
    """Extract the first JSON object from a (possibly noisy) LLM response.

    Args:
        raw            : the raw LLM string
        required_keys  : if given, the parsed object must contain these keys
        list_recovery  : if True, attempt to repair JSON truncated mid-list
                         by appending closing brackets

    Returns the parsed dict, or None on failure.
    """
    raw = strip_model_noise(raw)
    if not raw:
        return None

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None

    payload = m.group()
    parsed = None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        if list_recovery:
            for suffix in ["]}", "]\n}", "  ]\n}", "}]}", "\"}]}"]:
                try:
                    parsed = json.loads(payload + suffix)
                    break
                except json.JSONDecodeError:
                    continue
        if parsed is None:
            return None

    if required_keys and not set(required_keys).issubset(parsed.keys()):
        return None
    return parsed


def parse_ab(raw):
    """Parse a pairwise judge response into 'A', 'B', or None."""
    raw = strip_model_noise(raw)
    if not raw:
        return None
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        u = line.upper()
        if u.startswith("A") and (len(u) == 1 or not u[1].isalpha() or u[1] == "."):
            return "A"
        if u.startswith("B") and (len(u) == 1 or not u[1].isalpha() or u[1] == "."):
            return "B"
    return None


# ============================================================
# AGGREGATION HELPERS
# ============================================================
def mean_std(values):
    """Return (mean, std) of a list, or (None, None) if empty."""
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def fmt(v, decimals=3):
    """Format a number for tables; '-' if None/NaN."""
    if v is None:
        return "-"
    if isinstance(v, float) and np.isnan(v):
        return "-"
    return f"{v:.{decimals}f}"


# ============================================================
# SimpleRAG — vanilla RAG baseline (shared by UC1 and UC2)
# ============================================================
# SimpleRAG is the "no graph" baseline: it embeds the policy name as a
# query and retrieves the top-k most similar arguments by cosine
# similarity from the FULL corpus, separately for PRO and CON stances.
#
# It is STANCE-AWARE on purpose: PRO and CON are retrieved independently
# so SimpleRAG produces retrieved_pros / retrieved_cons just like every
# graph-based system. That makes it directly comparable on every
# per-stance metric (nDCG, Precision, ILD, conflict density, pairwise).
#
# This is the same baseline Edge et al. GraphRAG (2024) use as their
# naive-RAG comparison. Previously this logic was duplicated in
# stage2_eval_uc2_paragraph.py; it now lives here so UC1 and UC2 use
# the IDENTICAL SimpleRAG implementation.

_corpus_cache = {}


def load_corpus(export_path):
    """Load the full argument corpus from a Neo4j JSON export.

    Returns {policy_name: {"pros": [...], "cons": [...]}} where each
    entry is {"text": str, "vec": list[float]}.

    Cached per path so repeated calls are free.
    """
    if export_path in _corpus_cache:
        return _corpus_cache[export_path]

    print(f"  Loading full corpus from {export_path}...")
    export    = load_json(export_path)
    arguments = export.get("arguments", {})
    policies  = export.get("policies", {})

    corpus = {}
    for name, data in policies.items():
        pros, cons = [], []
        for aid in data.get("pros", []):
            a = arguments.get(aid, {})
            if a.get("embedding") and a.get("text"):
                pros.append({"text": a["text"], "vec": a["embedding"]})
        for aid in data.get("cons", []):
            a = arguments.get(aid, {})
            if a.get("embedding") and a.get("text"):
                cons.append({"text": a["text"], "vec": a["embedding"]})
        corpus[name] = {"pros": pros, "cons": cons}

    total = sum(len(v["pros"]) + len(v["cons"]) for v in corpus.values())
    print(f"  Corpus: {len(corpus)} policies, {total} arguments")
    _corpus_cache[export_path] = corpus
    return corpus


def simple_rag_retrieve(policy_name, corpus, k):
    """Vanilla RAG retrieval: top-k PRO and top-k CON by cosine
    similarity of the policy query against the full corpus.

    No graph, no MMR, no PageRank. Stance-aware. Returns
    (retrieved_pros, retrieved_cons).
    """
    if policy_name not in corpus:
        return [], []

    q = embed_texts([policy_name])[0]  # already L2-normalised

    def topk(items):
        if not items:
            return []
        vecs = np.array([it["vec"] for it in items], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (vecs / norms) @ q
        top  = np.argsort(sims)[-k:][::-1]
        return [items[i]["text"] for i in top]

    return topk(corpus[policy_name]["pros"]), topk(corpus[policy_name]["cons"])