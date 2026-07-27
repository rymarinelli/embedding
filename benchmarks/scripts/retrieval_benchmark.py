"""Retrieval benchmark on the NorQuAD news corpus/queries for several embedding models.

For each model: encode the corpus and the questions, rank passages by cosine
similarity, and score against the single gold passage per question with
Recall@1, Recall@5, Recall@10, MRR@10 and nDCG@10.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class MeanPoolingEncoder:
    """.encode(...)-compatible wrapper for plain transformers encoders (like
    ltg/norbert4-*) that aren't packaged as sentence-transformers models —
    mean-pools the last hidden state over non-padding tokens, the standard
    way to turn a raw BERT-family encoder into a sentence embedding.
    """

    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def encode(self, texts, batch_size=64, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True):
        all_emb = []
        rng = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm.auto import tqdm
            rng = tqdm(rng)
        for i in rng:
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt")
            with torch.inference_mode():
                out = self.model(**enc)
            hidden = out.last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts
            if normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.numpy() if convert_to_numpy else emb)
        return np.concatenate(all_emb, axis=0) if convert_to_numpy else all_emb


# (model_id, query_prefix, passage_prefix, batch_size, loader) — E5-family
# and Qwen3-Embedding require instruction/prefix strings to work as
# documented; pplx-embed-v1 deliberately requires none (see its model
# card). Both Qwen3-based models (pplx-embed-v1 is a Qwen3 continued-
# pretrain; Qwen3-Embedding is Qwen3 natively) scale badly with batch_size
# on CPU at realistic passage length (~500 tokens): batch_size=64 measured
# 15-30x slower per-item than batch_size=8, apparently from attention/
# padding cost on this architecture without a CPU-optimized kernel.
# Non-Qwen3 models are unaffected and keep batch_size=64. loader "st" uses
# SentenceTransformer; "meanpool" uses MeanPoolingEncoder for raw encoders
# (ltg/norbert4-*) that aren't packaged as sentence-transformers models.
MODELS = [
    ("NbAiLab/nb-sbert-base", "", "", 64, "st"),
    ("NbAiLab/nb-sentence-bert-base-mnli-test", "", "", 64, "st"),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "", 64, "st"),
    ("intfloat/multilingual-e5-small", "query: ", "passage: ", 64, "st"),
    ("intfloat/multilingual-e5-base", "query: ", "passage: ", 64, "st"),
    ("intfloat/multilingual-e5-large", "query: ", "passage: ", 64, "st"),
    ("BAAI/bge-m3", "", "", 64, "st"),
    ("perplexity-ai/pplx-embed-v1-0.6b", "", "", 8, "st"),
    (
        "Qwen/Qwen3-Embedding-0.6B",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
        8,
        "st",
    ),
    ("ltg/norbert4-base", "", "", 32, "meanpool"),
    ("ltg/norbert4-large", "", "", 16, "meanpool"),
]

TOP_K = 10


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def dcg_at_k(rank, k):
    # binary relevance, single relevant doc at position `rank` (1-indexed, or None)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def evaluate(model_id, query_prefix, passage_prefix, corpus, queries, batch_size=64, loader="st"):
    print(f"\n=== {model_id} (batch_size={batch_size}, loader={loader}) ===")
    t0 = time.time()
    if loader == "meanpool":
        model = MeanPoolingEncoder(model_id)
    else:
        model = SentenceTransformer(model_id, trust_remote_code=True)

    doc_ids = [c["doc_id"] for c in corpus]
    passages = [passage_prefix + c["text"] for c in corpus]
    questions = [query_prefix + q["question"] for q in queries]

    passage_emb = model.encode(
        passages, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    query_emb = model.encode(
        questions, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    sims = query_emb @ passage_emb.T  # [n_queries, n_passages]
    doc_id_to_col = {d: i for i, d in enumerate(doc_ids)}

    ranks = []
    for i, q in enumerate(queries):
        gold_col = doc_id_to_col[q["gold_doc_id"]]
        order = np.argsort(-sims[i])
        rank = int(np.where(order == gold_col)[0][0]) + 1  # 1-indexed
        ranks.append(rank)
    ranks = np.array(ranks)

    n = len(ranks)
    recall_at = lambda k: float(np.mean(ranks <= k))
    mrr_at_10 = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    ndcg_at_10 = float(np.mean([dcg_at_k(r, 10) for r in ranks]))  # ideal DCG = 1.0 (single relevant doc)

    elapsed = time.time() - t0
    result = {
        "model": model_id,
        "n_queries": n,
        "n_passages": len(corpus),
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "mrr@10": mrr_at_10,
        "ndcg@10": ndcg_at_10,
        "seconds": round(elapsed, 1),
    }
    for k, v in result.items():
        print(f"  {k}: {v}")

    del model
    return result


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    only = set(sys.argv[1:]) or None  # run everything if no args given
    models_to_run = [m for m in MODELS if only is None or m[0] in only]

    out_path = RESULTS_DIR / "norquad_retrieval_results.csv"
    existing = {}
    if out_path.exists():
        for row in pd.read_csv(out_path).to_dict("records"):
            existing[row["model"]] = row

    RESULTS_DIR.mkdir(exist_ok=True)
    for model_id, qp, pp, bs, loader in models_to_run:
        try:
            row = evaluate(model_id, qp, pp, corpus, queries, batch_size=bs, loader=loader)
        except Exception as e:
            print(f"  FAILED: {model_id}: {e}")
            row = {"model": model_id, "error": str(e)}
        existing[model_id] = row
        # Save after every model — some of these take an hour+ on CPU, and an
        # interrupted run shouldn't lose already-completed results.
        pd.DataFrame(list(existing.values())).to_csv(out_path, index=False)

    df = pd.DataFrame(list(existing.values()))
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
