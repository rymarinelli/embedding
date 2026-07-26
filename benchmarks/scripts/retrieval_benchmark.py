"""Retrieval benchmark on the NorQuAD news corpus/queries for several embedding models.

For each model: encode the corpus and the questions, rank passages by cosine
similarity, and score against the single gold passage per question with
Recall@1, Recall@5, Recall@10, MRR@10 and nDCG@10.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# (model_id, query_prefix, passage_prefix) — E5-family models require the
# "query: " / "passage: " instruction prefixes to work as documented.
MODELS = [
    ("NbAiLab/nb-sbert-base", "", ""),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    ("intfloat/multilingual-e5-large", "query: ", "passage: "),
    ("BAAI/bge-m3", "", ""),
]

TOP_K = 10


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def dcg_at_k(rank, k):
    # binary relevance, single relevant doc at position `rank` (1-indexed, or None)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def evaluate(model_id, query_prefix, passage_prefix, corpus, queries):
    print(f"\n=== {model_id} ===")
    t0 = time.time()
    model = SentenceTransformer(model_id, trust_remote_code=True)

    doc_ids = [c["doc_id"] for c in corpus]
    passages = [passage_prefix + c["text"] for c in corpus]
    questions = [query_prefix + q["question"] for q in queries]

    passage_emb = model.encode(
        passages, batch_size=64, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    query_emb = model.encode(
        questions, batch_size=64, normalize_embeddings=True,
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

    rows = []
    for model_id, qp, pp in MODELS:
        try:
            rows.append(evaluate(model_id, qp, pp, corpus, queries))
        except Exception as e:
            print(f"  FAILED: {model_id}: {e}")
            rows.append({"model": model_id, "error": str(e)})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "norquad_retrieval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
