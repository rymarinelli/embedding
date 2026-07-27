"""Two-stage retrieve-then-rerank evaluation on the same NorQuAD news
corpus/queries as retrieval_benchmark.py, using the exact same metrics
(Recall@1/5/10, MRR@10, nDCG@10) so results land in the same
norquad_retrieval_results.csv for direct comparison against the pure
bi-encoder rows.

Stage 1 (retrieval): encode corpus + queries with a bi-encoder, take the
top-N candidates per query by cosine similarity (N > 10, so recall@10
after reranking can actually differ from stage-1 recall@10 — with N=10 a
rerank can only reorder within the already-final top-10 and recall@10
would be mathematically identical to stage 1).

Stage 2 (rerank): score each (question, candidate) pair with a
CrossEncoder and re-sort. A gold passage outside the stage-1 pool is
scored as a miss beyond N (a rerank stage can only reorder what stage 1
retrieved, never recover a stage-1 miss) — this is standard practice for
retrieve-then-rerank pipelines, not a scoring bug.

Usage:
    python3 retrieval_rerank_benchmark.py <reranker_model_or_path> <result_label> \
        [bi_encoder_id] [pool_size] [batch_size]
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_BI_ENCODER = "intfloat/multilingual-e5-large"  # best-scoring bi-encoder from retrieval_benchmark.py
DEFAULT_POOL_SIZE = 20
DEFAULT_BATCH_SIZE = 64


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def dcg_at_k(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def stage1_retrieve(bi_encoder_id, corpus, queries, pool_size):
    print(f"Stage 1: encoding with {bi_encoder_id}...")
    model = SentenceTransformer(bi_encoder_id, trust_remote_code=True)
    doc_ids = [c["doc_id"] for c in corpus]
    passages = ["passage: " + c["text"] for c in corpus]
    questions = ["query: " + q["question"] for q in queries]

    passage_emb = model.encode(passages, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    query_emb = model.encode(questions, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    del model

    sims = query_emb @ passage_emb.T
    top_idx = np.argsort(-sims, axis=1)[:, :pool_size]  # [n_queries, pool_size]
    return doc_ids, top_idx


def stage2_rerank(reranker_id, corpus, queries, doc_ids, top_idx, batch_size):
    print(f"Stage 2: reranking with {reranker_id}...")
    reranker = CrossEncoder(reranker_id)

    # Flatten all (question, candidate) pairs into one big batched predict()
    # call — much faster than scoring per-query pool separately.
    flat_pairs = []
    flat_owner = []  # which query index each pair belongs to
    for qi, cand_cols in enumerate(top_idx):
        question = queries[qi]["question"]
        for col in cand_cols:
            flat_pairs.append([question, corpus[col]["text"]])
            flat_owner.append(qi)

    t0 = time.time()
    flat_scores = reranker.predict(flat_pairs, batch_size=batch_size, show_progress_bar=True)
    print(f"  reranked {len(flat_pairs)} pairs in {time.time()-t0:.1f}s")

    n_queries, pool_size = top_idx.shape
    scores_by_query = np.array(flat_scores, dtype=np.float32).reshape(n_queries, pool_size)
    return scores_by_query


def main():
    if len(sys.argv) < 3:
        print("usage: retrieval_rerank_benchmark.py <reranker_model_or_path> <result_label> "
              "[bi_encoder_id] [pool_size] [batch_size]")
        sys.exit(1)
    reranker_id = sys.argv[1]
    label = sys.argv[2]
    bi_encoder_id = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_BI_ENCODER
    pool_size = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_POOL_SIZE
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_BATCH_SIZE

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    t0 = time.time()
    doc_ids, top_idx = stage1_retrieve(bi_encoder_id, corpus, queries, pool_size)
    scores_by_query = stage2_rerank(reranker_id, corpus, queries, doc_ids, top_idx, batch_size)

    doc_id_to_col = {d: i for i, d in enumerate(doc_ids)}
    ranks = []
    for qi, q in enumerate(queries):
        gold_col = doc_id_to_col[q["gold_doc_id"]]
        cand_cols = top_idx[qi]
        pos_in_pool = np.where(cand_cols == gold_col)[0]
        if len(pos_in_pool) == 0:
            ranks.append(pool_size + 1)  # stage-1 miss, beyond the rerank pool
            continue
        # re-sort this query's pool by reranker score, find gold's new rank
        order = np.argsort(-scores_by_query[qi])
        rank_within_pool = int(np.where(order == pos_in_pool[0])[0][0]) + 1
        ranks.append(rank_within_pool)
    ranks = np.array(ranks)

    n = len(ranks)
    recall_at = lambda k: float(np.mean(ranks <= k))
    mrr_at_10 = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    ndcg_at_10 = float(np.mean([dcg_at_k(r, 10) for r in ranks]))
    elapsed = time.time() - t0

    result = {
        "model": label,
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

    out_path = RESULTS_DIR / "norquad_retrieval_results.csv"
    existing = {}
    if out_path.exists():
        for row in pd.read_csv(out_path).to_dict("records"):
            existing[row["model"]] = row
    existing[label] = result
    pd.DataFrame(list(existing.values())).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
