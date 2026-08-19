"""Retrieval benchmark on the Amedia question/article set built by
build_amedia_corpus.py — same methodology (Recall@1/5/10, MRR@10, nDCG@10) and
same evaluate() function as retrieval_benchmark.py (NorQuAD), just pointed at
a different, smaller (100 passages / 100 queries, 1-to-1) corpus.

Each question was generated to be a higher-level comprehension question with
a unique gold article, so this specifically tests whether a model's
embedding captures thematic/causal content well enough to retrieve the right
article — not just keyword overlap, which is often weaker for "why"/"how"
questions than for factoid questions like NorQuAD's.

Amedia article text stays local (gitignored data/amedia_*.jsonl); this
script writes ONLY the metrics CSV, no article content, to
results/amedia_retrieval_results.csv.

Usage:
    python3 amedia_retrieval_benchmark.py [model_id ...]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_benchmark import MODELS, evaluate, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    corpus_path = DATA_DIR / "amedia_corpus.jsonl"
    queries_path = DATA_DIR / "amedia_queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        print("Missing amedia_corpus.jsonl / amedia_queries.jsonl — run build_amedia_corpus.py first.")
        sys.exit(1)

    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    print(f"Amedia retrieval: {len(corpus)} passages, {len(queries)} queries")

    only = set(sys.argv[1:]) or None
    models_to_run = [m for m in MODELS if only is None or m[0] in only]

    out_path = RESULTS_DIR / "amedia_retrieval_results.csv"
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
        pd.DataFrame(list(existing.values())).to_csv(out_path, index=False)

    df = pd.DataFrame(list(existing.values()))
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
