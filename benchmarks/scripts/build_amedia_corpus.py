"""Build a retrieval corpus + query set from the first 100 Amedia articles and
their generated questions (results/amedia_fable_questions.json), in the same
shape as build_norquad_corpus.py's output — so amedia_retrieval_benchmark.py
can reuse retrieval_benchmark.py's evaluate() unchanged.

Each question has exactly one gold article (a 1-to-1 relationship, unlike
NorQuAD where many questions share a passage), so this corpus is small
(100 passages / 100 queries) by design.

Amedia content is proprietary/paywalled: the produced corpus/queries files
contain full article text and are gitignored, never committed. Only
amedia_retrieval_benchmark.py's output CSV (metrics only) is committed.

Usage:
    python3 build_amedia_corpus.py <source_jsonl>
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_articles_by_uuid(source_path):
    by_uuid = {}
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_uuid[r["uuid"]] = r
    return by_uuid


def main():
    if len(sys.argv) < 2:
        print("usage: build_amedia_corpus.py <source_jsonl>")
        sys.exit(1)
    source_path = sys.argv[1]

    questions = json.loads((RESULTS_DIR / "amedia_fable_questions.json").read_text(encoding="utf-8"))
    articles = load_articles_by_uuid(source_path)

    corpus = {}
    queries = []
    skipped = 0
    for qid, q in enumerate(questions):
        if q.get("error") or not q.get("spørsmål"):
            skipped += 1
            continue
        art = articles.get(q["uuid"])
        if art is None:
            skipped += 1
            continue
        doc_id = q["content_id"]
        text = "\n".join(filter(None, [art.get("title", ""), art.get("lead_text", ""), art.get("body_text", "")]))
        corpus[doc_id] = text
        queries.append({"qid": qid, "question": q["spørsmål"], "gold_doc_id": doc_id})

    with open(DATA_DIR / "amedia_corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "amedia_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"corpus: {len(corpus)} passages, queries: {len(queries)} (skipped {skipped})")


if __name__ == "__main__":
    main()
