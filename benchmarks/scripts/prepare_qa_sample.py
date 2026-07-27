"""Build a reproducible random sample of NorQuAD news questions (with their
gold passage + answer) for the QA benchmark — used for models that can't do
embedding-based retrieval (e.g. Anthropic's API has no embeddings endpoint).
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_SIZE = 300
SEED = 42


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    corpus = {c["doc_id"]: c["text"] for c in load_jsonl(DATA_DIR / "corpus.jsonl")}
    queries = load_jsonl(DATA_DIR / "queries.jsonl")  # each row already carries its own gold_answer

    random.seed(SEED)
    sample = random.sample(queries, SAMPLE_SIZE)

    records = []
    for q in sample:
        records.append({
            "qid": q["qid"],
            "question": q["question"],
            "context": corpus[q["gold_doc_id"]],
            "gold_answer": q["gold_answer"],
        })

    out_path = DATA_DIR / "norquad_qa_sample.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} QA examples to {out_path}")


if __name__ == "__main__":
    main()
