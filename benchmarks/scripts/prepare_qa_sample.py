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
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    # queries.jsonl only has {qid, question, gold_doc_id}; re-read the raw
    # NorQuAD files to attach the gold answer text for each question.
    answers_by_qid = {}
    for fname in ["norquad_news_answers_1.json", "norquad_news_answers_2.json"]:
        data = json.loads((DATA_DIR / fname).read_text(encoding="utf-8"))["data"]
        for article in data:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    answers_by_qid[qa["id"]] = qa["answers"][0]["text"]

    random.seed(SEED)
    sample = random.sample(queries, SAMPLE_SIZE)

    records = []
    for q in sample:
        records.append({
            "qid": q["qid"],
            "question": q["question"],
            "context": corpus[q["gold_doc_id"]],
            "gold_answer": answers_by_qid[q["qid"]],
        })

    out_path = DATA_DIR / "norquad_qa_sample.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} QA examples to {out_path}")


if __name__ == "__main__":
    main()
