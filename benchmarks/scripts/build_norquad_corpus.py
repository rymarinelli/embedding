"""Build a retrieval corpus + query set from the NorQuAD news annotation files.

Reads the two SQuAD-formatted annotation files (answers_1.json, answers_2.json —
one per human annotator, covering disjoint documents) and produces:
  - corpus.jsonl   : {"doc_id": int, "text": str}          one row per news paragraph
  - queries.jsonl  : {"qid": int, "question": str, "gold_doc_id": int, "gold_answer": str}

NorQuAD's own qa["id"] is only unique *within* each annotator file — combining
both files produces hundreds of id collisions between two completely
unrelated questions (verified: e.g. id 2952 is "Hvem er Robert Næss?" in
answers_1.json and "Hvor skal F-35 fly?" in answers_2.json). qid here is
therefore reassigned as a fresh sequential counter across the combined set,
and gold_answer is embedded directly on each row instead of requiring a
separate id-keyed lookup — the original lookup-by-id approach in
prepare_qa_sample.py silently attached the wrong answer to ~1/3 of questions
whose original id collided.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_FILES = ["norquad_news_answers_1.json", "norquad_news_answers_2.json"]


def main():
    corpus = {}
    queries = []
    next_qid = 0

    for fname in SOURCE_FILES:
        data = json.loads((DATA_DIR / fname).read_text(encoding="utf-8"))["data"]
        for article in data:
            for paragraph in article["paragraphs"]:
                doc_id = paragraph["document_id"]
                text = paragraph["context"]
                if doc_id in corpus and corpus[doc_id] != text:
                    raise ValueError(f"document_id {doc_id} has conflicting contexts across files")
                corpus[doc_id] = text

                for qa in paragraph["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    queries.append({
                        "qid": next_qid,
                        "question": qa["question"],
                        "gold_doc_id": doc_id,
                        "gold_answer": qa["answers"][0]["text"],
                    })
                    next_qid += 1

    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, text in sorted(corpus.items()):
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")

    with open(DATA_DIR / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"corpus: {len(corpus)} passages")
    print(f"queries: {len(queries)} answerable questions")


if __name__ == "__main__":
    main()
