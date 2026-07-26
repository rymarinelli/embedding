"""Build a retrieval corpus + query set from the NorQuAD news annotation files.

Reads the two SQuAD-formatted annotation files (answers_1.json, answers_2.json —
one per human annotator, covering disjoint documents) and produces:
  - corpus.jsonl   : {"doc_id": int, "text": str}          one row per news paragraph
  - queries.jsonl  : {"qid": int, "question": str, "gold_doc_id": int}
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_FILES = ["norquad_news_answers_1.json", "norquad_news_answers_2.json"]


def main():
    corpus = {}
    queries = []

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
                        "qid": qa["id"],
                        "question": qa["question"],
                        "gold_doc_id": doc_id,
                    })

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
