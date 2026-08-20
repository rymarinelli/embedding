"""Sample (passage, answer, reference_question) triples from NorQuAD for the
QGEval-style question-generation evaluation.

Mirrors QGEval (Fu et al., EMNLP 2024), which sampled 200 passages from the
test splits of SQuAD/HotpotQA and used each passage's (passage, answer) pair
as the QG input, with the dataset's own question as the reference.

NorQuAD is SQuAD-formatted Norwegian news QA, so it plays the SQuAD role here.
One question is sampled per passage so no single passage dominates the sample.

Usage:
    python3 prepare_qgeval_sample.py [n_samples]   # default 200
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "qgeval_sample.json"
SEED = 42


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    corpus = {c["doc_id"]: c["text"] for c in load_jsonl(DATA_DIR / "corpus.jsonl")}
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    by_doc = defaultdict(list)
    for q in queries:
        by_doc[q["gold_doc_id"]].append(q)

    rng = random.Random(SEED)
    doc_ids = sorted(by_doc)
    rng.shuffle(doc_ids)
    doc_ids = doc_ids[:n]

    samples = []
    for doc_id in doc_ids:
        q = rng.choice(sorted(by_doc[doc_id], key=lambda r: r["qid"]))
        samples.append({
            "qid": q["qid"],
            "doc_id": doc_id,
            "passage": corpus[doc_id],
            "answer": q["gold_answer"],
            "reference_question": q["question"],
        })

    OUT_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sampled {len(samples)} (passage, answer, reference_question) triples -> {OUT_PATH}")
    print(f"  distinct passages: {len({s['doc_id'] for s in samples})}")


if __name__ == "__main__":
    main()
