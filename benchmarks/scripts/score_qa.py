"""Score QA answer files in results/qa_answers/ with the standard SQuAD-style
Exact Match and token-level F1 metrics against the NorQuAD gold answers.
"""
import json
import re
import string
from pathlib import Path

import pandas as pd

QA_DIR = Path(__file__).resolve().parent.parent / "results" / "qa_answers"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_model(path):
    items = json.loads(path.read_text(encoding="utf-8"))
    ems = [exact_match(it["predicted_answer"], it["gold_answer"]) for it in items]
    f1s = [f1_score(it["predicted_answer"], it["gold_answer"]) for it in items]
    n_empty = sum(1 for it in items if not it["predicted_answer"].strip())
    return {
        "model": path.stem,
        "n_questions": len(items),
        "n_empty_predictions": n_empty,
        "exact_match": round(sum(ems) / len(ems), 4),
        "f1": round(sum(f1s) / len(f1s), 4),
    }


def main():
    files = sorted(QA_DIR.glob("*.json"))
    if not files:
        print(f"No QA answer files found in {QA_DIR}")
        return
    rows = [score_model(p) for p in files]
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)

    out_path = RESULTS_DIR / "norquad_qa_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
