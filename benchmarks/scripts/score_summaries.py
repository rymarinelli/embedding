"""Score every generated-summary file in results/generated_summaries/ against
the NorSumm reference summaries using lexical metrics (ROUGE-1/2/L F1).

Each article has 3 reference summaries; per NorSumm/standard multi-reference
ROUGE practice we take the max score across references for each article,
then average across articles.
"""
import json
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GEN_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_summaries"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METRICS = ["rouge1", "rouge2", "rougeL"]


def score_model(gen_path, references_by_id):
    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=False)
    outputs = json.loads(gen_path.read_text(encoding="utf-8"))

    per_metric_scores = {m: [] for m in METRICS}
    n_empty = 0
    for item in outputs:
        summary = item["summary"].strip()
        refs = references_by_id[item["id"]]
        if not summary:
            n_empty += 1
            for m in METRICS:
                per_metric_scores[m].append(0.0)
            continue
        best = {m: 0.0 for m in METRICS}
        for ref in refs:
            scores = scorer.score(ref, summary)
            for m in METRICS:
                best[m] = max(best[m], scores[m].fmeasure)
        for m in METRICS:
            per_metric_scores[m].append(best[m])

    row = {"model": gen_path.stem, "n_articles": len(outputs), "n_empty_outputs": n_empty}
    for m in METRICS:
        row[f"{m}_f1"] = round(sum(per_metric_scores[m]) / len(per_metric_scores[m]), 4)
    return row


def main():
    articles = json.loads((DATA_DIR / "norsumm_test.json").read_text(encoding="utf-8"))
    references_by_id = {a["id"]: a["reference_summaries"] for a in articles}

    gen_files = sorted(GEN_DIR.glob("*.json"))
    if not gen_files:
        print(f"No generated summaries found in {GEN_DIR} — run generate_summaries_openrouter.py first")
        return

    rows = [score_model(p, references_by_id) for p in gen_files]
    df = pd.DataFrame(rows).sort_values("rougeL_f1", ascending=False)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "norsumm_lexical_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
