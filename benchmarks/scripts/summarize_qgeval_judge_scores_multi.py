"""Summarize qgeval_judge_scores_multi.json into a per-system, per-dimension
mean-score table.

Reads the output of score_qgeval_dimensions_multi.py and reports each
system's mean score (1-3 scale) on every QGEval dimension plus an overall
average across dimensions, sorted best-to-worst.

Usage:
    python3 summarize_qgeval_judge_scores_multi.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qgeval_dimensions import DIMENSIONS, DISPLAY_NAMES  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IN_PATH = RESULTS_DIR / "qgeval_judge_scores_multi.json"
OUT_PATH = RESULTS_DIR / "qgeval_judge_scores_multi_summary.csv"


def main():
    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        if r.get("error"):
            continue
        for sys_, dims in r["scores"].items():
            row = {"qid": r["qid"], "system": sys_}
            row.update(dims)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["overall"] = df[DIMENSIONS].mean(axis=1)
    summary = df.groupby("system")[DIMENSIONS + ["overall"]].mean()
    summary = summary.rename(columns=DISPLAY_NAMES)
    summary = summary.sort_values("overall", ascending=False)

    pd.set_option("display.width", 160)
    print(summary.round(3).to_string())

    summary.to_csv(OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")

    print("\nn passages per system:")
    print(df.groupby("system").size().to_string())


if __name__ == "__main__":
    main()
