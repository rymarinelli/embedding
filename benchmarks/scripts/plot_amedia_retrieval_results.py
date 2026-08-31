"""Plot amedia_retrieval_benchmark.py's results — nb-sbert-base vs
multilingual-e5-large on the 100-question Amedia higher-level-question
retrieval benchmark (see AMEDIA_RETRIEVAL.md).

Same visual style as report_qgeval_multi.py / plot_qgeval_lexical_metrics.py:
one panel per metric, horizontal bars, values printed at the bar end.

Usage:
    python3 plot_amedia_retrieval_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IN_PATH = RESULTS_DIR / "amedia_retrieval_results.csv"
PLOT_PATH = RESULTS_DIR / "amedia_retrieval_results.png"

PALETTE = ["#1F6F78", "#96712B", "#6E4C9E", "#3E7A3E", "#B0473E"]


def main():
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist — run amedia_retrieval_benchmark.py first.")
        sys.exit(1)
    df = pd.read_csv(IN_PATH).set_index("model")
    metric_cols = [c for c in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
                   if c in df.columns]
    models = list(df.index)
    print("=== Amedia retrieval results (100 queries / 100 passages) ===")
    print(df[metric_cols].round(4).to_string())

    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    ncols = 3
    nrows = -(-len(metric_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), dpi=200)
    axes = axes.reshape(-1)

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        sub = df[metric].sort_values(ascending=True)
        colors = [color_map[m] for m in sub.index]
        ax.barh(sub.index, sub.values, color=colors, zorder=3)
        for y, v in enumerate(sub.values):
            ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color="#E5E1D6", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8.5)

    for j in range(len(metric_cols), len(axes)):
        axes[j].axis("off")

    n_q = int(df["n_queries"].iloc[0]) if "n_queries" in df.columns else 100
    fig.suptitle(f"Amedia higher-level-question retrieval (n={n_q} queries / {n_q} passages)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nSaved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
