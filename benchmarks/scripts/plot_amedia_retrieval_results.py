"""Plot amedia_retrieval_benchmark.py's results — nb-sbert-base vs
multilingual-e5-large on the 100-question Amedia higher-level-question
retrieval benchmark (see AMEDIA_RETRIEVAL.md).

Single-axes grouped bar chart: one group per metric (Recall@1/5/10, MRR@10,
nDCG@10), one bar per model within each group, so all five metrics are
directly comparable on one shared 0-1 scale.

Usage:
    python3 plot_amedia_retrieval_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IN_PATH = RESULTS_DIR / "amedia_retrieval_results.csv"
PLOT_PATH = RESULTS_DIR / "amedia_retrieval_results.png"

PALETTE = ["#1F6F78", "#96712B", "#6E4C9E", "#3E7A3E", "#B0473E"]
METRIC_LABELS = {
    "recall@1": "Recall@1", "recall@5": "Recall@5", "recall@10": "Recall@10",
    "mrr@10": "MRR@10", "ndcg@10": "nDCG@10",
}


def main():
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist — run amedia_retrieval_benchmark.py first.")
        sys.exit(1)
    df = pd.read_csv(IN_PATH).set_index("model")
    metric_cols = [c for c in METRIC_LABELS if c in df.columns]
    models = list(df.index)
    print("=== Amedia retrieval results (100 queries / 100 passages) ===")
    print(df[metric_cols].round(4).to_string())

    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    n_metrics = len(metric_cols)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(2.0 * n_metrics + 1.5, 5.0), dpi=200)
    for i, m in enumerate(models):
        vals = df.loc[m, metric_cols].values.astype(float)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, color=color_map[m],
                       label=m, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[c] for c in metric_cols], fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E1D6", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=n_models, frameon=False)

    n_q = int(df["n_queries"].iloc[0]) if "n_queries" in df.columns else 100
    ax.set_title(f"Amedia higher-level-question retrieval (n={n_q} queries / {n_q} passages)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nSaved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
