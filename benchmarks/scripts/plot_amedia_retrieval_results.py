"""Plot amedia_retrieval_benchmark.py's results — nb-sbert-base vs
multilingual-e5-large on the 100-question Amedia higher-level-question
retrieval benchmark (see AMEDIA_RETRIEVAL.md).

Single-axes bar chart: one bar per model, Recall@1 only — the one metric
where the two models actually diverge (both are already 1.0 by rank 5, see
AMEDIA_RETRIEVAL.md).

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
METRIC = "recall@1"
METRIC_LABEL = "Recall@1"


def main():
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist — run amedia_retrieval_benchmark.py first.")
        sys.exit(1)
    df = pd.read_csv(IN_PATH).set_index("model")
    models = list(df.index)
    print("=== Amedia retrieval results (100 queries / 100 passages) ===")
    print(df[[METRIC]].round(4).to_string())

    vals = df[METRIC].values.astype(float)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 1.5, 5.0), dpi=200)
    x = np.arange(len(models))
    bars = ax.bar(x, vals, width=0.55, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(METRIC_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E1D6", zorder=0)
    ax.set_axisbelow(True)

    n_q = int(df["n_queries"].iloc[0]) if "n_queries" in df.columns else 100
    ax.set_title(f"Pilot Study {METRIC_LABEL} (n={n_q} queries / {n_q} passages)",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nSaved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
