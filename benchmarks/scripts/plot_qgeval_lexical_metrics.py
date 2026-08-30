"""Plot reference-based lexical metrics (BLEU-4, ROUGE-L, METEOR, BERTScore)
per system, from compare_qgeval_borealis_vs_fable.py's per-sample output.

Same visual style as report_qgeval_multi.py's judge-dimension plot, so the
two are easy to read side by side: one panel per metric, horizontal bars
sorted within each panel, values printed at the bar end.

Usage:
    python3 plot_qgeval_lexical_metrics.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IN_PATH = RESULTS_DIR / "qgeval_borealis_vs_fable_per_sample.csv"
PLOT_PATH = RESULTS_DIR / "qgeval_lexical_metrics.png"

DISPLAY_LABEL = {
    "claude-fable-5": "claude-fable-5",
    "borealis-27b": "borealis-27b",
    "borealis-27b-instruct-preview": "borealis-27b-instruct-preview",
}
PALETTE = ["#1F6F78", "#96712B", "#6E4C9E", "#3E7A3E", "#B0473E"]


def main():
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist — run compare_qgeval_borealis_vs_fable.py first.")
        sys.exit(1)
    df = pd.read_csv(IN_PATH)
    metric_cols = [c for c in ("BLEU-4", "ROUGE-L", "METEOR", "BERTScore") if c in df.columns]
    if not metric_cols:
        print("No metric columns found in per-sample CSV.")
        sys.exit(1)

    systems = sorted(df["system"].unique())
    summary = df.groupby("system")[metric_cols].mean().reindex(systems)
    n = df.groupby("system").size().reindex(systems)
    print("=== Mean lexical similarity to NorQuAD reference question ===")
    print(summary.round(4).to_string())

    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(systems)}

    ncols = 2
    nrows = -(-len(metric_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), dpi=200)
    axes = axes.reshape(-1)

    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        sub = summary[metric].sort_values(ascending=True)
        colors = [color_map[s] for s in sub.index]
        ax.barh(sub.index, sub.values, color=colors, zorder=3)
        xmax = max(sub.values.max() * 1.25, 0.05)
        for y, v in enumerate(sub.values):
            ax.text(v + xmax * 0.02, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color="#E5E1D6", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8.5)

    for j in range(len(metric_cols), len(axes)):
        axes[j].axis("off")

    n_shared = int(n.iloc[0]) if len(n) else 0
    fig.suptitle(f"QGEval lexical similarity to NorQuAD reference (n={n_shared} shared passages)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"\nSaved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
