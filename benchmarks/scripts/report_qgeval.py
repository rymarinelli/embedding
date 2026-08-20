"""Summarise the QGEval-style evaluation: per-dimension scores for Fable vs.
NorQuAD's own questions, paired significance tests, and score distributions.

Emulates QGEval's Table 4 (per-dimension mean scores, with the human
reference scored as its own row) and Figure 3 (score distributions).

Statistical note: the paper compares model rankings with t-tests. Scores here
are ordinal on a 1-3 scale and the two systems are scored on the *same* items,
so this uses a paired Wilcoxon signed-rank test instead, which does not assume
interval-scaled or normally distributed differences.

Usage:
    python3 report_qgeval.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qgeval_dimensions import DIMENSIONS, DISPLAY_NAMES, LINGUISTIC  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JUDGE_PATH = RESULTS_DIR / "qgeval_judge_scores.json"
TABLE_PATH = RESULTS_DIR / "qgeval_dimension_scores.csv"
PLOT_PATH = RESULTS_DIR / "qgeval_dimension_comparison.png"
CORR_PATH = RESULTS_DIR / "qgeval_metric_judge_correlation.csv"
CORR_PLOT_PATH = RESULTS_DIR / "qgeval_metric_judge_correlation.png"

FABLE_COLOR = "#1F6F78"
REF_COLOR = "#96712B"


def main():
    records = json.loads(JUDGE_PATH.read_text(encoding="utf-8"))
    records = [r for r in records if not r.get("error")]
    if not records:
        print("No scored records — run score_qgeval_dimensions.py first.")
        sys.exit(1)
    judge = records[0].get("judge_model", "unknown")
    print(f"Loaded {len(records)} judged pairs (judge: {judge})\n")

    fable = {d: np.array([r["fable_scores"][d] for r in records]) for d in DIMENSIONS}
    ref = {d: np.array([r["reference_scores"][d] for r in records]) for d in DIMENSIONS}

    rows = []
    for d in DIMENSIONS:
        f, g = fable[d], ref[d]
        diff = f - g
        wins, losses, ties = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        # Wilcoxon is undefined when every pair is tied.
        if np.all(diff == 0):
            p = float("nan")
        else:
            p = wilcoxon(f, g, zero_method="wilcox").pvalue
        rows.append({
            "dimension": DISPLAY_NAMES[d],
            "aspect": "linguistic" if d in LINGUISTIC else "task-oriented",
            "fable_mean": f.mean(),
            "reference_mean": g.mean(),
            "delta": f.mean() - g.mean(),
            "fable_wins": wins,
            "ties": ties,
            "reference_wins": losses,
            "wilcoxon_p": p,
        })

    df = pd.DataFrame(rows)
    avg_f = np.mean([fable[d].mean() for d in DIMENSIONS])
    avg_r = np.mean([ref[d].mean() for d in DIMENSIONS])
    df.loc[len(df)] = {
        "dimension": "Avg.", "aspect": "", "fable_mean": avg_f,
        "reference_mean": avg_r, "delta": avg_f - avg_r,
        "fable_wins": np.nan, "ties": np.nan, "reference_wins": np.nan,
        "wilcoxon_p": np.nan,
    }
    df.to_csv(TABLE_PATH, index=False)

    print("=== Per-dimension scores (1-3, higher is better) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved table -> {TABLE_PATH}")

    all_tied = [DISPLAY_NAMES[d] for d in DIMENSIONS if np.all(fable[d] == ref[d])]
    if all_tied:
        print(f"\nNote: every pair tied on {', '.join(all_tied)} — "
              "Wilcoxon undefined there (NaN, not a p-value of 1).")

    # ---- Plot: grouped means + score distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=200)

    ax = axes[0]
    x = np.arange(len(DIMENSIONS))
    w = 0.38
    fmeans = [fable[d].mean() for d in DIMENSIONS]
    rmeans = [ref[d].mean() for d in DIMENSIONS]
    ax.bar(x - w / 2, fmeans, w, label="Fable (generert)", color=FABLE_COLOR, zorder=3)
    ax.bar(x + w / 2, rmeans, w, label="NorQuAD (referanse)", color=REF_COLOR, zorder=3)
    for i, (a, b) in enumerate(zip(fmeans, rmeans)):
        ax.text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[d].replace(" ", "\n") for d in DIMENSIONS], fontsize=8.5)
    ax.set_ylabel("Gjennomsnittsskår (1-3)")
    ax.set_ylim(1, 3.18)
    ax.set_title("Skår per dimensjon")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E1D6", zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels, fpct, rpct = [], [], []
    for d in DIMENSIONS:
        labels.append(DISPLAY_NAMES[d].replace(" ", "\n"))
        fpct.append(100 * (fable[d] == 3).mean())
        rpct.append(100 * (ref[d] == 3).mean())
    ax.bar(x - w / 2, fpct, w, label="Fable (generert)", color=FABLE_COLOR, zorder=3)
    ax.bar(x + w / 2, rpct, w, label="NorQuAD (referanse)", color=REF_COLOR, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Andel med toppskår 3 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Andel perfekte skår")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E1D6", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"QGEval-dimensjoner på NorQuAD (n={len(records)}, dommer: {judge})",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"Saved plot -> {PLOT_PATH}")

    plot_correlation_heatmap()


def plot_correlation_heatmap():
    """Render the metric-vs-judge correlation table (QGEval's Table 5) as a heatmap."""
    if not CORR_PATH.exists():
        print(f"(skipping correlation heatmap — run score_qgeval_metrics.py first)")
        return

    corr = pd.read_csv(CORR_PATH, index_col="metric")
    data = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=200)
    # Symmetric scale centred on 0 so sign is readable at a glance; |r| here is
    # small, so a fixed +/-1 scale would render everything as flat white.
    lim = max(0.2, float(np.nanmax(np.abs(data))))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in corr.columns], fontsize=8.5)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = "n/a" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="#16202A")

    ax.set_title("Pearson-korrelasjon: automatisk metrikk vs. dommerskår\n"
                 "(alle nær null — overlapp med referansen forutsier ikke kvalitet)",
                 fontsize=10.5, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(CORR_PLOT_PATH, bbox_inches="tight")
    print(f"Saved correlation heatmap -> {CORR_PLOT_PATH}")


if __name__ == "__main__":
    main()
