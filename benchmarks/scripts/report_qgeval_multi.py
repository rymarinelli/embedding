"""Report QGEval's 7 dimensions across every model in the roster, from
score_qgeval_dimensions_multi.py's output.

Produces:
  - results/qgeval_multi_dimension_scores.csv   mean score per model x dimension
  - results/qgeval_multi_dimension_scores.png   grouped bar chart, one panel
                                                 per dimension (Fluency included)

Usage:
    python3 report_qgeval_multi.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qgeval_dimensions import DIMENSIONS, DISPLAY_NAMES  # noqa: E402
from score_qgeval_dimensions_multi import REFERENCE_KEY  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IN_PATH = RESULTS_DIR / "qgeval_judge_scores_multi.json"
TABLE_PATH = RESULTS_DIR / "qgeval_multi_dimension_scores.csv"
PLOT_PATH = RESULTS_DIR / "qgeval_multi_dimension_scores.png"

DISPLAY_LABEL = {REFERENCE_KEY: "NorQuAD (human)"}
PALETTE = ["#1F6F78", "#96712B", "#6E4C9E", "#3E7A3E", "#B0473E",
           "#3E6E9E", "#8C6F3E", "#5A5A5A"]


def main():
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist — run score_qgeval_dimensions_multi.py first.")
        sys.exit(1)
    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    records = [r for r in records if not r.get("error")]
    if not records:
        print("No scored records — run score_qgeval_dimensions_multi.py first.")
        sys.exit(1)

    systems = sorted({s for r in records for s in r["scores"]})
    print(f"Loaded {len(records)} judged passages across {len(systems)} systems: "
          f"{', '.join(systems)}")

    rows = []
    for sys_ in systems:
        scores_by_dim = {d: [] for d in DIMENSIONS}
        for r in records:
            if sys_ not in r["scores"]:
                continue
            for d in DIMENSIONS:
                scores_by_dim[d].append(r["scores"][sys_][d])
        row = {"system": DISPLAY_LABEL.get(sys_, sys_), "n": len(scores_by_dim["fluency"])}
        for d in DIMENSIONS:
            row[DISPLAY_NAMES[d]] = np.mean(scores_by_dim[d]) if scores_by_dim[d] else np.nan
        row["Avg."] = np.mean([row[DISPLAY_NAMES[d]] for d in DIMENSIONS])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Avg.", ascending=False)
    df.to_csv(TABLE_PATH, index=False)

    print("\n=== Mean score per dimension (1-3, higher is better) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved table -> {TABLE_PATH}")

    # ---- Plot: one panel per dimension, bars sorted by that dimension ----
    n_dims = len(DIMENSIONS)
    ncols = 4
    nrows = -(-n_dims // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), dpi=200)
    axes = np.array(axes).reshape(-1)

    color_map = {row["system"]: PALETTE[i % len(PALETTE)] for i, row in enumerate(rows)}
    # Human reference always the same neutral color, distinct from any model.
    ref_label = DISPLAY_LABEL[REFERENCE_KEY]
    if ref_label in df["system"].values:
        color_map[ref_label] = "#96712B"

    for i, d in enumerate(DIMENSIONS):
        ax = axes[i]
        name = DISPLAY_NAMES[d]
        sub = df.sort_values(name, ascending=True)
        colors = [color_map[s] for s in sub["system"]]
        ax.barh(sub["system"], sub[name], color=colors, zorder=3)
        for y, v in enumerate(sub[name]):
            ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlim(1, 3.25)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color="#E5E1D6", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8.5)

    for j in range(n_dims, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"QGEval dimensions across {len(systems)} systems on NorQuAD (n={len(records)} passages)",
                fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"Saved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
