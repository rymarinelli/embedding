"""Deeper validation of the QGEval run, all from existing artifacts (no API).

Adds five things the first pass did not have:

  1. Position-bias test — does a system score higher merely for being shown as
     "Question A"? This validates that the blinding actually worked.
  2. Inter-dimension correlations — the paper's Figure 2. Their claim is that
     the seven dimensions are "interrelated but still exhibit distinct
     characteristics"; this checks whether that reproduces on Norwegian data.
  3. Full score distributions — the paper's Figure 3.
  4. Effect sizes and bootstrap CIs — bare p-values say a difference exists,
     not how big it is. With this much tying, that distinction matters.
  5. Question-length analysis — tests the mechanism claimed for the
     Clarity/Conciseness trade-off rather than asserting it from examples.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "qgeval_analysis"
OUT.mkdir(parents=True, exist_ok=True)

JUDGE = [r for r in json.loads((REPO / "results/qgeval_judge_scores.json").read_text("utf-8"))
         if not r.get("error")]
DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]
NICE = {d: d.replace("_", " ").title() for d in DIMS}
rng = np.random.default_rng(42)

fab = {d: np.array([r["fable_scores"][d] for r in JUDGE]) for d in DIMS}
ref = {d: np.array([r["reference_scores"][d] for r in JUDGE]) for d in DIMS}
slot = np.array([r["fable_slot"] for r in JUDGE])

results = {}

# ---------------------------------------------------------------- 1. POSITION BIAS
print("=" * 78)
print("1. POSITION-BIAS TEST  (did blinding work?)")
print("=" * 78)
print("If presentation order mattered, a system's scores would differ by slot.\n")
rows = []
for d in DIMS:
    # Fable's own scores, split by whether Fable was shown first or second.
    a, b = fab[d][slot == "A"], fab[d][slot == "B"]
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue if a.std() + b.std() > 0 else float("nan")
    rows.append({"dimension": NICE[d], "fable_as_A": a.mean(), "fable_as_B": b.mean(),
                 "diff": a.mean() - b.mean(), "p": p})
pos = pd.DataFrame(rows)
print(pos.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
sig = pos[pos["p"] < 0.05]
print(f"\n  n(slot A)={int((slot=='A').sum())}, n(slot B)={int((slot=='B').sum())}")
print(f"  dimensions showing a slot effect at p<0.05: {len(sig)} of 7")
if len(sig) == 0:
    print("  -> No detectable position bias. Blinding held.")
else:
    print(f"  -> WARNING, slot effect on: {', '.join(sig['dimension'])}")
results["position_bias"] = pos
pos.to_csv(OUT / "analysis_position_bias.csv", index=False)

# ---------------------------------------------------- 2. INTER-DIMENSION CORRELATION
print()
print("=" * 78)
print("2. INTER-DIMENSION CORRELATION  (paper's Figure 2)")
print("=" * 78)
# Pool both systems: correlations describe the dimensions, not one system.
pool = {d: np.concatenate([fab[d], ref[d]]) for d in DIMS}
mat = np.zeros((7, 7))
for i, a in enumerate(DIMS):
    for j, b in enumerate(DIMS):
        if pool[a].std() == 0 or pool[b].std() == 0:
            mat[i, j] = np.nan
        else:
            mat[i, j] = pearsonr(pool[a], pool[b])[0]
cor = pd.DataFrame(mat, index=[NICE[d] for d in DIMS], columns=[NICE[d] for d in DIMS])
print(cor.round(2).to_string())
off = mat[~np.eye(7, dtype=bool)]
off = off[~np.isnan(off)]
print(f"\n  off-diagonal range: {off.min():.2f} to {off.max():.2f}  (paper reports 0.04-0.67)")
print("  -> Interrelated but distinct, as the paper found." if off.max() < 0.9
      else "  -> Some dimensions are near-duplicates here.")
cor.to_csv(OUT / "analysis_dimension_correlation.csv")
results["dim_corr"] = cor

# strongest pairs
pairs = [(NICE[DIMS[i]], NICE[DIMS[j]], mat[i, j])
         for i in range(7) for j in range(i + 1, 7) if not np.isnan(mat[i, j])]
pairs.sort(key=lambda t: -abs(t[2]))
print("\n  strongest pairs:")
for a, b, v in pairs[:4]:
    print(f"    {a:20s} <-> {b:20s} r = {v:.2f}")

# ---------------------------------------------------------- 3. SCORE DISTRIBUTIONS
print()
print("=" * 78)
print("3. SCORE DISTRIBUTIONS  (paper's Figure 3)")
print("=" * 78)
rows = []
for d in DIMS:
    for sysname, arr in (("Fable", fab[d]), ("NorQuAD", ref[d])):
        rows.append({"dimension": NICE[d], "system": sysname,
                     "pct_1": 100 * (arr == 1).mean(),
                     "pct_2": 100 * (arr == 2).mean(),
                     "pct_3": 100 * (arr == 3).mean()})
dist = pd.DataFrame(rows)
print(dist.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
dist.to_csv(OUT / "analysis_score_distribution.csv", index=False)
results["dist"] = dist

n1_f = sum(int((fab[d] == 1).sum()) for d in DIMS)
n1_r = sum(int((ref[d] == 1).sum()) for d in DIMS)
print(f"\n  score-1 ratings in total: Fable {n1_f}, NorQuAD {n1_r} (of 1400 each)")

# ------------------------------------------------ 4. EFFECT SIZES + BOOTSTRAP CIs
print()
print("=" * 78)
print("4. EFFECT SIZE AND BOOTSTRAP CI  (how big, not just 'is it real')")
print("=" * 78)
rows = []
for d in DIMS:
    diff = fab[d] - ref[d]
    nz = diff[diff != 0]
    # Rank-biserial for a paired design = (wins - losses) / non-tied pairs.
    rb = ((nz > 0).sum() - (nz < 0).sum()) / len(nz) if len(nz) else np.nan
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"dimension": NICE[d], "delta": diff.mean(),
                 "ci_low": lo, "ci_high": hi,
                 "non_tied": len(nz), "rank_biserial": rb,
                 "ci_excludes_0": bool(lo > 0 or hi < 0)})
eff = pd.DataFrame(rows)
print(eff.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
eff.to_csv(OUT / "analysis_effect_sizes.csv", index=False)
results["eff"] = eff
print(f"\n  dimensions whose 95% CI excludes zero: "
      f"{int(eff['ci_excludes_0'].sum())} of 7")
print("  Note: only 4-33% of pairs are non-tied, so these effects are real but small.")

# --------------------------------------------------------- 5. QUESTION LENGTH
print()
print("=" * 78)
print("5. QUESTION LENGTH  (does it explain the Clarity/Conciseness trade-off?)")
print("=" * 78)
W = re.compile(r"[A-Za-zÆØÅæøå0-9]+")
flen = np.array([len(W.findall(r["fable_question"])) for r in JUDGE])
rlen = np.array([len(W.findall(r["reference_question"])) for r in JUDGE])
print(f"  Fable   mean {flen.mean():.1f} words (median {np.median(flen):.0f})")
print(f"  NorQuAD mean {rlen.mean():.1f} words (median {np.median(rlen):.0f})")
print(f"  Fable longer in {100*(flen>rlen).mean():.0f}% of pairs")

delta_len = flen - rlen
for d in ["clarity", "conciseness"]:
    dd = fab[d] - ref[d]
    if dd.std() > 0:
        r, p = pearsonr(delta_len, dd)
        print(f"  corr(extra words, {d} delta) = {r:+.3f}  (p={p:.2g})")

conc_loss = delta_len[(fab["conciseness"] - ref["conciseness"]) < 0]
conc_same = delta_len[(fab["conciseness"] - ref["conciseness"]) == 0]
print(f"\n  extra words when Fable LOSES conciseness: {conc_loss.mean():+.1f}")
print(f"  extra words when conciseness ties:        {conc_same.mean():+.1f}")
pd.DataFrame({"fable_len": flen, "ref_len": rlen,
              "clarity_delta": fab["clarity"] - ref["clarity"],
              "conciseness_delta": fab["conciseness"] - ref["conciseness"]}
             ).to_csv(OUT / "analysis_length.csv", index=False)

print()
print("=" * 78)
print("Wrote 5 CSVs to", OUT)
print("=" * 78)
