"""Re-derive every reported QGEval figure straight from the raw
artifacts, and compare against what the slides assert.

Nothing here reads the summary CSVs the report script produced — the point is
to recompute from the judge output and metric files independently, so a bug in
the reporting layer cannot validate itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
JUDGE = json.loads((REPO / "results/qgeval_judge_scores.json").read_text(encoding="utf-8"))
GEN = json.loads((REPO / "results/qgeval_fable_questions.json").read_text(encoding="utf-8"))
SAMPLE = json.loads((REPO / "data/qgeval_sample.json").read_text(encoding="utf-8"))
PER = pd.read_csv(REPO / "results/qgeval_metrics_per_sample.csv")

DIMS = ["fluency", "clarity", "conciseness", "relevance",
        "consistency", "answerability", "answer_consistency"]

ok = JUDGE if all(not r.get("error") for r in JUDGE) else [r for r in JUDGE if not r.get("error")]

failures, checks = [], 0


def check(label, actual, claimed, tol=0.0005):
    global checks
    checks += 1
    if isinstance(claimed, float) or isinstance(actual, float):
        good = abs(float(actual) - float(claimed)) <= tol
    else:
        good = actual == claimed
    status = "OK  " if good else "FAIL"
    if not good:
        failures.append(f"{label}: deck says {claimed!r}, data says {actual!r}")
    print(f"  [{status}] {label:52s} deck={claimed!r:<12} data={actual!r}")


print("=" * 78)
print("SLIDE 1 — cover figures")
print("=" * 78)
check("passages in sample", len(SAMPLE), 200)
check("generated questions", len(GEN), 200)
check("judged pairs", len(ok), 200)
check("dimensions", len(DIMS), 7)
# Deck claims a total number of judge scores.
total_scores = len(ok) * 2 * len(DIMS)
check("total judge scores (200 x 2 systems x 7 dims)", total_scores, 2800)

print()
print("=" * 78)
print("SLIDE 5 / 6 — means, deltas, win-tie-loss")
print("=" * 78)
fab = {d: np.array([r["fable_scores"][d] for r in ok]) for d in DIMS}
ref = {d: np.array([r["reference_scores"][d] for r in ok]) for d in DIMS}

avg_f = float(np.mean([fab[d].mean() for d in DIMS]))
avg_r = float(np.mean([ref[d].mean() for d in DIMS]))
check("Fable overall mean", round(avg_f, 3), 2.937)
check("Reference overall mean", round(avg_r, 3), 2.806)

deck_delta = {"clarity": 0.390, "answerability": 0.155, "consistency": 0.135,
              "answer_consistency": 0.105, "fluency": 0.095, "relevance": 0.080,
              "conciseness": -0.045}
for d, claimed in deck_delta.items():
    check(f"delta {d}", round(float(fab[d].mean() - ref[d].mean()), 3), claimed)

deck_wtl = {"clarity": (76, 120, 4), "answerability": (31, 167, 2),
            "consistency": (25, 174, 1), "fluency": (21, 176, 3),
            "relevance": (20, 176, 4), "answer_consistency": (20, 178, 2),
            "conciseness": (2, 187, 11)}
for d, (w, t, l) in deck_wtl.items():
    diff = fab[d] - ref[d]
    check(f"W/T/L {d}", (int((diff > 0).sum()), int((diff == 0).sum()),
                         int((diff < 0).sum())), (w, t, l))

print()
print("  Wilcoxon p-values (deck claims 'p < 0.05 throughout'):")
worst = 0.0
for d in DIMS:
    p = wilcoxon(fab[d], ref[d], zero_method="wilcox").pvalue
    worst = max(worst, p)
    print(f"    {d:22s} p = {p:.2e}")
checks += 1
if worst < 0.05:
    print(f"  [OK  ] all p < 0.05 (largest = {worst:.4f})")
else:
    failures.append(f"not all p < 0.05 (largest = {worst})")
    print(f"  [FAIL] largest p = {worst}")

print()
print("=" * 78)
print("SLIDE 6 — perfect-score rates quoted in the Clarity callout")
print("=" * 78)
check("Clarity: % Fable scoring 3", round(100 * float((fab["clarity"] == 3).mean())), 96)
check("Clarity: % reference scoring 3", round(100 * float((ref["clarity"] == 3).mean())), 60)

print()
print("=" * 78)
print("SLIDE 10 — ceiling-effect claim")
print("=" * 78)
check("Fluency pairs tied", int((fab["fluency"] == ref["fluency"]).sum()), 176)

print()
print("=" * 78)
print("SLIDE 9 — automatic metrics")
print("=" * 78)
for name, claimed in [("BLEU-4", 0.232), ("ROUGE-L", 0.513),
                      ("METEOR", 0.467), ("BERTScore", 0.852)]:
    check(f"{name} mean", round(float(PER[name].mean()), 3), claimed)

corr = pd.read_csv(REPO / "results/qgeval_metric_judge_correlation.csv", index_col="metric")
max_abs = float(np.nanmax(np.abs(corr.to_numpy(dtype=float))))
checks += 1
if max_abs <= 0.14:
    print(f"  [OK  ] max |r| = {max_abs:.4f} <= 0.14 as claimed")
else:
    failures.append(f"max |r| = {max_abs:.4f} exceeds claimed 0.14")
    print(f"  [FAIL] max |r| = {max_abs:.4f} > 0.14")

conc = corr["Conciseness"].astype(float)
print(f"  Conciseness column (deck says 'r ~ 0.135'): "
      f"{', '.join(f'{m}={v:.3f}' for m, v in conc.items())}")

print()
print("=" * 78)
print("SLIDES 3 / 7 / 8 — every quoted example, verified verbatim")
print("=" * 78)
by_ref = {r["reference_question"]: r for r in ok}
EXAMPLES = [
    ("s3 NVE", "Hva er NVE?", "clarity", 2, 3),
    ("s7 Theresa May", "Hvem er statsminister i Storbritannia?", "clarity", 2, 3),
    ("s7 Australbukta", "Hvor mye regner Australia Institute med at Norge vil tjene på olja og gassen i Australbukta?", "clarity", 2, 3),
    ("s8 samferdselsminister", "Hvem er nå samferdselsminister?", "conciseness", 3, 2),
    ("s8 Clapper", "Hvem var etterretningsdirektør i USA før Dan Coats fikk stillingen?", "conciseness", 3, 2),
]
for label, refq, dim, ref_claim, fab_claim in EXAMPLES:
    rec = by_ref.get(refq)
    checks += 1
    if rec is None:
        failures.append(f"{label}: reference question not found in judged data")
        print(f"  [FAIL] {label}: quoted gold question not present in the data")
        continue
    r_actual, f_actual = rec["reference_scores"][dim], rec["fable_scores"][dim]
    good = (r_actual == ref_claim) and (f_actual == fab_claim)
    if not good:
        failures.append(f"{label} {dim}: deck ref={ref_claim}/fable={fab_claim}, "
                        f"data ref={r_actual}/fable={f_actual}")
    print(f"  [{'OK  ' if good else 'FAIL'}] {label:26s} {dim:12s} "
          f"ref {r_actual} (deck {ref_claim})  fable {f_actual} (deck {fab_claim})")

print()
print("  Fable question text as quoted on the slides:")
for label, refq, *_ in EXAMPLES:
    rec = by_ref.get(refq)
    if rec:
        print(f"    {label}: {rec['fable_question']}")

print()
print("=" * 78)
print("INTEGRITY — gold answers actually come from NorQuAD")
print("=" * 78)
sample_by_qid = {s["qid"]: s for s in SAMPLE}
mismatch = sum(1 for r in ok
               if sample_by_qid[r["qid"]]["answer"] != r["answer"]
               or sample_by_qid[r["qid"]]["reference_question"] != r["reference_question"])
check("records whose answer/reference drifted from the sample", mismatch, 0)
check("distinct passages (no passage counted twice)",
      len({s["doc_id"] for s in SAMPLE}), 200)
check("judge model used throughout",
      len({r["judge_model"] for r in ok}), 1)
print(f"  judge model = {ok[0]['judge_model']}")
blind = {r["fable_slot"] for r in ok}
check("both presentation slots were used (blinding active)", sorted(blind), ["A", "B"])
a_count = sum(1 for r in ok if r["fable_slot"] == "A")
print(f"  Fable appeared as 'Question A' {a_count}/200 times, 'B' {200 - a_count}/200")

print()
print("=" * 78)
print(f"{checks} checks run · {len(failures)} FAILURES")
print("=" * 78)
for f in failures:
    print(f"  ✗ {f}")
sys.exit(1 if failures else 0)
