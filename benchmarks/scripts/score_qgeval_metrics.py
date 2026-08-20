"""Reference-based automatic metrics for the QGEval-style evaluation, plus the
metric-vs-judge correlation analysis.

Emulates two analyses from QGEval (Fu et al., EMNLP 2024):

  * Reference-based metrics scoring the generated question against the
    dataset's own question (their Appendix D / Table 5 top half): BLEU-4,
    ROUGE-L, METEOR, BERTScore.
  * Pearson correlation between each automatic metric and the per-dimension
    judge scores (their Table 5) — the paper's headline negative result was
    that these correlations are low, i.e. n-gram overlap with a reference is
    a poor proxy for question quality.

Here NorQuAD's own question is the reference and Fable's answer-conditioned
question is the candidate, so the correlation is computed across samples for
a single system.

Usage:
    python3 score_qgeval_metrics.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qgeval_dimensions import DIMENSIONS, DISPLAY_NAMES  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JUDGE_PATH = RESULTS_DIR / "qgeval_judge_scores.json"
PER_SAMPLE_PATH = RESULTS_DIR / "qgeval_metrics_per_sample.csv"
CORR_PATH = RESULTS_DIR / "qgeval_metric_judge_correlation.csv"

BERT_MODEL = "bert-base-multilingual-cased"


def compute_bleu4(cands, refs):
    import sacrebleu
    # Sentence-level BLEU-4 with smoothing; corpus BLEU is degenerate per-sample.
    return [
        sacrebleu.sentence_bleu(c, [r], smooth_method="exp").score / 100.0
        for c, r in zip(cands, refs)
    ]


def compute_rouge_l(cands, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [scorer.score(r, c)["rougeL"].fmeasure for c, r in zip(cands, refs)]


def compute_meteor(cands, refs):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    # No Norwegian WordNet, so METEOR here reduces to exact/stem alignment.
    return [meteor_score([r.split()], c.split()) for c, r in zip(cands, refs)]


def compute_bertscore(cands, refs):
    import bert_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = bert_score.score(cands, refs, model_type=BERT_MODEL,
                                    num_layers=9, verbose=False, batch_size=32)
    return f1.tolist()


METRICS = {
    "BLEU-4": compute_bleu4,
    "ROUGE-L": compute_rouge_l,
    "METEOR": compute_meteor,
    "BERTScore": compute_bertscore,
}


def main():
    records = json.loads(JUDGE_PATH.read_text(encoding="utf-8"))
    records = [r for r in records if not r.get("error")]
    if not records:
        print("No scored records found — run score_qgeval_dimensions.py first.")
        sys.exit(1)
    print(f"Loaded {len(records)} judged samples")

    cands = [r["fable_question"] for r in records]
    refs = [r["reference_question"] for r in records]

    df = pd.DataFrame({
        "qid": [r["qid"] for r in records],
        "doc_id": [r["doc_id"] for r in records],
        "fable_question": cands,
        "reference_question": refs,
    })

    for name, fn in METRICS.items():
        print(f"  computing {name} ...")
        df[name] = fn(cands, refs)

    for dim in DIMENSIONS:
        df[f"fable_{dim}"] = [r["fable_scores"][dim] for r in records]
        df[f"reference_{dim}"] = [r["reference_scores"][dim] for r in records]

    df.to_csv(PER_SAMPLE_PATH, index=False)
    print(f"\nSaved per-sample metrics -> {PER_SAMPLE_PATH}")

    print("\n=== Reference-based metrics (Fable question vs. NorQuAD question) ===")
    for name in METRICS:
        print(f"  {name:11s} mean={df[name].mean():.4f}  median={df[name].median():.4f}")

    print("\n=== Pearson correlation: automatic metric vs. judge dimension ===")
    rows = []
    for name in METRICS:
        row = {"metric": name}
        for dim in DIMENSIONS:
            col = df[f"fable_{dim}"]
            # Constant judge scores make correlation undefined, not zero.
            if col.nunique() < 2:
                row[DISPLAY_NAMES[dim]] = float("nan")
            else:
                row[DISPLAY_NAMES[dim]] = pearsonr(df[name], col)[0]
        rows.append(row)
    corr = pd.DataFrame(rows).set_index("metric")
    corr.to_csv(CORR_PATH)
    print(corr.round(3).to_string())
    print(f"\nSaved correlations -> {CORR_PATH}")

    degenerate = [DISPLAY_NAMES[d] for d in DIMENSIONS if df[f"fable_{d}"].nunique() < 2]
    if degenerate:
        print(f"\nNote: judge gave a constant score on {', '.join(degenerate)} — "
              "correlation is undefined there (reported as NaN, not 0).")


if __name__ == "__main__":
    main()
