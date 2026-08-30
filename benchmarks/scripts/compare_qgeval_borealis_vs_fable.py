"""Lexical comparison of Borealis-27b QGEval question generations against
Claude Fable 5's, on their shared samples — no LLM judge / API key required.

Both Borealis variants (full release and instruct-preview) completed all 200
QGEval samples, matching Fable's full run, so this compares all three systems
on the same 200 (passage, answer) -> question items, scoring each system's
generated question against the NorQuAD reference question with BLEU-4,
ROUGE-L, METEOR, and (if bert_score is installed) BERTScore.

This is the reference-based half of score_qgeval_metrics.py's methodology,
generalized to multiple systems and without requiring
qgeval_judge_scores.json (which needs a working OpenRouter key to produce).

Usage:
    python3 compare_qgeval_borealis_vs_fable.py
"""
import json
import warnings
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
QUESTIONS_DIR = RESULTS_DIR / "qgeval_questions"
FABLE_PATH = RESULTS_DIR / "qgeval_fable_questions.json"
OUT_PER_SAMPLE = RESULTS_DIR / "qgeval_borealis_vs_fable_per_sample.csv"
OUT_SUMMARY = RESULTS_DIR / "qgeval_borealis_vs_fable_summary.csv"

SYSTEMS = {
    "claude-fable-5": FABLE_PATH,
    "borealis-27b": QUESTIONS_DIR / "borealis-27b.json",
    "borealis-27b-instruct-preview": QUESTIONS_DIR / "borealis-27b-instruct-preview.json",
}


def compute_bleu4(cands, refs):
    import sacrebleu
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
    return [meteor_score([r.split()], c.split()) for c, r in zip(cands, refs)]


def compute_bertscore(cands, refs):
    import bert_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = bert_score.score(
            cands, refs, model_type="bert-base-multilingual-cased",
            num_layers=9, verbose=False, batch_size=32,
        )
    return f1.tolist()


METRICS = [
    ("BLEU-4", compute_bleu4),
    ("ROUGE-L", compute_rouge_l),
    ("METEOR", compute_meteor),
    ("BERTScore", compute_bertscore),
]


def load(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["qid"]: r for r in records if r.get("generated_question") and not r.get("error")}


def main():
    per_system = {name: load(path) for name, path in SYSTEMS.items()}
    for name, recs in per_system.items():
        print(f"{name}: {len(recs)} usable records")

    shared_qids = set.intersection(*(set(r.keys()) for r in per_system.values()))
    print(f"\nShared qids across all systems: {len(shared_qids)}")
    shared_qids = sorted(shared_qids)

    frames = []
    for name, recs in per_system.items():
        cands = [recs[q]["generated_question"] for q in shared_qids]
        refs = [recs[q]["reference_question"] for q in shared_qids]
        df = pd.DataFrame({
            "system": name,
            "qid": shared_qids,
            "doc_id": [recs[q]["doc_id"] for q in shared_qids],
            "generated_question": cands,
            "reference_question": refs,
        })
        for metric_name, fn in METRICS:
            print(f"  [{name}] computing {metric_name} ...")
            try:
                df[metric_name] = fn(cands, refs)
            except ModuleNotFoundError as e:
                print(f"    skipping {metric_name}: {e}")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full.to_csv(OUT_PER_SAMPLE, index=False)
    print(f"\nSaved per-sample scores -> {OUT_PER_SAMPLE}")

    metric_cols = [m for m, _ in METRICS if m in full.columns]
    summary = full.groupby("system")[metric_cols].agg(["mean", "median"])
    summary.to_csv(OUT_SUMMARY)
    print(f"\n=== Mean / median lexical similarity to NorQuAD reference question, "
          f"n={len(shared_qids)} shared samples ===")
    print(summary.round(4).to_string())
    print(f"\nSaved summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
