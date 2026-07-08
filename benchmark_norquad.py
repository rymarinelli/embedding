#!/usr/bin/env python3
"""
Benchmark OpenRouter free-tier models on the news portion of NorQuAD
(Norwegian extractive question answering, Ivanova et al. 2023, NoDaLiDa).

- Loads the news-only NorQuAD test split directly from the NorQuAD GitHub
  repo: data/evaluation/news/test_dataset_flattened.json
  (github.com/ltgoslo/NorQuAD/tree/main/data/annotation/news is the raw,
  un-joined annotation data - id/answer/document_id only, no article text -
  so the flattened SQuAD-style file under data/evaluation/news is used
  instead, since it already joins each question with its source context).
- Runs a Norwegian reading-comprehension prompt adapted from NorEval Prompt B
  for NorQuAD (Mikhailov et al. 2025, Appendix A): the model is given the
  article text and instructed to answer using only information explicitly
  stated in it - i.e. the prompt is grounded in the provided context rather
  than the model's parametric knowledge.
- Scores each generated answer against all annotated reference answers with
  SQuAD-style Exact Match and token-overlap F1, taking the max over
  references per example.
- As a secondary signal, an LLM judge (default: meta-llama/llama-3.3-70b-instruct,
  the same judge NorEval itself uses) independently rates each answer as
  correct/incorrect given the context, question, and gold answers. This runs
  alongside EM/F1 rather than replacing them, so you can see where the judge
  and the automatic metric agree or disagree (e.g. a verbose-but-correct
  answer that EM/F1 penalizes for wording). Set use_judge=False to skip it
  and halve the number of API calls.
- Reports per-model averages.

Usage (command line):
    export OPENROUTER_API_KEY="sk-or-..."
    pip install requests pandas --break-system-packages
    python benchmark_norquad.py

Usage (Colab):
    from google.colab import userdata
    OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
    from benchmark_norquad import run
    results_df = run(api_key=OPENROUTER_API_KEY, n_examples=20)
"""

import collections
import re
import string
import sys
import time

import requests

from openrouter_client import get_api_key, call_model, build_headers
from llm_judge import DEFAULT_JUDGE_MODEL, judge_norquad_answer

NORQUAD_NEWS_TEST_URL = (
    "https://raw.githubusercontent.com/ltgoslo/NorQuAD/main/"
    "data/evaluation/news/test_dataset_flattened.json"
)

# Models that returned valid completions in the last free-tier connectivity test.
DEFAULT_MODELS = [
    "cohere/north-mini-code:free"
]

# Adapted from NorEval Prompt B for NorQuAD (Mikhailov et al. 2025, Appendix A).
# The added clause makes the grounding explicit: the answer must come from the
# text, not from the model's own knowledge, and should be a short extracted span.
PROMPT_TEMPLATE = (
    "Tekst: {context}\n\n"
    "Gitt teksten over, svar kort og presist på følgende spørsmål ved å bruke "
    "kun informasjon som eksplisitt står i teksten: \"{question}\"\n\n"
    "Svar:"
)


def load_norquad_news(n_examples: int = None):
    resp = requests.get(NORQUAD_NEWS_TEST_URL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    examples = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                answers = [a["text"].strip() for a in qa.get("answers", []) if a.get("text")]
                if not answers:
                    continue
                examples.append({
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "answers": answers,
                })
    if n_examples is not None:
        examples = examples[:n_examples]
    return examples


# ── SQuAD-style scoring (Rajpurkar et al. 2016), Norwegian-adapted: no article
# removal (Norwegian marks definiteness with suffixes, not standalone words). ──

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(gt, prediction) for gt in ground_truths)


def run(
    models=None,
    n_examples=20,
    delay=2.0,
    api_key=None,
    use_judge=True,
    judge_model=DEFAULT_JUDGE_MODEL,
):
    """
    Run the NorQuAD (news) benchmark against a list of OpenRouter free models.

    Usage:
        from benchmark_norquad import run
        results_df = run(api_key=OPENROUTER_API_KEY, n_examples=20)
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Missing dependency: pip install pandas --break-system-packages")

    api_key = get_api_key(explicit_key=api_key)
    headers = build_headers(api_key)

    models = models or DEFAULT_MODELS
    print(f"Loading {n_examples} NorQuAD news test examples...")
    dataset = load_norquad_news(n_examples)
    print(f"Loaded {len(dataset)} examples. Testing {len(models)} model(s).")
    if use_judge:
        print(f"LLM judge enabled ({judge_model}) - doubles the API calls made.\n")
    else:
        print()

    rows = []
    for model_id in models:
        print(f"\n=== {model_id} ===")
        for i, example in enumerate(dataset):
            prompt = PROMPT_TEMPLATE.format(
                context=example["context"], question=example["question"]
            )
            answer, error = call_model(
                model_id, prompt, headers, max_tokens=60, temperature=0.0
            )
            if error:
                print(f"  [{i+1}/{len(dataset)}] FAIL -> {error}")
                rows.append({
                    "model": model_id, "example_id": example["id"],
                    "exact_match": None, "f1": None, "judge_correct": None, "error": error,
                })
                time.sleep(delay)
                continue

            em = metric_max_over_ground_truths(compute_exact, answer, example["answers"])
            f1 = metric_max_over_ground_truths(compute_f1, answer, example["answers"])

            judge_correct = None
            if use_judge:
                judge_correct = judge_norquad_answer(
                    judge_model, headers, example["context"], example["question"],
                    example["answers"], answer,
                )
                time.sleep(delay)

            judge_str = "n/a" if judge_correct is None else str(judge_correct)
            print(f"  [{i+1}/{len(dataset)}] EM={em} f1={f1:.3f} judge={judge_str}")
            rows.append({
                "model": model_id, "example_id": example["id"],
                "exact_match": em, "f1": f1, "judge_correct": judge_correct, "error": None,
            })
            time.sleep(delay)

    df = pd.DataFrame(rows)
    scored = df.dropna(subset=["f1"]).copy()
    summary_table = (
        scored.groupby("model")[["exact_match", "f1"]]
        .mean()
        .sort_values("f1", ascending=False)
    )
    if use_judge:
        judge_scored = scored.dropna(subset=["judge_correct"])
        summary_table["judge_accuracy"] = judge_scored.groupby("model")["judge_correct"].mean()
        judge_scored = judge_scored.assign(
            agrees_with_em=judge_scored["judge_correct"] == judge_scored["exact_match"].astype(bool)
        )
        summary_table["em_judge_agreement"] = judge_scored.groupby("model")["agrees_with_em"].mean()

    fail_counts = df[df["error"].notna()].groupby("model").size()

    print("\n" + "=" * 60)
    print("RESULTS (mean over successfully scored examples)")
    print("=" * 60)
    print(summary_table.round(3).to_string())
    if not fail_counts.empty:
        print("\nFailures per model:")
        print(fail_counts.to_string())

    return df


if __name__ == "__main__":
    run()
