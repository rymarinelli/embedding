#!/usr/bin/env python3
"""
Benchmark OpenRouter free-tier models on NorSumm (Norwegian abstractive
news summarization, Touileb et al. 2025, NoDaLiDa).

- Loads the NorSumm test split (Bokmal / "nb" config) from Hugging Face.
- Runs a Norwegian summarization prompt (NorEval Prompt B, Mikhailov et al.
  2025) against a list of OpenRouter models. The prompt is grounded: it asks
  the model to summarize the given article rather than write about the topic
  in general, and the article is the only source of information provided.
- Scores each generated summary against all 3 human references with
  ROUGE-L and BERTScore, taking the max over references per example
  (matching the scoring approach used in the NorSumm/NorEval papers).
- Reports per-model averages.

Usage (command line):
    export OPENROUTER_API_KEY="sk-or-..."
    pip install datasets rouge-score bert-score requests --break-system-packages
    python benchmark_norsumm.py

Usage (Colab):
    from google.colab import userdata
    OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
    !pip install -q datasets rouge-score bert-score
    from benchmark_norsumm import run
    results_df = run(api_key=OPENROUTER_API_KEY, n_articles=5)

Notes:
- BERTScore downloads a multilingual BERT model (~700MB) on first run.
- The default model list below is the set that returned valid completions
  in a prior test_openrouter_free.py run - the free roster rotates, so
  swap in your own list via run(models=[...]) if these have changed.
"""

import sys
import time

from openrouter_client import get_api_key, call_model, build_headers

# Models that returned valid completions in the last free-tier connectivity test.
# Excludes models that hit persistent upstream 429s that session.
DEFAULT_MODELS = [
    "cohere/north-mini-code:free"
]

# NorEval Prompt B (Mikhailov et al. 2025, Appendix A) for NorSumm (Bokmal).
PROMPT_TEMPLATE = (
    "Oppsummer følgende artikkel med noen få setninger:\n\n{article}\n\nOppsummering:"
)


def load_norsumm(n_articles: int):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: pip install datasets --break-system-packages")

    ds = load_dataset("SamiaT/NorSumm", "nb", split="test")
    n = min(n_articles, len(ds))
    return ds.select(range(n))


def score_summary(candidate: str, references: list):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = max(scorer.score(ref, candidate)["rougeL"].fmeasure for ref in references)

    import bert_score

    _, _, f1 = bert_score.score(
        [candidate] * len(references),
        references,
        model_type="bert-base-multilingual-cased",
        verbose=False,
    )
    bertscore_f1 = float(f1.max())

    return rouge_l, bertscore_f1


def run(models=None, n_articles=5, delay=2.0, api_key=None):
    """
    Run the NorSumm benchmark against a list of OpenRouter free models.

    Usage:
        from benchmark_norsumm import run
        results_df = run(api_key=OPENROUTER_API_KEY, n_articles=5)
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Missing dependency: pip install pandas --break-system-packages")

    api_key = get_api_key(explicit_key=api_key)
    headers = build_headers(api_key)

    models = models or DEFAULT_MODELS
    print(f"Loading {n_articles} NorSumm test articles...")
    dataset = load_norsumm(n_articles)
    print(f"Loaded {len(dataset)} articles. Testing {len(models)} model(s).\n")

    print("Loading BERTScore model (bert-base-multilingual-cased, ~700MB on first run)...")
    # Trigger the download/load once up front rather than silently on first score_summary call.
    import bert_score  # noqa: F401

    rows = []
    for model_id in models:
        print(f"\n=== {model_id} ===")
        for i, example in enumerate(dataset):
            article = example["article"]
            references = example["summaries"]

            prompt = PROMPT_TEMPLATE.format(article=article)
            summary, error = call_model(model_id, prompt, headers, max_tokens=300)
            if error:
                print(f"  [{i+1}/{len(dataset)}] FAIL -> {error}")
                rows.append({
                    "model": model_id, "article_idx": i,
                    "rougeL": None, "bertscore_f1": None, "error": error,
                })
                time.sleep(delay)
                continue

            rouge_l, bertscore_f1 = score_summary(summary, references)
            print(f"  [{i+1}/{len(dataset)}] rougeL={rouge_l:.3f} bertscore={bertscore_f1:.3f}")
            rows.append({
                "model": model_id, "article_idx": i,
                "rougeL": rouge_l, "bertscore_f1": bertscore_f1, "error": None,
            })
            time.sleep(delay)

    df = pd.DataFrame(rows)
    summary_table = (
        df.dropna(subset=["rougeL"])
        .groupby("model")[["rougeL", "bertscore_f1"]]
        .mean()
        .sort_values("bertscore_f1", ascending=False)
    )
    fail_counts = df[df["error"].notna()].groupby("model").size()

    print("\n" + "=" * 60)
    print("RESULTS (mean over successfully scored articles)")
    print("=" * 60)
    print(summary_table.round(3).to_string())
    if not fail_counts.empty:
        print("\nFailures per model:")
        print(fail_counts.to_string())

    return df


if __name__ == "__main__":
    run()
