#!/usr/bin/env python3
"""
Benchmark OpenRouter free-tier models on NorMedQA (Norwegian medical
question answering, Riegler 2025, SimulaMet/NorMedQA on Hugging Face).

- Loads NorMedQA (single "train" split, 1401 Norwegian medical exam
  questions) from Hugging Face.
- Each question ships with one correct answer and a set of "wrong_answers"
  distractors but no supporting passage, so this is reframed as closed-book
  multiple-choice QA rather than free-text generation: the model is shown
  the question plus all of its answer options (correct + distractors, shuffled)
  and instructed to pick exactly one lettered option. This keeps the prompt
  grounded in the fixed answer set given in the dataset rather than letting
  the model hallucinate an unconstrained medical answer.
- Scores accuracy: does the model's chosen letter match the annotated
  correct answer.
- Reports per-model averages.

Usage (command line):
    export OPENROUTER_API_KEY="sk-or-..."
    pip install datasets requests pandas --break-system-packages
    python benchmark_normedqa.py

Usage (Colab):
    from google.colab import userdata
    OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
    from benchmark_normedqa import run
    results_df = run(api_key=OPENROUTER_API_KEY, n_examples=20)
"""

import random
import re
import string
import sys
import time

from openrouter_client import get_api_key, call_model, build_headers

# Models that returned valid completions in the last free-tier connectivity test.
DEFAULT_MODELS = [
    "cohere/north-mini-code:free"
]

# Grounded, closed-set multiple-choice prompt in the style of NorEval's
# multiple-choice QA prompts (e.g. Belebele, NorOpenBookQA - Mikhailov et al.
# 2025, Appendix A): the model must select strictly from the given options.
PROMPT_TEMPLATE = (
    "Spørsmål: {question}\n\n"
    "Svaralternativer:\n{options}\n\n"
    "Velg det ene alternativet som er riktig. Svar med kun bokstaven "
    "({letters}), ingen annen tekst.\n\n"
    "Svar:"
)


def load_normedqa(n_examples: int = None, seed: int = 42):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: pip install datasets --break-system-packages")

    ds = load_dataset("SimulaMet/NorMedQA", split="train")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    if n_examples is not None and n_examples < len(indices):
        indices = rng.sample(indices, n_examples)

    examples = []
    for idx in indices:
        row = ds[idx]
        correct = row["answer"].strip()
        distractors = [w.strip() for w in row["wrong_answers"].split(";") if w.strip()]
        options = [correct] + distractors

        shuffled = options[:]
        rng.shuffle(shuffled)
        letters = string.ascii_uppercase[: len(shuffled)]
        correct_letter = letters[shuffled.index(correct)]

        examples.append({
            "question": row["question"].strip(),
            "options": shuffled,
            "letters": letters,
            "correct_letter": correct_letter,
        })
    return examples


def format_prompt(example: dict) -> str:
    options_block = "\n".join(
        f"{letter}: {opt}" for letter, opt in zip(example["letters"], example["options"])
    )
    letters_str = ", ".join(example["letters"])
    return PROMPT_TEMPLATE.format(
        question=example["question"], options=options_block, letters=letters_str
    )


def parse_prediction(response: str, example: dict):
    resp = response.strip()
    match = re.search(r"\b([A-Z])\b", resp.upper())
    if match and match.group(1) in example["letters"]:
        return match.group(1)

    resp_lower = resp.lower()
    for letter, opt in zip(example["letters"], example["options"]):
        if opt.lower() in resp_lower:
            return letter
    return None


def run(models=None, n_examples=20, delay=2.0, api_key=None):
    """
    Run the NorMedQA benchmark against a list of OpenRouter free models.

    Usage:
        from benchmark_normedqa import run
        results_df = run(api_key=OPENROUTER_API_KEY, n_examples=20)
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Missing dependency: pip install pandas --break-system-packages")

    api_key = get_api_key(explicit_key=api_key)
    headers = build_headers(api_key)

    models = models or DEFAULT_MODELS
    print(f"Loading {n_examples} NorMedQA examples...")
    dataset = load_normedqa(n_examples)
    print(f"Loaded {len(dataset)} examples. Testing {len(models)} model(s).\n")

    rows = []
    for model_id in models:
        print(f"\n=== {model_id} ===")
        for i, example in enumerate(dataset):
            prompt = format_prompt(example)
            response, error = call_model(
                model_id, prompt, headers, max_tokens=10, temperature=0.0
            )
            if error:
                print(f"  [{i+1}/{len(dataset)}] FAIL -> {error}")
                rows.append({
                    "model": model_id, "example_idx": i,
                    "correct": None, "parsed": None, "error": error,
                })
                time.sleep(delay)
                continue

            predicted_letter = parse_prediction(response, example)
            correct = (predicted_letter == example["correct_letter"])
            print(
                f"  [{i+1}/{len(dataset)}] predicted={predicted_letter} "
                f"correct={example['correct_letter']} -> {'OK' if correct else 'WRONG'}"
            )
            rows.append({
                "model": model_id, "example_idx": i,
                "correct": correct, "parsed": predicted_letter is not None, "error": None,
            })
            time.sleep(delay)

    df = pd.DataFrame(rows)
    summary_table = (
        df.dropna(subset=["correct"])
        .groupby("model")[["correct", "parsed"]]
        .mean()
        .rename(columns={"correct": "accuracy", "parsed": "parse_rate"})
        .sort_values("accuracy", ascending=False)
    )
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
