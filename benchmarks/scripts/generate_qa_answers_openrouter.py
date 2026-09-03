"""Answer NorQuAD news questions with SOTA models via OpenRouter, using the
exact same prompt/sample/scoring as generate_qa_answers_anthropic.py so
Claude Opus 5 / Fable 5 are directly comparable to these models in one table.

Requires OPENROUTER_API_KEY in the environment (never hardcode/commit it).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_qa_answers_openrouter.py [model_name ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from qa_prompts import build_messages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "qa_answers"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Same roster as the NorSumm OpenRouter comparison, so the QA and
# summarization benchmarks cover the same set of models.
MODELS = {
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-fable-5-1": "anthropic/claude-fable-5-1",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemma-3-27b-it-base": "google/gemma-3-27b-it",
    "qwen3.6-27b": "qwen/qwen3.6-27b",
    "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
}

# Some of these are reasoning models that spend completion tokens on hidden
# chain-of-thought before the final answer (same issue hit in the NorSumm
# OpenRouter generation) — keep a generous budget so short QA answers still
# survive that.
MAX_TOKENS = 800
TIMEOUT = 60
RETRIES = 3


def call_openrouter(api_key, model_slug, messages):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_slug,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    finish_reason = data["choices"][0].get("finish_reason")
                    last_err = f"null content (finish_reason={finish_reason})"
                    time.sleep(2 ** attempt)
                    continue
                return content.strip()
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{model_slug} failed after {RETRIES} attempts: {last_err}")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    examples = json.loads((DATA_DIR / "norquad_qa_sample.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for name in only:
        slug = MODELS[name]
        out_path = OUT_DIR / f"{name}.json"
        if out_path.exists():
            print(f"skip {name} (already exists)")
            continue

        print(f"=== {name} ({slug}) ===")
        outputs = []
        for i, ex in enumerate(examples):
            try:
                answer = call_openrouter(api_key, slug, build_messages(ex["context"], ex["question"]))
            except RuntimeError as e:
                print(f"  [{i}] ERROR: {e}")
                answer = ""
            outputs.append({
                "qid": ex["qid"],
                "question": ex["question"],
                "gold_answer": ex["gold_answer"],
                "predicted_answer": answer,
            })
            print(f"  [{i+1}/{len(examples)}] qid={ex['qid']} pred={answer!r} gold={ex['gold_answer']!r}")
            out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
