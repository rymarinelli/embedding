"""Generate NorSumm summaries with SOTA models via the OpenRouter chat completions API.

Requires OPENROUTER_API_KEY in the environment (never hardcode the key here or
commit it — this script only reads it from os.environ).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_summaries_openrouter.py
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from prompts import build_messages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_summaries"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# id -> OpenRouter model slug. Includes borealis-27b's own Gemma-3-27B base
# model as a useful reference point even though Borealis itself isn't hosted
# on OpenRouter.
MODELS = {
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemma-3-27b-it-base": "google/gemma-3-27b-it",
    "qwen3.6-27b": "qwen/qwen3.6-27b",
    "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
}

MAX_TOKENS = 400
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
        "temperature": 0.2,
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
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

    articles = json.loads((DATA_DIR / "norsumm_test.json").read_text(encoding="utf-8"))
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
        for i, art in enumerate(articles):
            try:
                summary = call_openrouter(api_key, slug, build_messages(art["article"]))
            except RuntimeError as e:
                print(f"  [{i}] ERROR: {e}")
                summary = ""
            outputs.append({"id": art["id"], "summary": summary})
            print(f"  [{i+1}/{len(articles)}] {art['id']} -> {len(summary)} chars")

        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
