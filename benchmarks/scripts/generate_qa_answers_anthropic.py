"""Answer NorQuAD news questions with Claude Opus 5 / Fable 5 via Anthropic's
native Messages API (there's no Anthropic embeddings endpoint, so these
models can't take part in the embedding retrieval benchmark — this scores
them as extractive QA models instead, the standard way NorQuAD/SQuAD-style
datasets are evaluated).

Requires ANTHROPIC_API_KEY in the environment (never hardcode/commit it).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 generate_qa_answers_anthropic.py [model_name ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

from qa_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "qa_answers"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

MODELS = {
    "claude-opus-5": "claude-opus-5",
    "claude-fable-5": "claude-fable-5",
}

MAX_TOKENS = 64
TIMEOUT = 60
RETRIES = 3


def call_anthropic(api_key, model, context, question):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
        ],
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                text_blocks = [b["text"] for b in data["content"] if b.get("type") == "text"]
                return "".join(text_blocks).strip()
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{model} failed after {RETRIES} attempts: {last_err}")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    examples = json.loads((DATA_DIR / "norquad_qa_sample.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for name in only:
        model = MODELS[name]
        out_path = OUT_DIR / f"{name}.json"
        if out_path.exists():
            print(f"skip {name} (already exists)")
            continue

        print(f"=== {name} ({model}) ===")
        outputs = []
        for i, ex in enumerate(examples):
            try:
                answer = call_anthropic(api_key, model, ex["context"], ex["question"])
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
