"""Generate answer-conditioned questions with Claude Fable 5, QGEval-style.

Mirrors the zero-shot QG instruction QGEval (Fu et al., EMNLP 2024) used for
GPT-4-zeroshot (their Table 9):

    "Generate a question based on the given answer and context, the generated
     question must be answered by the given answer.
     Answer: {answer} Context: {passage} Question:"

translated to Norwegian so the generated question is in the language of the
passage. Structurally identical to the paper's prompt — no extra instructions
about question depth or style, unlike generate_questions_norquad_fable.py,
because the point here is to measure QG ability against NorQuAD's own
questions as the reference.

Requires OPENROUTER_API_KEY in the environment (never hardcode/commit it).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_questions_qgeval_fable.py
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SAMPLE_PATH = DATA_DIR / "qgeval_sample.json"
OUT_PATH = RESULTS_DIR / "qgeval_fable_questions.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-fable-5"

MAX_PASSAGE_CHARS = 6000
MAX_TOKENS = 200
TIMEOUT = 60
RETRIES = 3

SYSTEM_PROMPT = (
    "Du er en assistent som lager spørsmål på norsk. Du svarer kun med selve "
    "spørsmålet, uten forklaring eller annen tekst."
)

USER_TEMPLATE = (
    "Lag et spørsmål basert på det oppgitte svaret og konteksten. Det "
    "genererte spørsmålet må kunne besvares med det oppgitte svaret.\n"
    "Svar: {answer}\n"
    "Kontekst: {passage}\n"
    "Spørsmål:"
)


def call_fable(api_key, answer, passage):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user = USER_TEMPLATE.format(answer=answer, passage=passage[:MAX_PASSAGE_CHARS])
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content:
                    return content.strip()
                last_err = "empty content"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Fable failed after {RETRIES} attempts: {last_err}")


def clean_question(text):
    """Model is asked for the bare question; strip any stray label/quotes."""
    t = text.strip()
    for prefix in ("Spørsmål:", "Question:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t.split("\n")[0].strip()


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(samples)} QGEval samples")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    if OUT_PATH.exists():
        outputs = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    done = {o["qid"] for o in outputs if not o.get("error")}

    for i, s in enumerate(samples):
        if s["qid"] in done:
            continue
        question, err = "", None
        try:
            question = clean_question(call_fable(api_key, s["answer"], s["passage"]))
            if not question:
                err = "empty question after cleaning"
        except Exception as e:
            err = str(e)

        outputs.append({
            "qid": s["qid"],
            "doc_id": s["doc_id"],
            "answer": s["answer"],
            "reference_question": s["reference_question"],
            "generated_question": question,
            **({"error": err} if err else {}),
        })
        flag = " [ERROR]" if err else ""
        print(f"  [{i+1}/{len(samples)}] qid={s['qid']}{flag} -> {question[:80]!r}")
        OUT_PATH.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for o in outputs if o.get("generated_question") and not o.get("error"))
    print(f"\nSaved {len(outputs)} records ({n_ok} with a question) to {OUT_PATH}")


if __name__ == "__main__":
    main()
