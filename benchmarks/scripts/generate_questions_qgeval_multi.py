"""Answer-conditioned question generation for the QGEval benchmark, across the
full model roster — so the seven dimensions (Fluency, Clarity, Conciseness,
Relevance, Consistency, Answerability, Answer Consistency) can be compared
between models, not just between Fable and NorQuAD's human references.

Uses the identical prompt and decoding as generate_questions_qgeval_fable.py,
which is what makes the resulting scores comparable with the Fable run already
in results/qgeval_fable_questions.json.

Requires OPENROUTER_API_KEY in the environment (never hardcode/commit it).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_questions_qgeval_multi.py                 # whole roster
    python3 generate_questions_qgeval_multi.py gpt-5.6-sol     # named models only
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "qgeval_questions"
SAMPLE_PATH = DATA_DIR / "qgeval_sample.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Same roster as the QA and NorSumm benchmarks, so the three tables cover the
# same models. Fable is included for completeness even though its questions
# already exist — re-running it is harmless and keeps one code path.
MODELS = {
    "claude-fable-5": "anthropic/claude-fable-5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemma-3-27b-it-base": "google/gemma-3-27b-it",
    "qwen3.6-27b": "qwen/qwen3.6-27b",
    "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
}

MAX_PASSAGE_CHARS = 6000
MAX_TOKENS = 200
TIMEOUT = 90
RETRIES = 3

# Verbatim from generate_questions_qgeval_fable.py — any drift here breaks
# comparability with the Fable run.
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


def clean_question(text):
    t = (text or "").strip()
    for prefix in ("Spørsmål:", "Question:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t.split("\n")[0].strip()


def call_model(api_key, model_path, answer, passage):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_path,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                answer=answer, passage=passage[:MAX_PASSAGE_CHARS])},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content:
                    return content.strip()
                # Reasoning models can spend the whole budget on hidden CoT and
                # return null content; that is retryable, not a crash.
                last_err = "empty content"
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts: {last_err}")


def run_model(api_key, label, model_path, samples):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.json"
    outputs = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {o["qid"] for o in outputs if o.get("generated_question")}
    if len(done) >= len(samples):
        print(f"  {label}: already complete ({len(done)})")
        return

    print(f"  {label}: {len(done)}/{len(samples)} done, continuing")
    for s in samples:
        if s["qid"] in done:
            continue
        q, err = "", None
        try:
            q = clean_question(call_model(api_key, model_path, s["answer"], s["passage"]))
            if not q:
                err = "empty after cleaning"
        except Exception as e:
            err = str(e)
        outputs.append({
            "qid": s["qid"], "doc_id": s["doc_id"], "answer": s["answer"],
            "reference_question": s["reference_question"], "generated_question": q,
            **({"error": err} if err else {}),
        })
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for o in outputs if o.get("generated_question") and not o.get("error"))
    print(f"  {label}: {n_ok}/{len(samples)} generated -> {out_path}")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    only = set(sys.argv[1:])
    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"{len(samples)} passages; roster: {', '.join(MODELS)}")

    for label, path in MODELS.items():
        if only and label not in only:
            continue
        try:
            run_model(api_key, label, path, samples)
        except Exception as e:
            print(f"  {label}: ABORTED — {e}")


if __name__ == "__main__":
    main()
