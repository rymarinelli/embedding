"""Generate one higher-level Norwegian comprehension question per NorQuAD
news passage, using Claude Fable 5 via OpenRouter — same methodology as
generate_questions_amedia_fable.py, pointed at benchmarks/data/corpus.jsonl
(NorQuAD's own passages) instead of the Amedia corpus.

Purpose: NorQuAD's own questions are short factoid lookups (see
norquad_lexical_comparison.py for the measured contrast). This script lets
us compare what Fable naturally asks about the same passages against what
NorQuAD's human annotators actually asked, using the same prompt design
validated on the Amedia benchmark.

Requires OPENROUTER_API_KEY in the environment (never hardcode/commit it).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_questions_norquad_fable.py [n_passages]
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_PATH = RESULTS_DIR / "norquad_fable_questions.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-fable-5"

MAX_TEXT_CHARS = 6000
MAX_TOKENS = 600
TIMEOUT = 60
RETRIES = 3

SYSTEM_PROMPT = (
    "Du er en norsk redaksjonell ekspert som lager spørsmål til en "
    "leseforståelses- og gjenfinningsbenchmark. Du svarer alltid på norsk, "
    "og du svarer utelukkende med gyldig JSON."
)

USER_TEMPLATE = (
    "Les nyhetsteksten nedenfor. Lag ETT spørsmål på norsk som tester dyp "
    "forståelse av teksten.\n\n"
    "Krav til spørsmålet:\n"
    "- Det skal være et høynivåspørsmål som krever forståelse av sammenhenger, "
    "årsaker, konsekvenser, motiver, hovedbudskap eller implikasjoner — ikke et "
    "overfladisk faktaspørsmål med ett-ords svar.\n"
    "- Det skal kunne besvares ut fra innholdet i teksten.\n"
    "- Det skal være formulert slik at nettopp denne teksten er den beste "
    "kilden til å besvare det.\n\n"
    "Legg også ved en begrunnelse på norsk som forklarer hvorfor akkurat denne "
    "teksten er det beste treffet for spørsmålet — altså hvilket innhold i "
    "teksten som gjør den til den unike og beste kilden.\n\n"
    "Svar KUN med gyldig JSON på nøyaktig dette formatet, uten noen annen tekst:\n"
    '{{"spørsmål": "...", "begrunnelse": "..."}}\n\n'
    "Tekst:\n{text}"
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json_reply(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    return json.loads(t)


def call_fable(api_key, text):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user = USER_TEMPLATE.format(text=text[:MAX_TEXT_CHARS])
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.5,
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content:
                    return content
                last_err = "empty content"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Fable failed after {RETRIES} attempts: {last_err}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    passages = load_jsonl(DATA_DIR / "corpus.jsonl")
    if n:
        passages = passages[:n]
    print(f"Loaded {len(passages)} NorQuAD passages")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    if OUT_PATH.exists():
        outputs = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    done = {o["doc_id"] for o in outputs}

    for i, p in enumerate(passages):
        if p["doc_id"] in done:
            continue
        question, explanation, err = "", "", None
        try:
            raw = call_fable(api_key, p["text"])
            parsed = parse_json_reply(raw)
            question = (parsed.get("spørsmål") or parsed.get("sporsmal") or "").strip()
            explanation = (parsed.get("begrunnelse") or "").strip()
            if not question:
                err = f"no question parsed from: {raw[:200]}"
        except Exception as e:
            err = str(e)

        outputs.append({
            "doc_id": p["doc_id"],
            "spørsmål": question,
            "begrunnelse": explanation,
            **({"error": err} if err else {}),
        })
        flag = " [ERROR]" if err else ""
        print(f"  [{i+1}/{len(passages)}] doc_id={p['doc_id']}{flag} -> {question[:80]!r}")
        OUT_PATH.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for o in outputs if o.get("spørsmål") and not o.get("error"))
    print(f"\nSaved {len(outputs)} records ({n_ok} with a question) to {OUT_PATH}")


if __name__ == "__main__":
    main()
