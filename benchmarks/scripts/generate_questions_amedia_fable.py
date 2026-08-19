"""Generate one higher-level Norwegian comprehension question per article for
the first N Amedia articles, using Claude Fable 5 via OpenRouter.

For each article Fable produces, in Norwegian:
  - spørsmål:   a higher-level question testing genuine understanding of the
                article (causes, consequences, motives, main point,
                implications) — not a shallow one-word-answer fact lookup —
                phrased so this article is the best source to answer it.
  - begrunnelse: an explanation of why precisely this article is the best
                match / source for that question.

The Amedia corpus is proprietary/paywalled, so this script keeps the raw
article bodies OUT of the committed output: the result file references each
article only by uuid / content_id / url / title alongside the generated
question + explanation. The source JSONL is read locally (never committed).

Requires OPENROUTER_API_KEY in the environment (never hardcode/commit it).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 generate_questions_amedia_fable.py <source_jsonl> [n_articles]
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_PATH = RESULTS_DIR / "amedia_fable_questions.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-fable-5"

MAX_BODY_CHARS = 6000
MAX_TOKENS = 600
TIMEOUT = 60
RETRIES = 3

SYSTEM_PROMPT = (
    "Du er en norsk redaksjonell ekspert som lager spørsmål til en "
    "leseforståelses- og gjenfinningsbenchmark. Du svarer alltid på norsk, "
    "og du svarer utelukkende med gyldig JSON."
)

USER_TEMPLATE = (
    "Les nyhetsartikkelen nedenfor. Lag ETT spørsmål på norsk som tester dyp "
    "forståelse av artikkelen.\n\n"
    "Krav til spørsmålet:\n"
    "- Det skal være et høynivåspørsmål som krever forståelse av sammenhenger, "
    "årsaker, konsekvenser, motiver, hovedbudskap eller implikasjoner — ikke et "
    "overfladisk faktaspørsmål med ett-ords svar.\n"
    "- Det skal kunne besvares ut fra innholdet i artikkelen.\n"
    "- Det skal være formulert slik at nettopp denne artikkelen er den beste "
    "kilden til å besvare det.\n\n"
    "Legg også ved en begrunnelse på norsk som forklarer hvorfor akkurat denne "
    "artikkelen er det beste treffet for spørsmålet — altså hvilket innhold i "
    "artikkelen som gjør den til den unike og beste kilden.\n\n"
    "Svar KUN med gyldig JSON på nøyaktig dette formatet, uten noen annen tekst:\n"
    '{{"spørsmål": "...", "begrunnelse": "..."}}\n\n'
    "Tittel: {title}\n"
    "Ingress: {lead}\n"
    "Brødtekst:\n{body}"
)


def load_first_n(source_path, n):
    rows = []
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def parse_json_reply(text):
    """Fable is asked for pure JSON, but strip fences / surrounding prose just in case."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    return json.loads(t)


def call_fable(api_key, title, lead, body):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user = USER_TEMPLATE.format(title=title, lead=lead, body=body[:MAX_BODY_CHARS])
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
    if len(sys.argv) < 2:
        print("usage: generate_questions_amedia_fable.py <source_jsonl> [n_articles]")
        sys.exit(1)
    source_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY in the environment", file=sys.stderr)
        sys.exit(1)

    articles = load_first_n(source_path, n)
    print(f"Loaded first {len(articles)} articles from {source_path}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    if OUT_PATH.exists():
        outputs = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    done = {o["uuid"] for o in outputs}

    for i, art in enumerate(articles):
        if art["uuid"] in done:
            continue
        title = art.get("title", "")
        lead = art.get("lead_text", "")
        body = art.get("body_text", "")
        question, explanation, err = "", "", None
        try:
            raw = call_fable(api_key, title, lead, body)
            parsed = parse_json_reply(raw)
            question = (parsed.get("spørsmål") or parsed.get("sporsmal") or "").strip()
            explanation = (parsed.get("begrunnelse") or "").strip()
            if not question:
                err = f"no question parsed from: {raw[:200]}"
        except Exception as e:
            err = str(e)

        outputs.append({
            "uuid": art.get("uuid"),
            "content_id": art.get("content_id"),
            "url": art.get("url"),
            "title": title,          # headline metadata only — no article body committed
            "spørsmål": question,
            "begrunnelse": explanation,
            **({"error": err} if err else {}),
        })
        flag = " [ERROR]" if err else ""
        print(f"  [{i+1}/{len(articles)}] {art.get('content_id')}{flag} -> {question[:80]!r}")
        OUT_PATH.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for o in outputs if o.get("spørsmål") and not o.get("error"))
    print(f"\nSaved {len(outputs)} records ({n_ok} with a question) to {OUT_PATH}")


if __name__ == "__main__":
    main()
