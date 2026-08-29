"""Answer-conditioned QGEval question generation for a locally-hosted GGUF
model — for Norwegian models that aren't on OpenRouter (normistral-7b-warm-
instruct, or any other GGUF checkpoint you point it at).

Identical prompt and decoding intent to generate_questions_qgeval_fable.py /
generate_questions_qgeval_multi.py, so scores are comparable across all three
generation paths. Output lands in results/qgeval_questions/<label>.json, the
same directory score_qgeval_dimensions_multi.py scans — it globs every file
there, so this needs no registration anywhere else.

Requires llama-cpp-python and a local .gguf file. If llama-cpp-python isn't
built yet, see generate_summaries_local_gguf.py's header for the Firecracker
AVX512 workaround this repo has needed before.

Usage:
    python3 generate_questions_qgeval_local_gguf.py <gguf_path> <output_label> \\
        [n_ctx] [n_threads] [max_tokens]

Example:
    python3 generate_questions_qgeval_local_gguf.py \\
        models/normistral-7b-warm-instruct-Q8_0.gguf normistral-7b-warm-instruct
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "qgeval_questions"
SAMPLE_PATH = DATA_DIR / "qgeval_sample.json"

# Verbatim from generate_questions_qgeval_fable.py / _multi.py — any drift
# here breaks comparability with those runs.
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
MAX_PASSAGE_CHARS = 6000


def clean_question(text):
    t = (text or "").strip()
    for prefix in ("Spørsmål:", "Question:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t.split("\n")[0].strip()


def main():
    if len(sys.argv) < 3:
        print("usage: generate_questions_qgeval_local_gguf.py <gguf_path> <output_label> "
              "[n_ctx] [n_threads] [max_tokens]")
        sys.exit(1)
    gguf_path = sys.argv[1]
    label = sys.argv[2]
    n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 64

    from llama_cpp import Llama
    print(f"Loading {gguf_path} (n_ctx={n_ctx}, n_threads={n_threads})...")
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(samples)} QGEval samples")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.json"

    outputs = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {o["qid"] for o in outputs if o.get("generated_question") and not o.get("error")}

    for i, s in enumerate(samples):
        if s["qid"] in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                answer=s["answer"], passage=s["passage"][:MAX_PASSAGE_CHARS])},
        ]
        question, err = "", None
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                              temperature=0.0)
            content = resp["choices"][0]["message"]["content"]
            question = clean_question(content)
            if not question:
                err = "empty after cleaning"
        except Exception as e:
            err = str(e)

        outputs.append({
            "qid": s["qid"], "doc_id": s["doc_id"], "answer": s["answer"],
            "reference_question": s["reference_question"], "generated_question": question,
            **({"error": err} if err else {}),
        })
        flag = " [ERROR]" if err else ""
        print(f"  [{i+1}/{len(samples)}] qid={s['qid']}{flag} -> {question[:80]!r}")
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for o in outputs if o.get("generated_question") and not o.get("error"))
    print(f"\nSaved {len(outputs)} records ({n_ok} ok) to {out_path}")


if __name__ == "__main__":
    main()
