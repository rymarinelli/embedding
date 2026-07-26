"""Generate NorSumm summaries with a local GGUF model via llama.cpp (CPU).

Used for Norwegian-native LLMs that aren't available on OpenRouter or HF's
Inference Router (NorMistral, Borealis-27b). Uses the exact same prompt
template and article truncation as generate_summaries_openrouter.py so
results are comparable.

Usage:
    python3 generate_summaries_local_gguf.py <gguf_path> <output_name> [n_ctx] [n_threads] [max_tokens]
"""
import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from prompts import build_messages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_summaries"


def main():
    if len(sys.argv) < 3:
        print("usage: generate_summaries_local_gguf.py <gguf_path> <output_name> [n_ctx] [n_threads] [max_tokens]")
        sys.exit(1)
    gguf_path = sys.argv[1]
    out_name = sys.argv[2]
    n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    # "Thinking"/reasoning GGUF models (e.g. normistral-11b-thinking) spend part
    # of the completion budget on hidden chain-of-thought before the answer,
    # same issue as the OpenRouter reasoning models — give them more room.
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 500

    articles = json.loads((DATA_DIR / "norsumm_test.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_name}.json"

    print(f"Loading {gguf_path} (n_ctx={n_ctx}, n_threads={n_threads})...")
    t0 = time.time()
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    print(f"Loaded in {time.time()-t0:.1f}s")

    outputs = []
    if out_path.exists():
        outputs = json.loads(out_path.read_text(encoding="utf-8"))
    done_ids = {o["id"] for o in outputs}

    for i, art in enumerate(articles):
        if art["id"] in done_ids:
            continue
        t0 = time.time()
        messages = build_messages(art["article"])
        try:
            resp = llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
            summary = (content or "").strip()
            if not summary:
                print(f"  [{i}] WARNING: empty content (finish_reason={resp['choices'][0].get('finish_reason')})")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            summary = ""
        outputs.append({"id": art["id"], "summary": summary})
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(articles)}] {art['id']} -> {len(summary)} chars ({elapsed:.1f}s)")
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
