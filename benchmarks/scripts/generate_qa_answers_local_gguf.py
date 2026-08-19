"""Answer NorQuAD news questions with a local GGUF model via llama.cpp (CPU),
using the exact same prompt/sample as generate_qa_answers_anthropic.py /
generate_qa_answers_openrouter.py so results land in the same comparable
table.

Usage:
    python3 generate_qa_answers_local_gguf.py <gguf_path> <output_name> [n_ctx] [n_threads] [max_tokens]
"""
import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from qa_prompts import build_messages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "qa_answers"


def main():
    if len(sys.argv) < 3:
        print("usage: generate_qa_answers_local_gguf.py <gguf_path> <output_name> "
              "[n_ctx] [n_threads] [max_tokens] [repeat_penalty]")
        sys.exit(1)
    gguf_path = sys.argv[1]
    out_name = sys.argv[2]
    n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    # 1.0 = off (preserves earlier committed runs); pass e.g. 1.1 on a fairer
    # Borealis re-run to suppress degenerate repetition. See AZURE_BOREALIS.md.
    repeat_penalty = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

    examples = json.loads((DATA_DIR / "norquad_qa_sample.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_name}.json"

    print(f"Loading {gguf_path} (n_ctx={n_ctx}, n_threads={n_threads})...")
    t0 = time.time()
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    print(f"Loaded in {time.time()-t0:.1f}s")

    outputs = []
    if out_path.exists():
        outputs = json.loads(out_path.read_text(encoding="utf-8"))
    done_qids = {o["qid"] for o in outputs}

    for i, ex in enumerate(examples):
        if ex["qid"] in done_qids:
            continue
        t0 = time.time()
        messages = build_messages(ex["context"], ex["question"])
        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                              temperature=0.0, repeat_penalty=repeat_penalty)
            content = resp["choices"][0]["message"]["content"]
            answer = (content or "").strip()
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            answer = ""
        outputs.append({
            "qid": ex["qid"],
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "predicted_answer": answer,
        })
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(examples)}] qid={ex['qid']} pred={answer!r} gold={ex['gold_answer']!r} ({elapsed:.1f}s)")
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
