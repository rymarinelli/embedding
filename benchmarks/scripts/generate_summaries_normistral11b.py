"""Generate NorSumm summaries with norallm/normistral-11b-thinking (GGUF, CPU).

This model's chat template uses llama.cpp/minja-specific syntax that
llama-cpp-python's plain Jinja2 renderer can't parse (raises
TemplateSyntaxError), so create_chat_completion()'s auto-templating doesn't
work here. Its template is simple enough to reproduce by hand:

    <s><system_prompt> {system}</system_prompt><instruction> {user}</instruction>

The model is a "thinking" model — during training, the assistant turn is
`<think> {reasoning}</think> {content}</s>`. At inference, the model doesn't
spontaneously open a <think> block on its own (tested: it just emits bare
reasoning-style prose and burns the whole token budget without ever
producing an answer) — the opening tag has to be part of the prompt to
signal that turn, same as most "thinking" chat models. We append " <think>"
to the prompt and then take everything after the closing </think> as the
summary (mirrors how the OpenRouter reasoning-model responses separate
`reasoning` from `content`).
"""
import json
import time
from pathlib import Path

import llama_cpp.llama_chat_format as _lcf
from llama_cpp import Llama

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, truncate_article

# llama-cpp-python's Llama.__init__ unconditionally eagerly parses every chat
# template embedded in the GGUF's metadata into a Jinja2ChatFormatter, even
# when chat_format is set explicitly and even though we never call
# create_chat_completion() (we build prompts by hand and use
# create_completion() instead). This model's embedded template uses
# llama.cpp/minja-specific syntax that plain Jinja2 can't compile, which
# would otherwise crash the constructor. Since we never use these chat
# handlers, make a broken template parse fall back to a harmless no-op
# instead of raising.
_orig_jinja_init = _lcf.Jinja2ChatFormatter.__init__


def _tolerant_jinja_init(self, *args, **kwargs):
    try:
        _orig_jinja_init(self, *args, **kwargs)
    except Exception:
        kwargs["template"] = "{{ messages[-1]['content'] }}"
        _orig_jinja_init(self, *args, **kwargs)


_lcf.Jinja2ChatFormatter.__init__ = _tolerant_jinja_init

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_summaries"

MODEL_PATH = "/home/user/embedding/benchmarks/models/normistral-11B-thinking-Q4_K_M.gguf"
OUT_NAME = "normistral-11b-thinking"
N_CTX = 4096
N_THREADS = 4
MAX_TOKENS = 1200

def build_prompt(article: str) -> str:
    user = USER_PROMPT_TEMPLATE.format(article=truncate_article(article))
    return (
        "<s><system_prompt> " + SYSTEM_PROMPT.strip() + "</system_prompt>"
        "<instruction> " + user.strip() + "</instruction> <think>"
    )


def extract_answer(raw: str) -> str:
    if "</think>" in raw:
        return raw.split("</think>", 1)[1].strip()
    # Ran out of max_tokens before closing </think> — no usable answer.
    return ""


def main():
    articles = json.loads((DATA_DIR / "norsumm_test.json").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{OUT_NAME}.json"

    print(f"Loading {MODEL_PATH} (n_ctx={N_CTX}, n_threads={N_THREADS})...")
    t0 = time.time()
    # chat_format="chatml" is a dummy: create_completion() below uses the raw
    # `prompt` string directly and never touches the chat formatter, but
    # passing something other than None stops llama-cpp-python from eagerly
    # parsing the GGUF's embedded chat template (which uses minja-specific
    # syntax plain Jinja2 can't compile, raising TemplateSyntaxError).
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, chat_format="chatml", verbose=False)
    print(f"Loaded in {time.time()-t0:.1f}s")

    outputs = []
    if out_path.exists():
        outputs = json.loads(out_path.read_text(encoding="utf-8"))
    done_ids = {o["id"] for o in outputs}

    for i, art in enumerate(articles):
        if art["id"] in done_ids:
            continue
        t0 = time.time()
        prompt = build_prompt(art["article"])
        try:
            resp = llm.create_completion(
                prompt=prompt, max_tokens=MAX_TOKENS, temperature=0.2, stop=["</s>"],
            )
            raw = (resp["choices"][0]["text"] or "").strip()
            summary = extract_answer(raw)
            if not summary:
                print(f"  [{i}] WARNING: no closing </think> within {MAX_TOKENS} tokens")
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
