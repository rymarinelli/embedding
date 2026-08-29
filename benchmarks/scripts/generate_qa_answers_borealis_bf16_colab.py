"""NorQuAD extractive-QA answers from NbAiLab/borealis-27b at full bf16
precision (no quantization), for Google Colab.

Companion to generate_summaries_borealis_bf16_colab.py. That script covers the
NorSumm summarization benchmark; this one covers the NorQuAD QA benchmark
(300 questions, Exact Match / token-F1) so Borealis can appear in
results/norquad_qa_results.csv alongside the API models.

Loading the model costs ~51 GiB of download and several minutes, so
`main(model=..., processor=...)` accepts an already-loaded model — run the
summarization script first in the same Colab session and hand its model over
rather than loading twice.

DECODING
--------
Matched to generate_qa_answers_local_gguf.py, which produced the other local
rows: temperature 0.0 (greedy), max_tokens 64, repeat_penalty 1.0. QA is a
short-span extraction task, so greedy is the right default here — note this
differs from the summarization run, which matched its own baseline at
temperature 0.2.

Prompts come from qa_prompts.py unchanged, so the EM/F1 numbers are directly
comparable with every other model in the table.

USAGE (Colab)
-------------
    import generate_qa_answers_borealis_bf16_colab as qa
    qa.main(model=run.model, processor=run.processor)   # reuse the loaded model
    # or qa.main()                                       # load it itself

Model follows generate_summaries_borealis_bf16_colab.MODEL_ID, so setting it
once covers both benchmarks in a session.

Output: <model-name>-bf16-full.json in the {question, gold_answer,
predicted_answer} shape score_qa.py expects.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

# qa_prompts.py sits next to this file in the repo checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

# Follows whatever the summarization module is set to, so both benchmarks in one
# Colab session run the same model without setting it twice.
import generate_summaries_borealis_bf16_colab as _summ  # noqa: E402
MODEL_ID = _summ.MODEL_ID

# Matched to generate_qa_answers_local_gguf.py.
MAX_NEW_TOKENS = 64
REPETITION_PENALTY = 1.0

def derive_paths(model_id):
    """Own subdirectory — the summarization run writes the same filename, since
    both tables want the same model label."""
    label = model_id.split("/")[-1] + "-bf16-full"
    name = label + ".json"
    if os.path.isdir("/content/drive/MyDrive"):
        d = "/content/drive/MyDrive/borealis_bench/qa"
        os.makedirs(d, exist_ok=True)
        return label, name, os.path.join(d, name)
    return label, name, name


RUN_LABEL, OUT_NAME, OUT_PATH = None, None, None

# The 300-question sample lives in the repo checkout, not on HF.
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "norquad_qa_sample.json"


def clean_answer(text):
    """The prompt asks for the bare span; strip any label or quoting anyway."""
    t = (text or "").strip()
    for prefix in ("Svar:", "Answer:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t.split("\n")[0].strip()


def main(model=None, processor=None):
    # Pick up any MODEL_ID the caller set on the summarization module.
    global MODEL_ID, RUN_LABEL, OUT_NAME, OUT_PATH
    MODEL_ID = _summ.MODEL_ID
    RUN_LABEL, OUT_NAME, OUT_PATH = derive_paths(MODEL_ID)
    print(f"Model: {MODEL_ID}")

    if model is None or processor is None:
        # Reuse the loader (and its precision check) from the summarization script.
        _summ.preflight()
        processor, model = _summ.load_model()
    else:
        print("Reusing the already-loaded model — no second multi-GiB download.")
        # Still prove it is unquantized, in case a different model was passed in.
        _summ.verify_unquantized(model)

    if not SAMPLE_PATH.exists():
        raise SystemExit(
            f"Missing {SAMPLE_PATH}.\n"
            "Clone the repo first — the 300-question sample is versioned there:\n"
            "  !git clone --branch claude/norquad-norsumm-benchmarks-9qhyke \\\n"
            "      https://github.com/rymarinelli/embedding.git /content/embedding"
        )
    examples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(examples)} NorQuAD questions")

    outputs = []
    if os.path.exists(OUT_PATH):
        outputs = json.loads(Path(OUT_PATH).read_text(encoding="utf-8"))
        print(f"Resuming: {len(outputs)} answers already done.")
    done = {o["question"] for o in outputs}

    t0 = time.time()
    for i, ex in enumerate(examples):
        if ex["question"] in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                context=ex["context"], question=ex["question"])},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        try:
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,          # greedy, matching the GGUF QA run
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=REPETITION_PENALTY,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
            answer = clean_answer(processor.decode(new_tokens, skip_special_tokens=True))
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            answer = ""

        outputs.append({
            "question": ex["question"],
            "gold_answer": ex["gold_answer"],
            "predicted_answer": answer,
        })
        # Checkpoint every question — this is a long run on offloaded weights.
        Path(OUT_PATH).write_text(
            json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        if (i + 1) % 10 == 0 or i < 3:
            el = time.time() - t0
            print(f"  [{len(outputs)}/{len(examples)}] {el/60:.1f} min elapsed  "
                  f"pred={answer!r} gold={ex['gold_answer']!r}")

    n_empty = sum(1 for o in outputs if not o["predicted_answer"].strip())
    print(f"\nDone. {len(outputs)} answers ({n_empty} empty) -> {OUT_PATH}")
    print("Place it in benchmarks/results/qa_answers/ and run:")
    print("    python3 benchmarks/scripts/score_qa.py")


if __name__ == "__main__":
    main()
