"""Answer-conditioned QGEval question generation for a Borealis-27b variant at
full bf16 precision, for Google Colab.

Third benchmark alongside generate_summaries_borealis_bf16_colab.py (NorSumm)
and generate_qa_answers_borealis_bf16_colab.py (NorQuAD QA) — all three share
one model load in a Colab session. Same pattern: follows
generate_summaries_borealis_bf16_colab.MODEL_ID, so setting the model once in
that module covers all three benchmarks.

Same prompt as generate_questions_qgeval_fable.py / _multi.py /
_local_gguf.py, so scores are comparable across every generation path.

Output: <model-name>.json, checkpointed to
/content/drive/MyDrive/borealis_bench/qgeval/ if Drive is mounted (else the
local repo checkout). Copy it into results/qgeval_questions/ in the repo —
that's the directory score_qgeval_dimensions_multi.py globs, so no separate
registration is needed once it's there.

Usage (Colab, after loading the model via generate_summaries_borealis_bf16_colab):
    import generate_questions_qgeval_borealis_colab as qg
    qg.main(model=model, processor=processor)
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_summaries_borealis_bf16_colab as _summ  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "qgeval_sample.json"

# Checkpoint to Drive if mounted, matching the norsumm/qa scripts' pattern —
# a disconnect during a 200-passage run should cost nothing, not the whole run.
DRIVE_DIR = "/content/drive/MyDrive/borealis_bench/qgeval"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "results" / "qgeval_questions"

# Verbatim from the other three qgeval generators — any drift breaks
# comparability.
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
MAX_NEW_TOKENS = 64
REPETITION_PENALTY = 1.0


def clean_question(text):
    t = (text or "").strip()
    for prefix in ("Spørsmål:", "Question:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if len(t) > 1 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    return t.split("\n")[0].strip()


def out_path_for(model_id):
    name = f"{model_id.split('/')[-1]}.json"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return Path(DRIVE_DIR) / name
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_DIR / name


def main(model=None, processor=None):
    model_id = _summ.MODEL_ID
    out_path = out_path_for(model_id)
    print(f"Model: {model_id}")
    print(f"Output: {out_path}")

    if model is None or processor is None:
        _summ.preflight()
        processor, model = _summ.load_model()
    else:
        print("Reusing the already-loaded model — no second multi-GiB download.")
        _summ.verify_unquantized(model)

    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(samples)} QGEval samples")

    outputs = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {o["qid"] for o in outputs if o.get("generated_question") and not o.get("error")}

    t0 = time.time()
    for i, s in enumerate(samples):
        if s["qid"] in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                answer=s["answer"], passage=s["passage"][:MAX_PASSAGE_CHARS])},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        question, err = "", None
        try:
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=REPETITION_PENALTY,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
            question = clean_question(processor.decode(new_tokens, skip_special_tokens=True))
            if not question:
                err = "empty after cleaning"
        except Exception as e:
            err = str(e)

        outputs.append({
            "qid": s["qid"], "doc_id": s["doc_id"], "answer": s["answer"],
            "reference_question": s["reference_question"], "generated_question": question,
            **({"error": err} if err else {}),
        })
        out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

        if (i + 1) % 10 == 0 or i < 3:
            el = time.time() - t0
            print(f"  [{len(outputs)}/{len(samples)}] {el/60:.1f} min elapsed "
                  f"-> {question[:80]!r}")

    n_ok = sum(1 for o in outputs if o.get("generated_question") and not o.get("error"))
    print(f"\nDone. {n_ok}/{len(samples)} generated -> {out_path}")
    print("Copy it into results/qgeval_questions/ in the repo and run:")
    print("    python3 benchmarks/scripts/score_qgeval_dimensions_multi.py")


if __name__ == "__main__":
    main()
