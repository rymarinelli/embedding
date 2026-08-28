"""Run NbAiLab/borealis-27b at FULL bf16 precision (no quantization) on Google
Colab, to test whether the model's low NorSumm score was a quantization artifact.

WHY THIS IS AWKWARD
-------------------
Borealis-27b is a Gemma3-architecture model whose bf16 weights total 51.1 GiB.
Colab's largest GPU is an A100 with 40 GB, and Colab offers no 80 GB option, so
the weights cannot live on the GPU alone. This script therefore uses accelerate's
CPU offload: every weight stays bf16 (nothing is quantized), but the layers that
do not fit on the GPU are held in system RAM and streamed in as needed.

That is slow — expect roughly 1-3 tokens/sec rather than the 30+ you would get
from a fully resident model. The run is checkpointed after every article so a
Colab disconnect costs one article, not the whole job.

REQUIRED COLAB RUNTIME
----------------------
    Runtime > Change runtime type > A100 GPU  +  High-RAM
Anything less (T4 16 GB, L4 22.5 GB, or standard-RAM) will not complete: the
preflight check below will tell you so before the download starts.

ISOLATING THE VARIABLE
----------------------
The earlier low score was attributed to two things at once: 4-bit quantization
AND a missing repetition penalty. Changing both at the same time cannot tell you
which mattered. REPETITION_PENALTY therefore defaults to 1.0 (off), matching the
original quantized run, so this run isolates the effect of precision alone. Set
it to 1.1 for a second run if you want the combined effect.

USAGE (Colab)
-------------
    !pip install -q -U transformers accelerate huggingface_hub pandas pyarrow
    from huggingface_hub import login; login(token="<your HF token>")
    from google.colab import drive; drive.mount("/content/drive")   # optional but recommended
    # then paste this file and call main()

Output: borealis-27b-bf16.json, in the same {id, summary} shape the other
models produce, so score_summaries.py picks it up unchanged.
"""
import json
import os
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = "NbAiLab/borealis-27b"

# Prompts must match benchmarks/scripts/prompts.py exactly, or this run is not
# comparable with the other models in norsumm_lexical_results.csv.
SYSTEM_PROMPT = (
    "Du er en norsk redaksjonell assistent som skriver korte, presise sammendrag "
    "av nyhetsartikler på bokmål."
)
USER_PROMPT_TEMPLATE = (
    "Skriv et sammendrag av følgende nyhetsartikkel på norsk bokmål. "
    "Sammendraget skal være 3-5 setninger, kun inneholde informasjon som "
    "fremkommer i artikkelen, og ikke inneholde noen innledende setning som "
    "\"Her er et sammendrag\".\n\n"
    "Artikkel:\n{article}\n\nSammendrag:"
)
MAX_ARTICLE_CHARS = 6000
MAX_NEW_TOKENS = 400

# 1.0 = off, matching the original quantized run so precision is the only change.
REPETITION_PENALTY = 1.0

# Checkpoint to Drive if it is mounted, so a disconnect does not lose the run.
OUT_NAME = "borealis-27b-bf16.json"
DRIVE_DIR = "/content/drive/MyDrive/borealis_bench"
OUT_PATH = os.path.join(DRIVE_DIR, OUT_NAME) if os.path.isdir("/content/drive/MyDrive") else OUT_NAME

# Leave headroom on the GPU for KV cache and activations; the remainder spills
# to CPU RAM. Tune down if you hit OOM during generation rather than loading.
GPU_BUDGET = "34GiB"
CPU_BUDGET = "44GiB"
OFFLOAD_DIR = "/content/offload"


def preflight():
    """Fail loudly before a 51 GiB download if the runtime cannot finish the job."""
    print("=" * 70)
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU. Runtime > Change runtime type > A100 GPU + High-RAM.\n"
            "On CPU alone this would take days."
        )
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
    except Exception:
        ram = float("nan")
    print(f"GPU:  {name}  ({vram:.1f} GiB)")
    print(f"RAM:  {ram:.1f} GiB")
    print(f"Need: 51.1 GiB of bf16 weights, split across GPU + RAM")
    print("=" * 70)

    if vram + (0 if ram != ram else ram) < 60:
        print(
            "\nWARNING: GPU + RAM looks too small for 51.1 GiB of weights plus\n"
            "activations. If this is a T4/L4 or a standard-RAM runtime, stop now\n"
            "and switch to A100 + High-RAM — otherwise the load will OOM or the\n"
            "run will fall back to disk offload and take many hours.\n"
        )
    if vram < 30:
        raise SystemExit(
            f"{name} has only {vram:.1f} GiB. Full bf16 needs an A100 (40 GB).\n"
            "Colab has no 80 GB option, so there is no configuration where these\n"
            "weights fit on the GPU alone."
        )


def load_norsumm_test():
    path = hf_hub_download(
        repo_id="SamiaT/NorSumm",
        filename="nb/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    return [{"id": r["id"], "article": r["article"]} for _, r in df.iterrows()]


def load_model():
    print(f"Loading {MODEL_ID} at bf16 (no quantization) — this downloads ~51 GiB.")
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,          # full precision — nothing is quantized
        device_map="auto",
        max_memory={0: GPU_BUDGET, "cpu": CPU_BUDGET},
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Report the split so it is obvious how much ended up off-GPU (and therefore
    # how slow to expect this to be).
    devs = {}
    for _, dev in getattr(model, "hf_device_map", {}).items():
        devs[str(dev)] = devs.get(str(dev), 0) + 1
    print(f"Layer placement: {devs}")
    if any(d not in ("0", "cuda:0") for d in devs):
        print("Some layers are off-GPU — generation will be slow. This is expected.")
    return processor, model


def main():
    preflight()
    processor, model = load_model()
    articles = load_norsumm_test()

    outputs = []
    if os.path.exists(OUT_PATH):
        outputs = json.load(open(OUT_PATH, encoding="utf-8"))
        print(f"Resuming: {len(outputs)} articles already done.")
    done = {o["id"] for o in outputs}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True) if os.path.dirname(OUT_PATH) else None

    t0 = time.time()
    for i, art in enumerate(articles):
        if art["id"] in done:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": USER_PROMPT_TEMPLATE.format(article=art["article"][:MAX_ARTICLE_CHARS])},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

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
        summary = processor.decode(new_tokens, skip_special_tokens=True).strip()

        outputs.append({"id": art["id"], "summary": summary})
        # Checkpoint every article — Colab disconnects are a matter of when.
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        rate = elapsed / max(1, len(outputs) - (len(done)))
        print(f"[{len(outputs)}/{len(articles)}] {art['id']} -> {len(summary)} chars "
              f"({elapsed/60:.1f} min elapsed, ~{rate/60:.1f} min/article)")

    print(f"\nDone. Wrote {OUT_PATH}")
    print("Download it, rename to borealis-27b-bf16.json, and place it in "
          "benchmarks/results/generated_summaries/ so score_summaries.py scores it "
          "alongside the quantized run.")


if __name__ == "__main__":
    main()
