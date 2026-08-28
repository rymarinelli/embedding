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
On Colab Pro+, also turn on background execution so the run survives a closed
browser — this takes hours, and an interrupted runtime loses the GPU.

Anything less (T4 16 GB, L4 22.5 GB, or standard-RAM) will not complete: the
preflight check below says so before the 51 GiB download starts, not after.

The GPU/CPU split is sized from the card actually assigned, so if Colab hands
out an 80 GB A100 the model becomes fully GPU-resident and the run drops from
hours to minutes without any edit here.

WHAT THIS IS COMPARED AGAINST, AND THE ONE CONFOUND
---------------------------------------------------
The committed borealis-27b row (ROUGE-L 0.2702) came from
generate_summaries_local_gguf.py: llama.cpp, GGUF-quantized, on CPU, with
temperature=0.2, max_tokens=500 and repeat_penalty=1.0.

Everything controllable is matched to that run — dataset, prompts (asserted
identical to prompts.py), article truncation, decoding settings, and the scorer.
REPETITION_PENALTY stays 1.0 rather than 1.1, because the earlier diagnosis
blamed quantization AND a missing repetition penalty together; changing both at
once could not attribute the result.

One difference cannot be removed: the inference stack. The baseline ran through
llama.cpp on GGUF weights, this runs through transformers on safetensors. So a
score change here is "bf16-via-transformers vs quant-via-llama.cpp", not
purely a precision effect — tokenization, chat templating and sampling all
differ slightly between the two runtimes.

To attribute the difference to precision alone, also run
generate_summaries_borealis_colab.py (4-bit, same transformers stack) and
compare against that instead. Then only the precision changes.

USAGE (Colab)
-------------
    !pip install -q -U transformers accelerate huggingface_hub pandas pyarrow
    from huggingface_hub import login; login(token="<your HF token>")
    from google.colab import drive; drive.mount("/content/drive")   # optional but recommended
    # then paste this file and call main()

Output: borealis-27b-bf16-full.json, in the same {id, summary} shape the other
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

# Decoding settings below are matched to the ORIGINAL Borealis run, which used
# generate_summaries_local_gguf.py (llama.cpp, GGUF-quantized, on CPU) — not the
# 4-bit Colab script. That run used temperature=0.2, max_tokens=500 and
# repeat_penalty=1.0. Any of these left unmatched becomes a second variable
# alongside precision and makes the comparison unattributable.
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.2
REPETITION_PENALTY = 1.0

# Set True for greedy decoding instead. Cleaner and reproducible, but then this
# run differs from the original in decoding as well as precision — only do it if
# you are re-running the GGUF baseline greedily too.
GREEDY = False

# Checkpoint to Drive if it is mounted, so a disconnect does not lose the run.
# Filename must match the label wanted in norsumm_lexical_results.csv —
# score_summaries.py takes the model name from the filename stem.
OUT_NAME = "borealis-27b-bf16-full.json"
# Each benchmark gets its own subdirectory: the QA run writes the SAME filename
# (it needs the same label in its own table), so a shared folder would have the
# second run silently overwrite the first.
DRIVE_DIR = "/content/drive/MyDrive/borealis_bench/norsumm"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(DRIVE_DIR, exist_ok=True)
    OUT_PATH = os.path.join(DRIVE_DIR, OUT_NAME)
else:
    OUT_PATH = OUT_NAME

WEIGHTS_GIB = 51.1          # bf16 total, from the safetensors index
GPU_HEADROOM_GIB = 6        # KV cache + activations must not compete with weights
DISK_NEEDED_GIB = 56.0      # the download itself, plus a little slack
OFFLOAD_DIR = "/content/offload"


def memory_budgets():
    """Size the GPU/CPU split from the hardware actually assigned.

    Colab hands out different cards (T4 16 GB, L4 22.5 GB, A100 40 GB, rarely
    A100 80 GB). Hardcoding a 40 GB split would leave an 80 GB card half idle —
    and on an 80 GB card the whole model is GPU-resident, which is roughly an
    order of magnitude faster than offloading.

    The CPU budget is deliberately well under total RAM. accelerate fills the
    budget it is given, and the loading process, torch, CUDA context and the
    dataset all need room on top. A budget near total RAM gets the kernel
    OOM-killed mid-load, which Colab surfaces only as a WebSocket CloseEvent
    in the browser with no traceback.
    """
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024**3
    except Exception:
        ram = 50.0
    gpu_budget = max(1.0, vram - GPU_HEADROOM_GIB)
    fits_on_gpu = gpu_budget >= WEIGHTS_GIB
    cpu_budget = max(1.0, min(ram * 0.70, ram - 14.0))
    return gpu_budget, cpu_budget, fits_on_gpu, vram, ram


def check_disk():
    """The 51 GiB download needs somewhere to land before anything else matters."""
    import shutil
    free = shutil.disk_usage("/").free / 1024**3
    print(f"Disk:   {free:.1f} GiB free (need ~{DISK_NEEDED_GIB:.0f} GiB for the download)")
    if free < DISK_NEEDED_GIB:
        raise SystemExit(
            f"\nOnly {free:.1f} GiB free. The bf16 weights are {WEIGHTS_GIB} GiB and will\n"
            "not fit. Free space, or use a runtime with a larger disk. Clearing a\n"
            "previous partial download often recovers enough:\n"
            "    !rm -rf /root/.cache/huggingface/hub/models--NbAiLab--borealis-27b"
        )
    return free


def preflight():
    """Fail loudly before a 51 GiB download if the runtime cannot finish the job."""
    print("=" * 70)
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU. Runtime > Change runtime type > A100 GPU + High-RAM.\n"
            "On CPU alone this would take days."
        )
    name = torch.cuda.get_device_name(0)
    gpu_budget, cpu_budget, fits_on_gpu, vram, ram = memory_budgets()

    print(f"GPU:    {name}  ({vram:.1f} GiB, {gpu_budget:.1f} GiB usable for weights)")
    print(f"RAM:    {ram:.1f} GiB  ({cpu_budget:.1f} GiB budgeted, rest reserved for the process)")
    check_disk()
    print(f"Weights: {WEIGHTS_GIB} GiB bf16 — nothing quantized")

    if fits_on_gpu:
        print("\nEntire model fits on the GPU. No offload — expect fast generation\n"
              "(tens of tokens/sec) and roughly 10-20 minutes for all 33 articles.")
    else:
        spill = WEIGHTS_GIB - gpu_budget
        print(f"\n{gpu_budget:.1f} GiB on GPU, ~{spill:.1f} GiB offloaded to CPU RAM.")
        print("Offloaded layers are streamed per token, so expect ~1-3 tokens/sec\n"
              "and roughly 1.5-4 hours for all 33 articles.")
        if spill > cpu_budget:
            raise SystemExit(
                f"\nThe ~{spill:.1f} GiB spill does not fit in {cpu_budget:.1f} GiB of RAM.\n"
                "Switch to a High-RAM runtime (Runtime > Change runtime type)."
            )
        print("\nOn Colab Pro+, enable background execution so the run survives a\n"
              "closed browser — a job this long will otherwise be interrupted.")
    print("=" * 70)

    if vram < 30:
        raise SystemExit(
            f"\n{name} has only {vram:.1f} GiB. Even with CPU offload this card will\n"
            "spill ~{:.0f} GiB to RAM and thrash. Switch to an A100 runtime.".format(
                WEIGHTS_GIB - gpu_budget)
        )


def load_norsumm_test():
    path = hf_hub_download(
        repo_id="SamiaT/NorSumm",
        filename="nb/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    return [{"id": r["id"], "article": r["article"]} for _, r in df.iterrows()]


def verify_unquantized(model):
    """Prove from the loaded model that nothing was quantized.

    The point of this run is precision, so 'it is unquantized because the script
    does not import bitsandbytes' is not good enough — a stray config or a
    prequantized checkpoint would defeat that. This inspects what is actually in
    memory: parameter dtypes, the layer classes bitsandbytes swaps in, and the
    total parameter count.
    """
    print("\n--- precision check ---")
    by_dtype = {}
    for p in model.parameters():
        by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel()
    total = sum(by_dtype.values())
    for dt, n in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
        print(f"  {dt:<16} {n/1e9:7.2f} B params  ({100*n/total:5.1f}%)")
    print(f"  {'TOTAL':<16} {total/1e9:7.2f} B params")

    # bitsandbytes replaces nn.Linear with these; their presence means quantized.
    bad = {"Linear4bit", "Linear8bitLt", "Params4bit", "Int8Params"}
    found = {type(m).__name__ for m in model.modules()} & bad
    quantized_dtypes = {d for d in by_dtype if "int8" in d or "uint8" in d or "int4" in d}

    if found or quantized_dtypes:
        raise SystemExit(
            f"\nQUANTIZED WEIGHTS DETECTED — this run would not answer the question.\n"
            f"  quantized layer classes: {found or 'none'}\n"
            f"  integer param dtypes:    {quantized_dtypes or 'none'}\n"
            "Check that bitsandbytes is not installed and no quantization_config is set."
        )

    if total < 20e9:
        print(f"\n  WARNING: only {total/1e9:.1f}B parameters materialised. Expected ~27B.\n"
              "  Some weights may have failed to load — do not trust the scores.")
    else:
        print(f"  OK: all {total/1e9:.1f}B parameters are bf16. Nothing quantized.")
    print("-----------------------\n")


def load_model():
    print(f"Loading {MODEL_ID} at bf16 (no quantization) — this downloads ~51 GiB.")
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    gpu_budget, cpu_budget, _, _, _ = memory_budgets()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,          # full precision — nothing is quantized
        device_map="auto",
        max_memory={0: f"{gpu_budget:.0f}GiB", "cpu": f"{cpu_budget:.0f}GiB"},
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )
    model.eval()
    verify_unquantized(model)

    # Report the split so it is obvious how much ended up off-GPU (and therefore
    # how slow to expect this to be), plus what RAM survived the load.
    devs = {}
    for _, dev in getattr(model, "hf_device_map", {}).items():
        devs[str(dev)] = devs.get(str(dev), 0) + 1
    print(f"Layer placement: {devs}")
    try:
        import psutil
        print(f"RAM after load: {psutil.virtual_memory().available/1024**3:.1f} GiB still free")
    except Exception:
        pass
    if any("disk" in d for d in devs):
        print("WARNING: layers landed on DISK. Generation will be extremely slow —\n"
              "use a High-RAM runtime so the spill fits in memory instead.")
    elif any(d not in ("0", "cuda:0") for d in devs):
        print("Some layers are on CPU — generation will be slow. This is expected.")
    return processor, model


def main(model=None, processor=None):
    """Run the NorSumm benchmark.

    Accepts an already-loaded model so a Colab session can run this and the
    NorQuAD QA benchmark off one 51 GiB load instead of two.
    """
    if model is None or processor is None:
        preflight()
        processor, model = load_model()
    else:
        print("Reusing the already-loaded model — no second 51 GiB download.")
        verify_unquantized(model)
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
                repetition_penalty=REPETITION_PENALTY,
                **({"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
                   if GREEDY else
                   {"do_sample": True, "temperature": TEMPERATURE}),
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
