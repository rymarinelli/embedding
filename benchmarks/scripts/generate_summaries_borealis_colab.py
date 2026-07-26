"""Run this in a Google Colab GPU runtime to generate NorSumm summaries with
NbAiLab/borealis-27b (4-bit quantized — a single T4/L4/A100 GPU is enough).

This is NOT run as part of the rest of the benchmark pipeline, because the
container running the other scripts has no GPU and OpenRouter does not host
Borealis. Run this notebook cell-by-cell in Colab, then download the output
file it writes (borealis-27b.json) and drop it into
benchmarks/results/generated_summaries/ in the repo so score_summaries.py
picks it up alongside the OpenRouter models.

Colab setup:
    !pip install -q -U transformers accelerate bitsandbytes huggingface_hub pandas pyarrow

    from huggingface_hub import login
    login(token="<your HF token>")   # needed once per runtime, never commit it

Then run this file's `main()`.
"""
import json

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "NbAiLab/borealis-27b"

# Must match benchmarks/scripts/prompts.py exactly so every model is scored
# on the same instruction/input.
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


def load_norsumm_test():
    path = hf_hub_download(
        repo_id="SamiaT/NorSumm",
        filename="nb/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    return [{"id": r["id"], "article": r["article"]} for _, r in df.iterrows()]


def main():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    articles = load_norsumm_test()
    outputs = []
    for i, art in enumerate(articles):
        article_text = art["article"][:MAX_ARTICLE_CHARS]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(article=article_text)},
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
            )
        new_tokens = gen[0][inputs["input_ids"].shape[-1]:]
        summary = processor.decode(new_tokens, skip_special_tokens=True).strip()

        outputs.append({"id": art["id"], "summary": summary})
        print(f"[{i+1}/{len(articles)}] {art['id']} -> {len(summary)} chars")

    with open("borealis-27b.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print("Wrote borealis-27b.json — download it and place it in "
          "benchmarks/results/generated_summaries/ in the repo.")


if __name__ == "__main__":
    main()
