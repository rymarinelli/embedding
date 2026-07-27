"""Fine-tune the mmarco-mMiniLMv2 cross-encoder reranker on NorSumm
(summary, article) pairs, mirroring this repo's own main.py design where
NorSumm's dev/validation split is training data and NorQuAD is the
held-out eval set (never used in training).

Uses the NorSumm nb *validation* split (30 articles x 3 summaries = 90
positive pairs) -- NOT the test split, which is reserved for the NorSumm
summarization benchmark elsewhere in this repo, to avoid any overlap
between "data used to fine-tune the reranker" and "data used to evaluate
generation quality". Negatives are (summary, unrelated article) pairs
sampled from other articles in the same split.

This is a small-data proof-of-concept fine-tune (90 positives is tiny for
a cross-encoder) -- expect a directional signal, not a robust one.
"""
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

BASE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "reranker-norsumm-finetuned"
NEGATIVES_PER_POSITIVE = 4
EPOCHS = 4
BATCH_SIZE = 8
SEED = 42


def load_norsumm_validation():
    path = hf_hub_download(repo_id="SamiaT/NorSumm", filename="nb/validation-00000-of-00001.parquet",
                            repo_type="dataset")
    df = pd.read_parquet(path)
    return [{"id": r["id"], "article": r["article"], "summaries": list(r["summaries"])} for _, r in df.iterrows()]


def build_examples(articles):
    random.seed(SEED)
    examples = []
    n = len(articles)
    for i, art in enumerate(articles):
        for summary in art["summaries"]:
            examples.append(InputExample(texts=[summary, art["article"]], label=1.0))
            other_idx = [j for j in range(n) if j != i]
            negs = random.sample(other_idx, min(NEGATIVES_PER_POSITIVE, len(other_idx)))
            for j in negs:
                examples.append(InputExample(texts=[summary, articles[j]["article"]], label=0.0))
    random.shuffle(examples)
    return examples


def main():
    print("Loading NorSumm nb validation split...")
    articles = load_norsumm_validation()
    print(f"  {len(articles)} articles")

    examples = build_examples(articles)
    n_pos = sum(1 for e in examples if e.label == 1.0)
    print(f"Built {len(examples)} training pairs ({n_pos} positive, {len(examples)-n_pos} negative)")

    model = CrossEncoder(BASE_MODEL, num_labels=1)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)

    print(f"Fine-tuning for {EPOCHS} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=EPOCHS,
        loss_fct=BinaryCrossEntropyLoss(model),
        warmup_steps=int(0.1 * len(train_dataloader) * EPOCHS),
        show_progress_bar=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_DIR))
    print(f"Saved fine-tuned reranker to {OUT_DIR}")


if __name__ == "__main__":
    main()
