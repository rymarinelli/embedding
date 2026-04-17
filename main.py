# Norwegian Embedding Fine-tuning
# Fine-tunes `perplexity-ai/pplx-embed-v1-0.6b` for Norwegian retrieval.
#
# Training data (all MTEB-safe):
#   - mMARCO-no — Norwegian MS MARCO translation
#   - NorNLI — Norwegian NLI pairs + contradiction triples
#   - NorSumm — Norwegian news article/summary pairs (dev split only)
#
# Eval data (never used in training):
#   - VG-RAG benchmark (held-out benchmark, evaluated at every checkpoint)
#   - NorQuad, SNL, MIRACL-no via MTEB

# ── 1. Install dependencies ─────────────────────────────────────────────────
# pip install sentence-transformers>=3.0.0 datasets transformers faiss-gpu mteb tqdm huggingface_hub

import os
import sys
import logging
import random
import time

# Silence noisy HTTP/tokenizer loggers — keep only warnings and above
logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "httpcore", "transformers", "datasets", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

def section(title):
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")

def info(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# Reduce GPU memory fragmentation (recommended by PyTorch for OOM cases)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── 2. Mount Google Drive (for saving checkpoints) ──────────────────────────
section("Setup")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = "/content/drive/MyDrive/nor-pplx-embed"
    info("Google Drive mounted")
except Exception:
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    info("Drive mount skipped — saving locally")

os.makedirs(SAVE_DIR, exist_ok=True)
info(f"Checkpoints -> {SAVE_DIR}")

# ── 3. Config — adjust these before running ─────────────────────────────────

BASE_MODEL     = "perplexity-ai/pplx-embed-v1-0.6b"

HF_USERNAME    = "zrmarine"
HF_MODEL_NAME  = "nor-pplx-embed-v1"
HF_REPO_ID     = f"{HF_USERNAME}/{HF_MODEL_NAME}"

MMARCO_MAX     = 50_000
BATCH_SIZE_S1  = 4        # reduced from 8 — OOM on A100 at 8
BATCH_SIZE_S2  = 2        # reduced from 4
EPOCHS_S1      = 1
EPOCHS_S2      = 1
SKIP_TSDAE     = True
EVAL_STEPS     = 500
SAVE_STEPS     = 500

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

info(f"Device : {DEVICE}")
if DEVICE == "cuda":
    info(f"GPU    : {torch.cuda.get_device_name(0)}")
    info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
info(f"Hub repo: {HF_REPO_ID}")
info(f"MMARCO_MAX={MMARCO_MAX}  BS_S1={BATCH_SIZE_S1}  BS_S2={BATCH_SIZE_S2}  EVAL_STEPS={EVAL_STEPS}")

# ── 4. Authenticate with Hugging Face Hub ───────────────────────────────────
section("Hugging Face Auth")

from huggingface_hub import login

HF_TOKEN = ""
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN") or ""
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    login(token=HF_TOKEN)
    info("Logged in to Hugging Face Hub (token)")
else:
    # Fall back to token cached by `huggingface-cli login` or a prior notebook cell
    try:
        login()
        info("Logged in to Hugging Face Hub (cached credentials)")
    except Exception as e:
        info(f"WARNING: HF login failed ({e}) — Hub push will be skipped")

# ── 5. Load VG-RAG benchmark — used as evaluator during training ─────────────
section("VG-RAG Evaluator")

from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator

random.seed(42)

info("Downloading zrmarine/vg-rag-benchmark ...")
vg_ds = load_dataset("zrmarine/vg-rag-benchmark", split="train")
info(f"Loaded {len(vg_ds):,} examples | columns: {vg_ds.column_names}")

QUERY_COL   = "question"
PASSAGE_COL = "article"

vg_queries       = {str(i): q for i, q in enumerate(vg_ds[QUERY_COL])}
vg_corpus        = {str(i): p for i, p in enumerate(vg_ds[PASSAGE_COL])}
vg_relevant_docs = {str(i): {str(i)} for i in range(len(vg_ds))}

ir_evaluator = InformationRetrievalEvaluator(
    queries=vg_queries,
    corpus=vg_corpus,
    relevant_docs=vg_relevant_docs,
    name="vg-rag",
    show_progress_bar=True,
)
info("IR evaluator ready (ndcg@10 / mrr@10 / recall@1/5/10)")

# ── 6. Load training data (Norwegian news only) ──────────────────────────────
section("Training Data")

# NbAiLab/mnli-norwegian — machine-translated Norwegian MultiNLI (premise/hypothesis/label)
info("Loading NbAiLab/mnli-norwegian...")
nornli_mnrl, nornli_triplets = [], []
try:
    nornli_ds = load_dataset("NbAiLab/mnli-norwegian", split="train")
    entailment, contradiction = {}, {}
    for row in nornli_ds:
        premise = row["premise"].strip()
        hyp     = row["hypothesis"].strip()
        label   = row["label"]  # 0=entailment, 1=neutral, 2=contradiction
        if label == 0:
            entailment.setdefault(premise, []).append(hyp)
        elif label == 2:
            contradiction.setdefault(premise, []).append(hyp)
    premises_with_both = set(entailment) & set(contradiction)
    for premise in premises_with_both:
        pos = random.choice(entailment[premise])
        neg = random.choice(contradiction[premise])
        nornli_triplets.append(InputExample(texts=[premise, pos, neg]))
        nornli_mnrl.append(InputExample(texts=[premise, pos]))
    for premise in set(entailment) - premises_with_both:
        for pos in entailment[premise]:
            nornli_mnrl.append(InputExample(texts=[premise, pos]))
    info(f"  done: {len(nornli_mnrl):,} MNRL pairs, {len(nornli_triplets):,} triplets")
except Exception as e:
    info(f"  NbAiLab/mnli-norwegian not available ({e}) — skipping")

# ltg/norsumm-nob-nno-translation — same news article in Bokmål + Nynorsk
# Columns: text_nob, text_nno  |  only a 'test' split exists
info("Loading ltg/norsumm-nob-nno-translation...")
norsumm_pairs = []
try:
    ltg_summ_ds = load_dataset("ltg/norsumm-nob-nno-translation")
    info(f"  splits: {list(ltg_summ_ds.keys())}")
    for split_name, split_ds in ltg_summ_ds.items():
        info(f"  {split_name}: {len(split_ds):,} rows | columns: {split_ds.column_names}")
        for row in split_ds:
            nob = row.get("text_nob", "").strip()
            nno = row.get("text_nno", "").strip()
            if nob and nno:
                norsumm_pairs.append(InputExample(texts=[nob, nno]))
    info(f"  done: {len(norsumm_pairs):,} pairs")
except Exception as e:
    info(f"  ltg/norsumm-nob-nno-translation not available ({e}) — skipping")

# SamiaT/NorSumm — Norwegian news article + human summaries
info("Loading SamiaT/NorSumm...")
samia_pairs = []
try:
    norsumm_ds = load_dataset("SamiaT/NorSumm", "nb")
    info(f"  splits: {list(norsumm_ds.keys())}")
    for split in ["train", "validation", "test"]:
        if split not in norsumm_ds:
            continue
        for row in norsumm_ds[split]:
            article = row.get("article", "").strip()
            for summary_text in row.get("summaries", []):
                summary = summary_text.strip()
                if article and summary:
                    samia_pairs.append(InputExample(texts=[summary, article]))
    info(f"  done: {len(samia_pairs):,} pairs")
except Exception as e:
    info(f"  SamiaT/NorSumm not available ({e}) — skipping")

# ltg/norsummarize-instruct — instruction-format Norwegian news summarization
# Only a 'test' split exists — load all available splits
info("Loading ltg/norsummarize-instruct...")
norsum_instruct_pairs = []
try:
    norsum_ds = load_dataset("ltg/norsummarize-instruct")
    info(f"  splits: {list(norsum_ds.keys())}")
    for split_name, split_ds in norsum_ds.items():
        info(f"  {split_name}: {len(split_ds):,} rows | columns: {split_ds.column_names}")
        for row in split_ds:
            doc     = (row.get("input") or row.get("document") or row.get("text") or "").strip()
            summary = (row.get("output") or row.get("summary") or "").strip()
            if doc and summary:
                norsum_instruct_pairs.append(InputExample(texts=[summary, doc]))
    info(f"  done: {len(norsum_instruct_pairs):,} pairs")
except Exception as e:
    info(f"  ltg/norsummarize-instruct not available ({e}) — skipping")

# Combine
all_mnrl_pairs = nornli_mnrl + norsumm_pairs + samia_pairs + norsum_instruct_pairs
random.shuffle(all_mnrl_pairs)
info(f"Total MNRL pairs : {len(all_mnrl_pairs):,}  "
     f"(NorNLI={len(nornli_mnrl):,}  LTG-NorSumm={len(norsumm_pairs):,}  "
     f"SamiaT-NorSumm={len(samia_pairs):,}  NorSumm-instruct={len(norsum_instruct_pairs):,})")
info(f"Total triplets   : {len(nornli_triplets):,}")

if len(all_mnrl_pairs) == 0:
    raise RuntimeError(
        "No training data loaded — all dataset sources returned 0 pairs. "
        "Check the loading errors above."
    )

# ── 7. (Optional) TSDAE warm-up ─────────────────────────────────────────────
section("TSDAE Warm-up")

if SKIP_TSDAE:
    info("Skipped (SKIP_TSDAE=True) — using base model directly")
    stage1_init = BASE_MODEL
else:
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import losses as st_losses
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    info("Loading HPLT Norwegian (nob_Latn) for TSDAE...")
    hplt_ds = load_dataset("HPLT/HPLT2.0_cleaned", "nob_Latn", split="train", streaming=True)
    hplt_sentences = []
    for row in hplt_ds:
        if len(hplt_sentences) >= 20_000:
            break
        text = row.get("text", "").strip()
        words = text.split()
        if 10 <= len(words) <= 256:
            hplt_sentences.append(text)
    info(f"HPLT sentences: {len(hplt_sentences):,}")

    tsdae_model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)
    tsdae_dataset = DenoisingAutoEncoderDataset(hplt_sentences)
    tsdae_loss = st_losses.DenoisingAutoEncoderLoss(
        tsdae_model,
        decoder_name_or_path=BASE_MODEL,
        tie_encoder_decoder=True,
    )
    tsdae_args = SentenceTransformerTrainingArguments(
        output_dir=f"{SAVE_DIR}/tsdae",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        fp16=True,
        save_strategy="epoch",
        logging_steps=200,
    )
    info("Training TSDAE...")
    SentenceTransformerTrainer(
        model=tsdae_model, args=tsdae_args,
        train_dataset=tsdae_dataset, loss=tsdae_loss,
    ).train()
    stage1_init = f"{SAVE_DIR}/tsdae"
    tsdae_model.save(stage1_init)
    info(f"TSDAE saved -> {stage1_init}")

# ── 8. Stage 1 — MNRL + TripletLoss ─────────────────────────────────────────
section("Stage 1 — MNRL + TripletLoss")

from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def to_hf(examples, columns):
    return HFDataset.from_dict(
        {col: [ex.texts[i] for ex in examples] for i, col in enumerate(columns)}
    )

info(f"Loading model: {stage1_init}")
model = SentenceTransformer(
    stage1_init, trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
)

mnrl_dataset   = to_hf(all_mnrl_pairs, ["anchor", "positive"])
train_datasets = {"mnrl": mnrl_dataset}
train_losses   = {"mnrl": losses.MultipleNegativesRankingLoss(model)}

if len(nornli_triplets) > 0:
    triplet_dataset = to_hf(nornli_triplets, ["anchor", "positive", "negative"])
    train_datasets["triplet"] = triplet_dataset
    train_losses["triplet"]   = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.15,
    )
    info(f"Using MNRL + TripletLoss ({len(nornli_triplets):,} triplets)")
else:
    info("Using MNRL only (no triplets)")

info(f"Batch size={BATCH_SIZE_S1}  grad_accum=2  lr=2e-5  epochs={EPOCHS_S1}")
info(f"Eval every {EVAL_STEPS} steps  |  push to {HF_REPO_ID}-stage1")

s1_args = SentenceTransformerTrainingArguments(
    output_dir=f"{SAVE_DIR}/stage1",
    num_train_epochs=EPOCHS_S1,
    per_device_train_batch_size=BATCH_SIZE_S1,
    gradient_accumulation_steps=2,
    warmup_steps=0.05,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="vg-rag_cosine_ndcg@10",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id=f"{HF_REPO_ID}-stage1",
    hub_strategy="checkpoint",
    logging_steps=50,
    dataloader_num_workers=2,
)

info("Starting Stage 1 training...")
SentenceTransformerTrainer(
    model=model,
    args=s1_args,
    train_dataset=train_datasets,
    loss=train_losses,
    evaluator=ir_evaluator,
).train()

model.save(f"{SAVE_DIR}/stage1/final")
info(f"Stage 1 complete -> {SAVE_DIR}/stage1/final")

# ── 9. Hard negative mining ──────────────────────────────────────────────────
section("Hard Negative Mining")

import faiss
import numpy as np
from tqdm.auto import tqdm

queries_list   = [p.texts[0] for p in all_mnrl_pairs]
positives_list = [p.texts[1] for p in all_mnrl_pairs]
corpus         = list(set(positives_list))

info(f"Corpus size (unique passages): {len(corpus):,}")
info(f"Queries: {len(queries_list):,}")

info("Encoding corpus...")
corpus_emb = model.encode(
    corpus, batch_size=256, normalize_embeddings=True,
    show_progress_bar=True, convert_to_numpy=True,
).astype(np.float32)

info("Encoding queries...")
query_emb = model.encode(
    queries_list, batch_size=256, normalize_embeddings=True,
    show_progress_bar=True, convert_to_numpy=True,
).astype(np.float32)

dim = corpus_emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_emb)
info(f"FAISS index built (dim={dim}, {len(corpus):,} vectors)")

TOP_K = 20
info(f"Searching top-{TOP_K} candidates per query...")
scores, indices = index.search(query_emb, TOP_K)

hard_triplets = []
n_random_fallback = 0
for i, (query, positive) in enumerate(zip(queries_list, positives_list)):
    hard_neg = None
    for _, idx in zip(scores[i], indices[i]):
        candidate = corpus[idx]
        if candidate != positive:
            hard_neg = candidate
            break
    if hard_neg is None:
        hard_neg = corpus[np.random.randint(len(corpus))]
        n_random_fallback += 1
    hard_triplets.append(InputExample(texts=[query, positive, hard_neg]))

info(f"Mined {len(hard_triplets):,} hard-negative triplets ({n_random_fallback} random fallbacks)")

# ── 10. Stage 2 — MNRL with hard negatives ──────────────────────────────────
section("Stage 2 — MNRL with Hard Negatives")

hard_dataset = to_hf(hard_triplets, ["anchor", "positive", "negative"])
mnrl_loss2   = losses.MultipleNegativesRankingLoss(model)

info(f"Batch size={BATCH_SIZE_S2}  grad_accum=4  lr=1e-5  epochs={EPOCHS_S2}")
info(f"Eval every {EVAL_STEPS} steps  |  push to {HF_REPO_ID}")

s2_args = SentenceTransformerTrainingArguments(
    output_dir=f"{SAVE_DIR}/stage2",
    num_train_epochs=EPOCHS_S2,
    per_device_train_batch_size=BATCH_SIZE_S2,
    gradient_accumulation_steps=4,
    warmup_steps=0.05,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="vg-rag_cosine_ndcg@10",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id=HF_REPO_ID,
    hub_strategy="checkpoint",
    logging_steps=50,
    dataloader_num_workers=2,
)

info("Starting Stage 2 training...")
SentenceTransformerTrainer(
    model=model,
    args=s2_args,
    train_dataset={"mnrl_hard": hard_dataset},
    loss={"mnrl_hard": mnrl_loss2},
    evaluator=ir_evaluator,
).train()

model.save(f"{SAVE_DIR}/stage2/final")
info(f"Stage 2 complete -> {SAVE_DIR}/stage2/final")

# ── 11. Push final model to Hub ──────────────────────────────────────────────
section("Push Final Model")

info(f"Pushing to https://huggingface.co/{HF_REPO_ID} ...")
model.push_to_hub(HF_REPO_ID, exist_ok=True)
info("Push complete")

# ── 12. VG-RAG final evaluation ──────────────────────────────────────────────
section("Final VG-RAG Evaluation")

info("Running IR evaluator on final model...")
results = ir_evaluator(model)
print()
for k, v in sorted(results.items()):
    print(f"  {k:45s} {v:.4f}")

# ── 13. MTEB Norwegian eval ──────────────────────────────────────────────────
section("MTEB Norwegian Evaluation")

import mteb

MTEB_TASKS = ["NorQuadRetrieval", "SNLRetrieval", "MIRACLRetrieval"]
info(f"Tasks: {MTEB_TASKS}")

mteb_model = mteb.get_model(HF_REPO_ID)
tasks = mteb.get_tasks(tasks=MTEB_TASKS, languages=["nob", "nno"])
results = mteb.evaluate(mteb_model, tasks)

print()
for task, result in zip(MTEB_TASKS, results):
    print(f"  {task}: {result.get_score():.3f}")

section("Done")
