# Running Borealis-27b fairly on an Azure VM

The Borealis-27b number in the NorSumm/NorQuAD tables was produced on a small CPU box
(15 GB RAM), which forced the aggressive **Q3_K_M** (~3-bit) quantization *and* ran with no
repetition penalty. That combination produced short, sometimes verbatim-repetitive summaries
(avg ~338 chars vs ~538 for the full-precision base model) and almost certainly **understates**
the model. This guide reruns Borealis at a much higher-fidelity quant with a repeat penalty so
the result is a fair comparison.

`NbAiLab/borealis-27b` (HF safetensors) and `NbAiLab/borealis-27b-gguf` are the **same weights** —
the GGUF repo is just the CPU-optimized packaging. On CPU, the GGUF is the right form to use;
on GPU you can use either.

## Which quant?

| quant | size | quality | CPU speed |
|---|---|---|---|
| Q3_K_M | 13.4 GB | aggressive (what we used) | fastest |
| Q5_K_M | 19.3 GB | very good | medium |
| Q6_K | 22.2 GB | near-lossless | slower |
| **Q8_0** | **28.7 GB** | **essentially lossless** | **slowest** |
| BF16 (unquantized) | 54 GB | reference | slowest, needs 64 GB+ RAM |

Recommended: **Q8_0** for the definitive "fair" number, or **Q5_K_M** if you want it faster.
BF16 is not worth it over Q8 on CPU.

## Pick a VM (CPU path — no GPU quota needed)

| VM size | vCPU / RAM | ~$/hr PAYG | notes |
|---|---|---|---|
| `Standard_F32s_v2` | 32 / 64 GB | ~$1.35 | fast (AVX-512); fits up to Q8. Needs FSv2 quota ≥32. |
| `Standard_E16s_v5` | 16 / 128 GB | ~$1.0 | fewer cores (slower) but RAM-comfy even for BF16; often within default quota. |
| `Standard_E32s_v5` | 32 / 256 GB | ~$2.0 | fast + huge RAM, for BF16. |

The whole workload is ~32K output tokens (33 summaries + 300 QA answers). On a 32-core CPU VM at
Q8 expect **~4–8 hours** wall-clock → **~$5–15 total.** (A GPU VM would finish in minutes — see the
GPU note at the bottom — but needs GPU-quota approval.)

### Check / request vCPU quota

CPU quota is per-family, per-region, and new subscriptions are often capped below 32 vCPUs.

```bash
az vm list-usage -l eastus -o table | grep -iE "FSv2|ESv5"
```

If the family's limit is below what your VM needs, request an increase in the Portal:
**Subscription → Usage + quotas → filter by region + family → Request increase.** CPU increases
are usually approved within minutes (unlike GPU).

## Create the VM (Azure CLI)

```bash
az login
az group create -n borealis-rg -l eastus

az vm create \
  --resource-group borealis-rg \
  --name borealis-vm \
  --image Ubuntu2204 \
  --size Standard_F32s_v2 \
  --os-disk-size-gb 128 \
  --admin-username azureuser \
  --generate-ssh-keys

# get the public IP and SSH in
IP=$(az vm show -d -g borealis-rg -n borealis-vm --query publicIps -o tsv)
ssh azureuser@$IP
```

(Portal equivalent: *Create a resource → Virtual machine*, Ubuntu 22.04, change size to
`Standard_F32s_v2`, set OS disk to 128 GB, create with SSH key.)

## Run the benchmark on the VM

```bash
sudo apt-get update && sudo apt-get install -y python3-pip git
pip install "llama-cpp-python" huggingface_hub rouge-score pandas pyarrow

git clone https://github.com/rymarinelli/embedding && cd embedding
git checkout claude/norquad-norsumm-benchmarks-9qhyke

# download the fair-quant Borealis weights (~29 GB for Q8_0)
huggingface-cli download NbAiLab/borealis-27b-gguf borealis-27b-Q8_0.gguf \
  --local-dir benchmarks/models

cd benchmarks/scripts
GGUF=../models/borealis-27b-Q8_0.gguf
NTHREADS=$(nproc)

# NorSumm summaries: n_ctx=4096, threads=all, max_tokens=500, repeat_penalty=1.1
python3 generate_summaries_local_gguf.py "$GGUF" borealis-27b 4096 $NTHREADS 500 1.1

# NorQuAD QA (300-question sample): max_tokens=64, repeat_penalty=1.1
python3 generate_qa_answers_local_gguf.py "$GGUF" borealis-27b 4096 $NTHREADS 64 1.1

# score both
python3 score_summaries.py
python3 score_qa.py
python3 make_report.py
```

The output files (`results/generated_summaries/borealis-27b.json`,
`results/qa_answers/borealis-27b.json`, and the updated CSVs) are drop-in — commit/push them, or
copy them back and commit locally. Consider a distinct label like `borealis-27b-Q8` if you want to
keep the original Q3 row for comparison rather than overwrite it.

> If comparing against the existing rows, remember the other Norwegian generative models
> (NorMistral-7B/11B) were also run at Q4/CPU without a repeat penalty — for a strict
> apples-to-apples you'd re-run those the same way, or just note Borealis's row is the higher-fidelity one.

## Tear down (important — avoid idle billing)

A **stopped but still-allocated** VM keeps billing. Fully remove it:

```bash
az group delete -n borealis-rg --yes --no-wait
```

## GPU alternative (minutes instead of hours, but needs GPU quota)

Rent `Standard_NC24ads_A100_v4` (1× A100 80 GB, ~$3.7/hr), and either:
- run the same GGUF via `llama-cpp-python` built with CUDA (`n_gpu_layers=-1`), or
- run the original `NbAiLab/borealis-27b` safetensors via transformers — bf16 fits on the 80 GB
  card, or 4-bit (`bitsandbytes`) fits on a 24 GB card. `scripts/generate_summaries_borealis_colab.py`
  already does the 4-bit transformers path.

GPU generation finishes in minutes; the blocker is getting `NCADSA100v4` quota approved, which
is why the CPU path above is often the lower-friction choice.
