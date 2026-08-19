#!/usr/bin/env bash
# Run the Borealis-27b NorSumm + NorQuAD-QA benchmark on the Azure VM.
#
# Usage (on the VM):
#   bash run_borealis_on_vm.sh [QUANT] [REPEAT_PENALTY]
#   # or one-liner:
#   curl -sL https://raw.githubusercontent.com/rymarinelli/embedding/claude/norquad-norsumm-benchmarks-9qhyke/benchmarks/scripts/run_borealis_on_vm.sh | bash -s Q8_0 1.1
#
# QUANT default Q8_0 (~lossless, 28.7GB). Alternatives: Q5_K_M (faster),
# Q6_K, Q3_K_M (what the low CPU run used). REPEAT_PENALTY default 1.1.
set -euo pipefail

QUANT="${1:-Q8_0}"
REPEAT_PENALTY="${2:-1.1}"
BRANCH="claude/norquad-norsumm-benchmarks-9qhyke"

echo ">> Installing system deps..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git build-essential cmake

echo ">> Python venv + packages (venv avoids Ubuntu's PEP-668 'externally managed' error)..."
python3 -m venv ~/borealis-venv
# shellcheck disable=SC1091
source ~/borealis-venv/bin/activate
pip install --upgrade pip
pip install llama-cpp-python huggingface_hub rouge-score pandas pyarrow

echo ">> Cloning repo @ $BRANCH..."
if [ ! -d embedding ]; then git clone https://github.com/rymarinelli/embedding; fi
cd embedding
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull origin "$BRANCH"

GGUF_FILE="borealis-27b-${QUANT}.gguf"
echo ">> Downloading $GGUF_FILE (this is the large step)..."
huggingface-cli download NbAiLab/borealis-27b-gguf "$GGUF_FILE" --local-dir benchmarks/models

cd benchmarks/scripts
GGUF="../models/${GGUF_FILE}"
NTHREADS="$(nproc)"
LABEL="borealis-27b-${QUANT}"   # distinct label -> keeps the original Q3 row for comparison

echo ">> NorSumm summaries (label=$LABEL, threads=$NTHREADS, repeat_penalty=$REPEAT_PENALTY)..."
python3 generate_summaries_local_gguf.py "$GGUF" "$LABEL" 4096 "$NTHREADS" 500 "$REPEAT_PENALTY"

echo ">> NorQuAD QA (300-question sample)..."
python3 generate_qa_answers_local_gguf.py "$GGUF" "$LABEL" 4096 "$NTHREADS" 64 "$REPEAT_PENALTY"

echo ">> Scoring..."
python3 score_summaries.py
python3 score_qa.py
python3 make_report.py

cat <<EOF

============================================================
Done. New rows are labeled '$LABEL' in:
  benchmarks/results/norsumm_lexical_results.csv
  benchmarks/results/norquad_qa_results.csv
  benchmarks/results/REPORT.md
and the raw outputs in benchmarks/results/{generated_summaries,qa_answers}/${LABEL}.json

To keep the results, either:
  (a) commit+push from here (needs a GitHub token):
        git add -A benchmarks/results && git commit -m "Add $LABEL (fair-quant Borealis)" && git push
  (b) or copy them back to your machine, e.g.:
        scp -r azureuser@<vm-ip>:embedding/benchmarks/results/*_results.csv .
        scp azureuser@<vm-ip>:embedding/benchmarks/results/{generated_summaries,qa_answers}/${LABEL}.json .
============================================================
EOF
