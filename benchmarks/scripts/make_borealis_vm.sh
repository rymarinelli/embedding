#!/usr/bin/env bash
# Create an Azure CPU VM sized to run Borealis-27b at a fair (high) quant.
#
# Easiest place to run this: Azure Cloud Shell (bash) — az is preinstalled and
# already logged in, so you can paste this straight in. Or run locally if you
# have the Azure CLI installed and have run `az login`.
#
# Safety: this creates its OWN dedicated resource group (default
# borealis-bench-rg) so that the teardown command at the end can delete the
# whole group WITHOUT touching your other projects (e.g. influence-rag-rg).
set -euo pipefail

# ---------- config (override via env vars if you like) ----------
RESOURCE_GROUP="${RESOURCE_GROUP:-borealis-bench-rg}"   # dedicated RG -> safe to delete later
LOCATION="${LOCATION:-northeurope}"
VM_NAME="${VM_NAME:-borealis-bench}"
VM_SIZE="${VM_SIZE:-Standard_E16s_v5}"                  # 16 vCPU / 128 GB RAM — fits Q8 (28.7GB) comfortably
OS_DISK_GB="${OS_DISK_GB:-256}"                         # room for the ~29GB model + env
ADMIN_USER="${ADMIN_USER:-azureuser}"
IMAGE="${IMAGE:-Ubuntu2204}"                            # broad wheel compatibility
# ----------------------------------------------------------------

command -v az >/dev/null || { echo "Azure CLI (az) not found — install from https://aka.ms/azcli or use Azure Cloud Shell"; exit 1; }
az account show >/dev/null 2>&1 || az login -o none

echo ">> Ensuring dedicated resource group '$RESOURCE_GROUP' in $LOCATION..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

echo ">> Creating VM '$VM_NAME' ($VM_SIZE, $OS_DISK_GB GB disk)..."
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --location "$LOCATION" \
  --image "$IMAGE" \
  --size "$VM_SIZE" \
  --os-disk-size-gb "$OS_DISK_GB" \
  --admin-username "$ADMIN_USER" \
  --security-type Standard \
  --public-ip-sku Standard \
  --generate-ssh-keys \
  -o table

echo ">> Opening SSH (port 22)..."
az vm open-port -g "$RESOURCE_GROUP" -n "$VM_NAME" --port 22 --priority 1000 -o none || true

echo ">> Setting auto-shutdown at 03:00 UTC (credit protection)..."
az vm auto-shutdown -g "$RESOURCE_GROUP" -n "$VM_NAME" --time 0300 -o none || true

IP="$(az vm show -d -g "$RESOURCE_GROUP" -n "$VM_NAME" --query publicIps -o tsv)"

cat <<EOF

============================================================
VM is ready.

  Connect:   ssh ${ADMIN_USER}@${IP}

  Then run the benchmark (Q8_0 quant, repeat_penalty 1.1):
    curl -sL https://raw.githubusercontent.com/rymarinelli/embedding/claude/norquad-norsumm-benchmarks-9qhyke/benchmarks/scripts/run_borealis_on_vm.sh | bash -s Q8_0 1.1

  (Optional) restrict SSH to just your IP:
    MYIP=\$(curl -s https://api.ipify.org)
    NSG=\$(az network nsg list -g $RESOURCE_GROUP --query "[0].name" -o tsv)
    az network nsg rule update -g $RESOURCE_GROUP --nsg-name "\$NSG" -n open-port-22 --source-address-prefixes "\$MYIP"

  WHEN DONE — delete everything to stop billing (safe: dedicated RG):
    az group delete -n $RESOURCE_GROUP --yes --no-wait
============================================================
EOF
