#!/usr/bin/env bash
# Paper-scale run on a CUDA node.
#
#   bash scripts/run_full.sh [config]
#
# Requires data/zsre (see docs/DATASET.md) and requirements-gpu.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/paper_full.yaml}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dlmem-mpl"
export TOKENIZERS_PARALLELISM=false

python scripts/check_gpu.py --config "$CONFIG" || {
  echo "GPU preflight failed; fix the environment before starting." >&2
  exit 1
}

python -m dlmem.cli prepare  --config "$CONFIG"
python -m dlmem.cli train    --config "$CONFIG"
python -m dlmem.cli evaluate --config "$CONFIG"
python -m dlmem.cli sweep    --config "$CONFIG"
python -m dlmem.cli ablate   --config "$CONFIG"
python -m dlmem.cli sft      --config "$CONFIG"

echo
echo "Results: runs/paper_full/results/table1.md (or the run_name in $CONFIG)"
