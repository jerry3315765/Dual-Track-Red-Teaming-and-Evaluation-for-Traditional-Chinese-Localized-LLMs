#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

python scripts/run_single_model.py --model_name "${MODEL_NAME}"
