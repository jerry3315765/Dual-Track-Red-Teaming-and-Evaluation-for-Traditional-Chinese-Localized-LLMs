#!/usr/bin/env bash
set -euo pipefail

# Example: launch an OpenAI-compatible server (vLLM) for safetensors models.
# Replace MODEL_PATH and GPU settings for your cloud environment.

MODEL_PATH="/models/your-model"  # safetensors model folder
TP_SIZE=1
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len 8192

# Then set models.yaml base_url to http://localhost:${PORT}/v1
