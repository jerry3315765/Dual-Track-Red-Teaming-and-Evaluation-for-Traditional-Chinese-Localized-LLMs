#!/usr/bin/env bash
set -euo pipefail

# Example environment variables for cloud GPU runs.
# Export your real keys before running.

export OPENAI_API_KEY="your_key_here"
export CUDA_VISIBLE_DEVICES="0"
