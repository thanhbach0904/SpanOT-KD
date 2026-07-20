#!/bin/bash
# Download Qwen8B teacher model from Hugging Face
#
# Usage:
#   HF_TOKEN=<your_hf_token> bash download_qwen8b.sh
#
# The model will be saved to $HOME/models/Qwen2-8B or the path specified by --output-dir

set -euo pipefail

export HOME=${HOME:-/workspace}
TEACHER_MODEL="Qwen/Qwen2-8B"
TEACHER_LOCAL_DIR="$HOME/models/Qwen2-8B"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: HF_TOKEN=<your_hf_token> bash download_qwen8b.sh" >&2
  exit 1
fi

echo "==> Downloading Qwen2-8B teacher model..."
mkdir -p "$HOME/models"

# Using huggingface_hub's hf command
hf download "$TEACHER_MODEL" --local-dir "$TEACHER_LOCAL_DIR"

echo ""
echo "==> Download complete."
echo "    Teacher model saved to: $TEACHER_LOCAL_DIR"
echo "    Use this path in training with --distillation_config_model_name $TEACHER_LOCAL_DIR"
