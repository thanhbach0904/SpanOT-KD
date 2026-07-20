#!/bin/bash
# Download BLOOMZ-560M teacher model from Hugging Face
#
# Usage:
#   HF_TOKEN=<your_hf_token> bash download_bloomz560m.sh
#
# The model will be saved to $HOME/models/bloomz-560M

set -euo pipefail

export HOME=${HOME:-/workspace}
TEACHER_MODEL="bigscience/bloomz-560m"
TEACHER_LOCAL_DIR="$HOME/models/bloomz-560M"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: HF_TOKEN=<your_hf_token> bash download_bloomz560m.sh" >&2
  exit 1
fi

echo "==> Downloading BLOOMZ-560M teacher model..."
mkdir -p "$HOME/models"

# Using huggingface_hub's hf command
hf download "$TEACHER_MODEL" --local-dir "$TEACHER_LOCAL_DIR"

echo ""
echo "==> Download complete."
echo "    Teacher model saved to: $TEACHER_LOCAL_DIR"
echo "    Use this path in training with --distillation_config_model_name $TEACHER_LOCAL_DIR"
