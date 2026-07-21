#!/bin/bash
# Setup Phase 2: Download teacher model
#
# Downloads a teacher model from Hugging Face. Defaults to Llama-2-7b-chat-hf,
# but can download any HF model. Requires HF_TOKEN for gated models.
#
# Usage:
#   HF_TOKEN=<token> bash setup_teacher.sh [model_id] [local_dir]
#
# Arguments:
#   model_id     : Hugging Face model ID (default: meta-llama/Llama-2-7b-chat-hf)
#   local_dir    : Local directory to save model (default: $HOME/models/Llama-2-7b-chat-hf)
#
# Examples:
#   HF_TOKEN=<token> bash setup_teacher.sh
#   HF_TOKEN=<token> bash setup_teacher.sh Qwen/Qwen2-8B /workspace/models/Qwen2-8B
#   HF_TOKEN=<token> bash setup_teacher.sh bigscience/bloomz-560m /workspace/models/bloomz-560M

set -euo pipefail
export HOME=${HOME:-/workspace}

# ── Config ────────────────────────────────────────────────────────────────────
# Default: Llama-2-7b-chat-hf
MODEL_ID="${1:-meta-llama/Llama-2-7b-chat-hf}"
LOCAL_DIR="${2:-$HOME/models/Llama-2-7b-chat-hf}"
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: HF_TOKEN=<your_hf_token> bash setup_teacher.sh" >&2
  exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "Phase 2: Teacher Model Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Model ID    : $MODEL_ID"
echo "Local path  : $LOCAL_DIR"
echo ""

# Activate venv if it exists
if [ -f "/venv/main/bin/activate" ]; then
  source /venv/main/bin/activate
fi

echo "==> Ensuring huggingface_hub is installed and authenticated..."
pip install -q -U huggingface_hub
hf auth login --token "$HF_TOKEN"

echo ""
echo "==> Downloading teacher model: $MODEL_ID ..."
mkdir -p "$HOME/models"

# Download to a temp directory first, then move to final location
# (This avoids conflicts if the model ID's path structure doesn't match local_dir)
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

hf download "$MODEL_ID" --local-dir "$TEMP_DIR/$MODEL_ID"
mkdir -p "$(dirname "$LOCAL_DIR")"
mv "$TEMP_DIR/$MODEL_ID" "$LOCAL_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Phase 2 Complete: Teacher Model"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Teacher model saved to: $LOCAL_DIR"
echo ""
echo "Use this path in training with:"
echo "  --distillation_config_model_name $LOCAL_DIR"
echo ""
