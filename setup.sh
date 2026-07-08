#!/bin/bash
# One-shot setup for SpanOT-KD on Vast.ai
# Merges config_steps.md (env activation, HF login, teacher model download)
# and the former setup.sh (repo clone, student models, training deps).
#
# Usage:
#   HF_TOKEN=<hf_token> GITHUB_TOKEN=<github_pat> bash setup.sh
#
# HF_TOKEN is required (teacher model download is gated). GITHUB_TOKEN is
# optional — omit it to clone over plain HTTPS (you'll be prompted for
# credentials if the repo is private).
set -euo pipefail
export HOME=/workspace

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_USER="thanhbach0904"
REPO_NAME="SpanOT-KD"
INSTALL_DIR="$HOME/SpanOT-KD"

TEACHER_MODEL="meta-llama/Llama-2-7b-chat-hf"
TEACHER_LOCAL_DIR="$HOME/models/Llama-2-7b-chat-hf"

STUDENT_MODELS=(
  "facebook/opt-350m|EleutherAI/opt-350m"
  "EleutherAI/pythia-410m|EleutherAI/pythia-410m"
)
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: HF_TOKEN=<your_hf_token> bash setup.sh" >&2
  exit 1
fi

source /venv/main/bin/activate

echo "==> Installing huggingface_hub and logging in..."
pip install -q -U huggingface_hub
hf auth login --token "$HF_TOKEN"

echo "==> Downloading teacher model: $TEACHER_MODEL ..."
mkdir -p "$HOME/models"
hf download "$TEACHER_MODEL" --local-dir "$HOME/models/meta-llama/Llama-2-7b-chat-hf"
mv "$HOME/models/meta-llama/Llama-2-7b-chat-hf" "$TEACHER_LOCAL_DIR"

# NOTE: this pin is intentionally transient for transformers — the
# "Installing Python dependencies" block further down re-pins
# transformers==4.40.2 (the version the SpanOT-KD/Multi-Level-OT training
# code is actually written against), which overrides it. optimum==1.17.0
# is NOT a no-op though: the final block used to uninstall optimum here and
# never reinstall it, even though models/models_utils.py imports
# `optimum.bettertransformer` unconditionally at module load. Re-pinned
# below so the final environment actually has it.
echo "==> Transient optimum/transformers pin (transformers pin overridden below, optimum pin kept)..."
pip uninstall -y -q optimum transformers
pip install -q optimum==1.17.0 transformers==4.48.0

echo "==> Cloning $REPO_NAME into $INSTALL_DIR ..."
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git" "$INSTALL_DIR"
else
  git clone "https://github.com/${GITHUB_USER}/${REPO_NAME}.git" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# The training code (llm_distillation/, train/, etc.) hardcodes
# "$HOME/Multi-Level-OT/..." in ~20+ files (benchmark scripts, dataset
# loaders, prompt.py) instead of deriving paths from the repo's actual
# location. Since this repo is cloned as "$REPO_NAME" ($INSTALL_DIR), not
# "Multi-Level-OT", every one of those hardcoded paths would 404. Symlink
# so both names resolve to the same checkout, rather than rewriting every
# call site.
if [ "$INSTALL_DIR" != "$HOME/Multi-Level-OT" ]; then
  echo "==> Symlinking $HOME/Multi-Level-OT -> $INSTALL_DIR (training code hardcodes this path)..."
  ln -sfn "$INSTALL_DIR" "$HOME/Multi-Level-OT"
fi

echo "==> Installing Python dependencies (final, training-pinned versions)..."
pip uninstall -y -q optimum transformers
pip install -q \
  transformers==4.40.2 \
  optimum==1.17.0 \
  datasets \
  peft==0.10.0 \
  accelerate \
  evaluate \
  rouge_score \
  sentencepiece \
  protobuf \
  scipy \
  matplotlib \
  huggingface_hub \
  wandb \
  scikit-learn \
  bert_score

echo "==> Downloading student models into EleutherAI/..."
mkdir -p EleutherAI

python - <<PYEOF
from huggingface_hub import snapshot_download

models = [
$(for pair in "${STUDENT_MODELS[@]}"; do
    hf_id="${pair%%|*}"
    local_dir="${pair##*|}"
    echo "    (\"$hf_id\", \"$local_dir\"),"
  done)
]
for hf_id, local_dir in models:
    print(f"  Downloading {hf_id} -> {local_dir}")
    snapshot_download(repo_id=hf_id, local_dir=local_dir, ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"])
PYEOF

echo ""
echo "==> Setup complete."
echo "    Repo      : $INSTALL_DIR  (github.com/${GITHUB_USER}/${REPO_NAME})"
echo "    Models    : $INSTALL_DIR/EleutherAI/"
echo "    Teacher   : $TEACHER_LOCAL_DIR"
echo ""
echo "    Next step : run a command from README.md"
