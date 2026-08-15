#!/bin/bash
# Setup Phase 1: Clone repo and install dependencies
#
# Clones the SpanOT-KD repository and installs all Python training dependencies.
# Must run before training, but does not require HF_TOKEN.
#
# Usage:
#   bash setup_repo_and_deps.sh [GITHUB_TOKEN]
#
# GITHUB_TOKEN is optional (omit to clone over HTTPS, you'll be prompted for
# credentials if the repo is private).

set -euo pipefail
export HOME=${HOME:-/workspace}

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_USER="thanhbach0904"
REPO_NAME="SpanOT-KD"
INSTALL_DIR="$HOME/SpanOT-KD"
GITHUB_TOKEN="${1:-}"
# ──────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "Phase 1: Repo and Dependencies Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Activate venv if it exists
if [ -f "/venv/main/bin/activate" ]; then
  echo "==> Activating venv at /venv/main/..."
  source /venv/main/bin/activate
else
  echo "Warning: /venv/main/bin/activate not found. Skipping venv activation."
fi

echo "==> Cloning $REPO_NAME into $INSTALL_DIR ..."
if [ -d "$INSTALL_DIR" ]; then
  echo "    (Repository already exists at $INSTALL_DIR, skipping clone)"
else
  if [ -n "$GITHUB_TOKEN" ]; then
    git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git" "$INSTALL_DIR"
  else
    git clone "https://github.com/${GITHUB_USER}/${REPO_NAME}.git" "$INSTALL_DIR"
  fi
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

echo ""
echo "==> Installing Python dependencies (training-pinned versions)..."
echo "    (This can take a while on some hosts — mostly PyPI download"
echo "     throughput, not dependency resolution. Full log: /tmp/pip_install.log)"

# NOTE: this approach re-pins transformers==4.40.2 (the version the SpanOT-KD/
# Multi-Level-OT training code is actually written against). optimum==1.17.0 is
# also required: models/models_utils.py imports `optimum.bettertransformer`
# unconditionally at module load.
#
# Installed with -v (not -q) piped through tee: on a slow/flaky vast.ai host
# a silent -q install can sit for 10+ minutes with zero output, indistinguishable
# from a resolver deadlock or a stray torch reinstall. -v makes the download vs.
# resolve phase visible live, and the log lets you grep after the fact, e.g.:
#   grep -i "Collecting torch" /tmp/pip_install.log
#   grep -ic "INFO: pip is looking at multiple versions" /tmp/pip_install.log
# (set -o pipefail above ensures a pip failure still fails this script through the pipe.)
pip uninstall -y -q optimum transformers 2>/dev/null || true
pip install -v \
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
  bert_score 2>&1 | tee /tmp/pip_install.log

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Phase 1 Complete: Repo and Dependencies"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Repository: $INSTALL_DIR"
echo "Symlink   : $HOME/Multi-Level-OT -> $INSTALL_DIR"
echo ""
echo "Next steps:"
echo "  1. Download teacher model:"
echo "     HF_TOKEN=<token> bash $INSTALL_DIR/setup_teacher.sh"
echo ""
echo "  2. Download student models:"
echo "     bash $INSTALL_DIR/setup_students.sh"
echo ""
