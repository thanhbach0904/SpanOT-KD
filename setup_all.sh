#!/bin/bash
# Master setup script: Runs all three phases (repo, teacher, students)
#
# This is a convenience wrapper that runs setup_repo_and_deps.sh,
# setup_teacher.sh, and setup_students.sh in sequence.
#
# For more control, run the individual phase scripts separately.
#
# Usage:
#   HF_TOKEN=<token> [GITHUB_TOKEN=<token>] bash setup_all.sh [teacher_id] [teacher_dir]
#
# Examples:
#   # Full setup with Llama-2-7b (default)
#   HF_TOKEN=<token> bash setup_all.sh
#
#   # Full setup with custom teacher
#   HF_TOKEN=<token> bash setup_all.sh Qwen/Qwen2-8B /workspace/models/Qwen2-8B
#
#   # Full setup with GitHub token for private repo
#   HF_TOKEN=<token> GITHUB_TOKEN=<token> bash setup_all.sh

set -euo pipefail
export HOME=${HOME:-/workspace}

# ── Config ────────────────────────────────────────────────────────────────────
REPO_PATH="$HOME/SpanOT-KD"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
TEACHER_ID="${1:-meta-llama/Llama-2-7b-chat-hf}"
TEACHER_DIR="${2:-$HOME/models/Llama-2-7b-chat-hf}"
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: HF_TOKEN=<your_hf_token> bash setup_all.sh" >&2
  exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SpanOT-KD Complete Setup (All Phases)                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Repo path       : $REPO_PATH"
echo "  Teacher model   : $TEACHER_ID"
echo "  Teacher dir     : $TEACHER_DIR"
echo "  GitHub Token    : $([ -n "$GITHUB_TOKEN" ] && echo 'set' || echo 'not set (will use HTTPS)')"
echo ""
echo "This will:"
echo "  1. Clone repo and install dependencies"
echo "  2. Download teacher model"
echo "  3. Download student models (OPT-350m, Pythia-410m)"
echo ""

# Phase 1: Repo and dependencies
echo "────────────────────────────────────────────────────────────────"
echo "Running Phase 1: Repo and Dependencies..."
echo "────────────────────────────────────────────────────────────────"
if [ -n "$GITHUB_TOKEN" ]; then
  bash "$REPO_PATH/setup_repo_and_deps.sh" "$GITHUB_TOKEN"
else
  bash "$REPO_PATH/setup_repo_and_deps.sh"
fi

# Phase 2: Teacher model
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "Running Phase 2: Teacher Model..."
echo "────────────────────────────────────────────────────────────────"
HF_TOKEN="$HF_TOKEN" bash "$REPO_PATH/setup_teacher.sh" "$TEACHER_ID" "$TEACHER_DIR"

# Phase 3: Student models
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "Running Phase 3: Student Models..."
echo "────────────────────────────────────────────────────────────────"
bash "$REPO_PATH/setup_students.sh"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  Setup Complete!                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Installed:"
echo "  ✓ Repository     : $REPO_PATH"
echo "  ✓ Dependencies   : transformers, datasets, peft, accelerate, etc."
echo "  ✓ Teacher model  : $TEACHER_DIR"
echo "  ✓ Student models : $REPO_PATH/EleutherAI/"
echo ""
echo "Next steps:"
echo "  1. Verify all components: bash $REPO_PATH/verify_setup.sh"
echo "  2. Download additional teachers (optional):"
echo "     HF_TOKEN=<token> bash $REPO_PATH/download_qwen8b.sh"
echo "     HF_TOKEN=<token> bash $REPO_PATH/download_bloomz560m.sh"
echo "  3. Run an experiment: bash $REPO_PATH/run_experiments_qed.sh $TEACHER_DIR opt-350m 42"
echo ""
