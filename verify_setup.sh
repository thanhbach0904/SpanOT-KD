#!/bin/bash
# Verify that all setup phases completed successfully
#
# Checks:
# - Repository exists and symlink is correct
# - Python dependencies are installed
# - Teacher model is present
# - Student models are present
# - Dataset loaders exist

set -euo pipefail
export HOME=${HOME:-/workspace}

# ── Config ────────────────────────────────────────────────────────────────────
REPO_PATH="${REPO_PATH:-$HOME/SpanOT-KD}"
MODELS_PATH="$HOME/models"
# ──────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "Verifying SpanOT-KD Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

ERRORS=0
WARNINGS=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
  echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
  echo -e "${RED}✗${NC} $1"
  ((ERRORS++))
}

check_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

# ────────────────────────────────────────────────────────────────────────────
echo "1. Repository"
echo "────────────────────────────────────────────────────────────────"

if [ -d "$REPO_PATH" ]; then
  check_pass "Repository exists at $REPO_PATH"
else
  check_fail "Repository not found at $REPO_PATH"
fi

if [ -L "$HOME/Multi-Level-OT" ]; then
  TARGET=$(readlink "$HOME/Multi-Level-OT")
  if [ "$TARGET" = "$REPO_PATH" ]; then
    check_pass "Symlink $HOME/Multi-Level-OT -> $REPO_PATH is correct"
  else
    check_fail "Symlink $HOME/Multi-Level-OT points to $TARGET (expected $REPO_PATH)"
  fi
elif [ -d "$HOME/Multi-Level-OT" ]; then
  if [ "$REPO_PATH" = "$HOME/Multi-Level-OT" ]; then
    check_pass "Repo is at $HOME/Multi-Level-OT (symlink not needed)"
  else
    check_warn "Directory $HOME/Multi-Level-OT exists but is not a symlink to $REPO_PATH"
  fi
else
  if [ "$REPO_PATH" = "$HOME/Multi-Level-OT" ]; then
    check_pass "Repo at expected location $HOME/Multi-Level-OT"
  else
    check_fail "No symlink or directory at $HOME/Multi-Level-OT (needed by training code)"
  fi
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "2. Python Dependencies"
echo "────────────────────────────────────────────────────────────────"

DEPS=(
  "transformers:4.40.2"
  "datasets"
  "peft"
  "accelerate"
  "torch"
  "huggingface_hub"
  "wandb"
  "evaluate"
  "rouge_score"
)

for dep in "${DEPS[@]}"; do
  pkg="${dep%%:*}"
  expected_ver="${dep##*:}"
  if python -c "import $pkg" 2>/dev/null; then
    if [ "$expected_ver" != "$pkg" ]; then
      ver=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null)
      if [ "$ver" = "unknown" ]; then
        check_pass "$pkg is installed"
      else
        check_pass "$pkg is installed (version: $ver)"
      fi
    else
      check_pass "$pkg is installed"
    fi
  else
    check_fail "$pkg is not installed"
  fi
done

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "3. Teacher Model"
echo "────────────────────────────────────────────────────────────────"

if [ -d "$MODELS_PATH" ]; then
  check_pass "Models directory exists at $MODELS_PATH"

  TEACHERS=()
  for teacher_dir in "$MODELS_PATH"/*; do
    if [ -d "$teacher_dir" ]; then
      TEACHERS+=("$(basename "$teacher_dir")")
    fi
  done

  if [ ${#TEACHERS[@]} -gt 0 ]; then
    check_pass "Found ${#TEACHERS[@]} teacher model(s):"
    for teacher in "${TEACHERS[@]}"; do
      echo "         - $teacher"
    done
  else
    check_warn "No teacher models found in $MODELS_PATH"
  fi
else
  check_fail "Models directory not found at $MODELS_PATH"
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "4. Student Models"
echo "────────────────────────────────────────────────────────────────"

STUDENT_DIR="$REPO_PATH/EleutherAI"
if [ -d "$STUDENT_DIR" ]; then
  check_pass "Student models directory exists at $STUDENT_DIR"

  STUDENTS=()
  for student_dir in "$STUDENT_DIR"/*; do
    if [ -d "$student_dir" ]; then
      STUDENTS+=("$(basename "$student_dir")")
    fi
  done

  if [ ${#STUDENTS[@]} -gt 0 ]; then
    check_pass "Found ${#STUDENTS[@]} student model(s):"
    for student in "${STUDENTS[@]}"; do
      echo "         - $student"
    done
  else
    check_warn "No student models found in $STUDENT_DIR"
  fi
else
  check_fail "Student models directory not found at $STUDENT_DIR"
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "5. Dataset Loaders"
echo "────────────────────────────────────────────────────────────────"

LOADERS_DIR="$REPO_PATH/llm_distillation/datasets/loader"
if [ -d "$LOADERS_DIR" ]; then
  check_pass "Dataset loaders directory exists"

  LOADERS=()
  for loader in "$LOADERS_DIR"/*.py; do
    if [ -f "$loader" ]; then
      LOADERS+=("$(basename "$loader" .py)")
    fi
  done

  check_pass "Found ${#LOADERS[@]} dataset loader(s): ${LOADERS[*]}"
else
  check_fail "Dataset loaders directory not found at $LOADERS_DIR"
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "6. Benchmark Scripts"
echo "────────────────────────────────────────────────────────────────"

BENCH_DIR="$REPO_PATH/llm_distillation/benchmark"
if [ -d "$BENCH_DIR" ]; then
  check_pass "Benchmark directory exists"

  BENCHMARKS=$(find "$BENCH_DIR" -maxdepth 1 -name "benchmark*.py" -type f | wc -l)
  check_pass "Found $BENCHMARKS benchmark script(s)"
else
  check_fail "Benchmark directory not found at $BENCH_DIR"
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "7. Training Script"
echo "────────────────────────────────────────────────────────────────"

if [ -f "$REPO_PATH/finetuning.py" ]; then
  check_pass "Training script exists at $REPO_PATH/finetuning.py"
else
  check_fail "Training script not found at $REPO_PATH/finetuning.py"
fi

# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $ERRORS -eq 0 ]; then
  echo -e "${GREEN}✓ Setup verification passed!${NC}"
  echo ""
  echo "You are ready to run experiments. Example:"
  echo "  bash $REPO_PATH/run_experiments_qed.sh \\
  echo "    $MODELS_PATH/Llama-2-7b-chat-hf opt-350m 42 true"
else
  echo -e "${RED}✗ Setup verification failed with $ERRORS error(s)${NC}"
  [ $WARNINGS -gt 0 ] && echo -e "${YELLOW}⚠ Also $WARNINGS warning(s)${NC}"
  echo ""
  echo "Please fix the errors above and re-run this script."
  exit 1
fi

[ $WARNINGS -gt 0 ] && echo -e "${YELLOW}⚠ $WARNINGS warning(s) detected (non-critical)${NC}"
echo ""
