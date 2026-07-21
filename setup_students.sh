#!/bin/bash
# Setup Phase 3: Download student models
#
# Downloads student models from Hugging Face. Defaults to OPT-350m and Pythia-410m,
# but can be customized with STUDENT_MODELS environment variable.
#
# Usage:
#   bash setup_students.sh [model_pairs...]
#
# Environment:
#   STUDENT_MODELS : Space-separated list of "hf_id|local_dir" pairs
#                    (default: "facebook/opt-350m|EleutherAI/opt-350m EleutherAI/pythia-410m|EleutherAI/pythia-410m")
#
# Examples:
#   # Default (OPT-350m + Pythia-410m)
#   bash setup_students.sh
#
#   # Custom students
#   bash setup_students.sh "facebook/opt-125m|EleutherAI/opt-125m" "EleutherAI/pythia-70m|EleutherAI/pythia-70m"
#
#   # Override via environment
#   STUDENT_MODELS="facebook/opt-1.3b|EleutherAI/opt-1.3b" bash setup_students.sh

set -euo pipefail
export HOME=${HOME:-/workspace}

# ── Config ────────────────────────────────────────────────────────────────────
REPO_PATH="${REPO_PATH:-$HOME/SpanOT-KD}"

# Default student models (HF ID | local directory)
DEFAULT_MODELS=(
  "facebook/opt-350m|EleutherAI/opt-350m"
  "EleutherAI/pythia-410m|EleutherAI/pythia-410m"
)

# Allow override via STUDENT_MODELS env var or command line args
if [ $# -gt 0 ]; then
  STUDENT_MODELS=("$@")
elif [ -n "${STUDENT_MODELS:-}" ]; then
  read -ra STUDENT_MODELS <<<"$STUDENT_MODELS"
else
  STUDENT_MODELS=("${DEFAULT_MODELS[@]}")
fi
# ──────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "Phase 3: Student Models Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Repo path       : $REPO_PATH"
echo "Student models  : ${#STUDENT_MODELS[@]}"
echo ""

if [ ! -d "$REPO_PATH" ]; then
  echo "ERROR: Repository not found at $REPO_PATH" >&2
  echo "Did you run setup_repo_and_deps.sh first?" >&2
  exit 1
fi

cd "$REPO_PATH"

# Activate venv if it exists
if [ -f "/venv/main/bin/activate" ]; then
  source /venv/main/bin/activate
fi

echo "==> Ensuring huggingface_hub is installed..."
pip install -q -U huggingface_hub

echo ""
echo "==> Downloading student models into $REPO_PATH/EleutherAI/..."
mkdir -p EleutherAI

# Build Python script to download all models in parallel
python - <<PYEOF
from huggingface_hub import snapshot_download
import sys

models = [
$(for pair in "${STUDENT_MODELS[@]}"; do
    hf_id="${pair%%|*}"
    local_dir="${pair##*|}"
    echo "    (\"$hf_id\", \"$local_dir\"),"
  done)
]

if not models:
    print("ERROR: No student models specified", file=sys.stderr)
    sys.exit(1)

for i, (hf_id, local_dir) in enumerate(models, 1):
    print(f"  [{i}/{len(models)}] Downloading {hf_id} -> {local_dir}")
    try:
        snapshot_download(
            repo_id=hf_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"]
        )
        print(f"      ✓ Complete")
    except Exception as e:
        print(f"      ✗ Failed: {e}", file=sys.stderr)
        sys.exit(1)
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Phase 3 Complete: Student Models"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Student models downloaded to: $REPO_PATH/EleutherAI/"
echo ""
for pair in "${STUDENT_MODELS[@]}"; do
  local_dir="${pair##*|}"
  echo "  - $local_dir"
done
echo ""
echo "Verify with: ls -la $REPO_PATH/EleutherAI/"
echo ""
