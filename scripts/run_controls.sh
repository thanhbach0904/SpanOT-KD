#!/bin/bash
# WP9: resumable orchestration of the control arms on vast.ai.
#
# Usage:
#   bash scripts/run_controls.sh <teacher_path> <student> <dataset: qed|fairytaleqa> "<seeds>" ["<arms>"]
# Example:
#   bash scripts/run_controls.sh /workspace/models/Llama-2-7b-chat-hf opt-350m qed "42 4 63"
#   bash scripts/run_controls.sh /workspace/models/Llama-2-7b-chat-hf opt-350m qed "42 4 63" "matched"
#
# Arms run in priority order (default: matched random true false). An (arm, seed)
# is SKIPPED when its final eval output (<eval_dir>_loss/0shots.json, written by
# the last step of run_experiments_<dataset>.sh) already exists, so re-running
# after an instance death resumes. Granularity is the whole arm/seed job: a job
# killed mid-sweep restarts from r=0.3.
#
# Per job: <output_dir>/run_meta.json (git commit, dirty flag, exact command,
# timestamps, exit code) and one line in experiments/controls_status.csv.
# Env vars (SPAN_TOP_R_EXTRA, MAX_SAMPLES, SKIP_EVAL) are passed through.

set -uo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: bash scripts/run_controls.sh <teacher_path> <student> <qed|fairytaleqa> \"<seeds>\" [\"<arms>\"]" >&2
  exit 1
fi
TEACHER="$1"; STUDENT="$2"; DATASET="$3"; SEEDS="$4"; ARMS="${5:-matched random true false}"
case "$DATASET" in qed|fairytaleqa) ;; *) echo "ERROR: dataset must be qed|fairytaleqa" >&2; exit 1 ;; esac
for A in $ARMS; do
  case "$A" in true|random|matched|false) ;; *) echo "ERROR: unknown arm '$A' (true|random|matched|false)" >&2; exit 1 ;; esac
done
if [ "${MAX_SAMPLES:-0}" != "0" ] || [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "WARNING: MAX_SAMPLES/SKIP_EVAL set — dry-run mode; jobs never produce the final eval file and are never marked done." >&2
fi

export HOME=${HOME:-/workspace}
REPO_PATH="$HOME/SpanOT-KD"
SCRIPT="$REPO_PATH/run_experiments_${DATASET}.sh"
STATUS_CSV="$REPO_PATH/experiments/controls_status.csv"
mkdir -p "$REPO_PATH/experiments"
[ -f "$STATUS_CSV" ] || echo "dataset,teacher,student,arm,seed,status,exit_code,run_dir,git_commit,git_dirty,timestamp" > "$STATUS_CSV"

GIT_COMMIT="$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$([ -n "$(git -C "$REPO_PATH" status --porcelain 2>/dev/null)" ] && echo true || echo false)"
TEACHER_TAG="$(basename "${TEACHER%/}")"

method_tag() {
  case "$1" in
    true) echo spanotkd ;; random) echo randomspan ;; matched) echo matchedweight ;; false) echo vanilla ;;
  esac
}

for ARM in $ARMS; do
  for SEED in $SEEDS; do
    RUN_TAG="${STUDENT%%-*}_${TEACHER_TAG}_$(method_tag "$ARM")_seed${SEED}"
    [ "${MAX_SAMPLES:-0}" != "0" ] && RUN_TAG="${RUN_TAG}_dryrun"
    RUN_DIR="$REPO_PATH/output_${DATASET}_${RUN_TAG}"
    DONE_FILE="$REPO_PATH/eval_results/${DATASET}_${RUN_TAG}_loss/0shots.json"
    if [ -f "$DONE_FILE" ]; then
      echo "[controls] SKIP $ARM seed=$SEED (found $DONE_FILE)"
      continue
    fi
    CMD="bash $SCRIPT $TEACHER $STUDENT $SEED $ARM"
    START="$(date -Iseconds)"
    mkdir -p "$RUN_DIR"
    cat > "$RUN_DIR/run_meta.json" <<EOF
{"arm": "$ARM", "seed": $SEED, "dataset": "$DATASET", "teacher": "$TEACHER", "student": "$STUDENT",
 "command": "$CMD", "env": {"SPAN_TOP_R_EXTRA": "${SPAN_TOP_R_EXTRA:-}", "MAX_SAMPLES": "${MAX_SAMPLES:-0}", "SKIP_EVAL": "${SKIP_EVAL:-0}"},
 "git_commit": "$GIT_COMMIT", "git_dirty": $GIT_DIRTY, "started": "$START", "finished": null, "exit_code": null}
EOF
    echo "[controls] RUN  $ARM seed=$SEED -> $RUN_DIR"
    $CMD 2>&1 | tee "$RUN_DIR/controls_stdout.log"
    RC=${PIPESTATUS[0]}
    END="$(date -Iseconds)"
    sed -i "s/\"finished\": null, \"exit_code\": null/\"finished\": \"$END\", \"exit_code\": $RC/" "$RUN_DIR/run_meta.json"
    STATUS=$([ "$RC" = "0" ] && echo done || echo failed)
    echo "$DATASET,$TEACHER_TAG,$STUDENT,$ARM,$SEED,$STATUS,$RC,$RUN_DIR,$GIT_COMMIT,$GIT_DIRTY,$END" >> "$STATUS_CSV"
    # A failed job does not stop the loop: later (arm, seed) cells still run,
    # and the failure is visible in the CSV and run_meta.json.
    [ "$RC" = "0" ] || echo "[controls] FAILED $ARM seed=$SEED (exit $RC) — continuing" >&2
  done
done
