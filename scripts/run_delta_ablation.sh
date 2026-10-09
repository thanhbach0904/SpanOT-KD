#!/bin/bash
# Resumable span_low_delta ablation on QED at a fixed top-r (no r sweep).
#
# Usage:
#   bash scripts/run_delta_ablation.sh <teacher_path> <student> "<seeds>" [fixed_r] ["<deltas>"]
# Example:
#   bash scripts/run_delta_ablation.sh /workspace/models/Llama-2-7b-chat-hf opt-350m "42 63 94"
#   bash scripts/run_delta_ablation.sh /workspace/models/Llama-2-7b-chat-hf opt-350m "42" 0.5 "0 0.25"
#
# Defaults: fixed_r=0.5 (config default, chosen a priori, not from test scores),
# deltas="0 0.1 0.25 0.5 0.75 1". delta=1 makes every weight 1.0 for any r, i.e.
# vanilla MLOT (tests/test_span_delta_ablation.py::test_delta1_equals_vanilla),
# so it runs arm 'false' instead of training an identical span run.
#
# A (delta, seed) job is SKIPPED when its final eval file exists, so re-running
# after an instance death resumes. Per job: <output_dir>/run_meta.json and one
# line in experiments/delta_ablation_status.csv. MAX_SAMPLES / SKIP_EVAL pass through.

set -uo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: bash scripts/run_delta_ablation.sh <teacher_path> <student> \"<seeds>\" [fixed_r] [\"<deltas>\"]" >&2
  exit 1
fi
TEACHER="$1"; STUDENT="$2"; SEEDS="$3"; FIXED_R="${4:-0.5}"; DELTAS="${5:-0 0.1 0.25 0.5 0.75 1}"
if [ "${MAX_SAMPLES:-0}" != "0" ] || [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "WARNING: MAX_SAMPLES/SKIP_EVAL set — dry-run mode; jobs never produce the final eval file and are never marked done." >&2
fi

export HOME=${HOME:-/workspace}
REPO_PATH="$HOME/SpanOT-KD"
SCRIPT="$REPO_PATH/run_experiments_qed.sh"
STATUS_CSV="$REPO_PATH/experiments/delta_ablation_status.csv"
mkdir -p "$REPO_PATH/experiments"
[ -f "$STATUS_CSV" ] || echo "teacher,student,delta,fixed_r,seed,status,exit_code,run_dir,git_commit,git_dirty,timestamp" > "$STATUS_CSV"

GIT_COMMIT="$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$([ -n "$(git -C "$REPO_PATH" status --porcelain 2>/dev/null)" ] && echo true || echo false)"
TEACHER_TAG="$(basename "${TEACHER%/}")"

for DELTA in $DELTAS; do
  for SEED in $SEEDS; do
    # Must mirror the RUN_TAG logic in run_experiments_qed.sh.
    if [ "$DELTA" = "1" ] || [ "$DELTA" = "1.0" ]; then
      ARM=false; METHOD_TAG="vanilla"; CMD_ENV=""; CMD_R=""
    else
      ARM=true; METHOD_TAG="spanotkd"; CMD_ENV="LOW_DELTA=$DELTA"; CMD_R="$FIXED_R"
      [ "$DELTA" != "0.1" ] && METHOD_TAG="${METHOD_TAG}_d${DELTA}"
      METHOD_TAG="${METHOD_TAG}_r${FIXED_R}"
    fi
    RUN_TAG="${STUDENT%%-*}_${TEACHER_TAG}_${METHOD_TAG}_seed${SEED}"
    [ "${MAX_SAMPLES:-0}" != "0" ] && RUN_TAG="${RUN_TAG}_dryrun"
    RUN_DIR="$REPO_PATH/output_qed_${RUN_TAG}"
    DONE_FILE="$REPO_PATH/eval_results/qed_${RUN_TAG}_loss/0shots.json"
    if [ -f "$DONE_FILE" ]; then
      echo "[delta] SKIP delta=$DELTA seed=$SEED (found $DONE_FILE — check its run_meta.json is from the current code)"
      continue
    fi
    CMD="bash $SCRIPT $TEACHER $STUDENT $SEED $ARM $CMD_R"
    START="$(date -Iseconds)"
    mkdir -p "$RUN_DIR"
    cat > "$RUN_DIR/run_meta.json" <<EOF
{"arm": "$ARM", "delta": "$DELTA", "fixed_r": "$CMD_R", "seed": $SEED, "dataset": "qed", "teacher": "$TEACHER", "student": "$STUDENT",
 "command": "$CMD_ENV $CMD", "env": {"MAX_SAMPLES": "${MAX_SAMPLES:-0}", "SKIP_EVAL": "${SKIP_EVAL:-0}"},
 "git_commit": "$GIT_COMMIT", "git_dirty": $GIT_DIRTY, "started": "$START", "finished": null, "exit_code": null}
EOF
    echo "[delta] RUN  delta=$DELTA seed=$SEED -> $RUN_DIR"
    if [ "$ARM" = "true" ]; then
      LOW_DELTA="$DELTA" $CMD 2>&1 | tee "$RUN_DIR/delta_stdout.log"
    else
      $CMD 2>&1 | tee "$RUN_DIR/delta_stdout.log"
    fi
    RC=${PIPESTATUS[0]}
    END="$(date -Iseconds)"
    sed -i "s/\"finished\": null, \"exit_code\": null/\"finished\": \"$END\", \"exit_code\": $RC/" "$RUN_DIR/run_meta.json"
    STATUS=$([ "$RC" = "0" ] && echo done || echo failed)
    echo "$TEACHER_TAG,$STUDENT,$DELTA,$CMD_R,$SEED,$STATUS,$RC,$RUN_DIR,$GIT_COMMIT,$GIT_DIRTY,$END" >> "$STATUS_CSV"
    [ "$RC" = "0" ] || echo "[delta] FAILED delta=$DELTA seed=$SEED (exit $RC) — continuing" >&2
  done
done
