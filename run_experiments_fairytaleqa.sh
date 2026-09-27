#!/bin/bash
# FairyTaleQA Dataset Experiments: Train and evaluate OPT-350m and Pythia-410m
# students distilled from Llama-2-7b-chat-hf, Qwen-7B-Chat, or BLOOMZ-560M
#
# Prerequisite (once per teacher): generate that teacher's labels, e.g.
#   python prepare_fairytaleqa_dataset.py --batch_size 8 --teacher_path /workspace/models/Qwen-7B-Chat
#
# Usage:
#   bash run_experiments_fairytaleqa.sh <teacher_model_path> <student_model> <seed> [span_kd_enabled]
#
# Example:
#   bash run_experiments_fairytaleqa.sh /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
#   bash run_experiments_fairytaleqa.sh /workspace/models/Qwen-7B-Chat opt-350m 63 false
#   bash run_experiments_fairytaleqa.sh /workspace/models/bloomz-560M opt-350m 42 true

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: bash run_experiments_fairytaleqa.sh <teacher_model_path> <student_model> <seed> [span_kd_enabled]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  teacher_model_path  : Path to teacher model (e.g., /workspace/models/Llama-2-7b-chat-hf)" >&2
  echo "  student_model       : opt-350m or pythia-410m" >&2
  echo "  seed                : Random seed for reproducibility" >&2
  echo "  span_kd_enabled     : true|false (default: true) - Enable SpanOT-KD reweighting" >&2
  exit 1
fi

export HOME=${HOME:-/workspace}
TEACHER_MODEL_PATH="$1"
STUDENT_MODEL="$2"
SEED="$3"
SPAN_KD_ENABLED="${4:-true}"

# Validate inputs
if [ ! -d "$TEACHER_MODEL_PATH" ]; then
  echo "ERROR: Teacher model path does not exist: $TEACHER_MODEL_PATH" >&2
  exit 1
fi

case "$STUDENT_MODEL" in
  opt-350m|pythia-410m)
    ;;
  *)
    echo "ERROR: Invalid student model. Must be 'opt-350m' or 'pythia-410m'" >&2
    exit 1
    ;;
esac

REPO_PATH="$HOME/SpanOT-KD"
STUDENT_PATH="$REPO_PATH/EleutherAI/$STUDENT_MODEL"
DATASET_FILE="$REPO_PATH/llm_distillation/datasets/loader/fairytaleQA.py"

# Labels (answers_generated) are the teacher's own generations, so both
# training (loader keys on the teacher basename) and the eval references below
# must come from THIS teacher's folder. Generate it with:
#   python prepare_fairytaleqa_dataset.py --teacher_path <teacher_model_path>
TEACHER_TAG="$(basename "${TEACHER_MODEL_PATH%/}")"
DATASET_ID="$REPO_PATH/llm_distillation/datasets/hf/uld_loss_${TEACHER_TAG}-FairytaleQA/fairytaleQA"
if [ ! -d "$DATASET_ID" ]; then
  echo "ERROR: No teacher-labelled dataset for $TEACHER_TAG at: $DATASET_ID" >&2
  echo "       Run: python prepare_fairytaleqa_dataset.py --teacher_path $TEACHER_MODEL_PATH" >&2
  exit 1
fi

# Output/eval dirs previously encoded only student+seed, so different teachers
# (and vanilla vs SpanOT-KD) overwrote each other's checkpoints and predictions.
# Same tagging as run_experiments_qed.sh.
if [ "$SPAN_KD_ENABLED" = "true" ]; then
  METHOD_TAG="spanotkd"
else
  METHOD_TAG="vanilla"
fi
RUN_TAG="${STUDENT_MODEL%%-*}_${TEACHER_TAG}_${METHOD_TAG}_seed${SEED}"
OUTPUT_DIR="$REPO_PATH/output_fairytaleqa_${RUN_TAG}"
EVAL_DIR="$REPO_PATH/eval_results/fairytaleqa_${RUN_TAG}"

echo "════════════════════════════════════════════════════════════════"
echo "FairyTaleQA Dataset Experiments"
echo "════════════════════════════════════════════════════════════════"
echo "Teacher Model    : $TEACHER_MODEL_PATH"
echo "Student Model    : $STUDENT_MODEL"
echo "Student Path     : $STUDENT_PATH"
echo "Seed             : $SEED"
echo "SpanOT-KD        : $SPAN_KD_ENABLED"
echo "Dataset          : FairyTaleQA ($DATASET_ID)"
echo "Output Dir       : $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Build training command
TRAIN_CMD="python $REPO_PATH/finetuning.py \
  --model_name $STUDENT_PATH \
  --dataset.file $DATASET_FILE \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $OUTPUT_DIR \
  --distillation_config_model_name $TEACHER_MODEL_PATH \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed $SEED \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --seed $SEED"

if [ "$SPAN_KD_ENABLED" = "true" ]; then
  TRAIN_CMD="$TRAIN_CMD \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation mean \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep"
fi

echo "[1/3] Training..."
CUDA_VISIBLE_DEVICES=0 eval "$TRAIN_CMD"

echo ""
echo "[2/3] Evaluating (best_dev_rouge_l checkpoint)..."
EVAL_CMD_ROUGE_L="python $REPO_PATH/llm_distillation/benchmark/benchmarkfairytaleQAbasellama.py \
  --model_id $OUTPUT_DIR/best_dev_rouge_l \
  --model_tokenizer $STUDENT_PATH \
  --dataset_id $DATASET_ID \
  --split_name validation \
  --mapping $REPO_PATH/llm_distillation/benchmark/mapping/fairytaleqa_uld_loss.json \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path $EVAL_DIR/"

CUDA_VISIBLE_DEVICES=0 eval "$EVAL_CMD_ROUGE_L"

echo ""
echo "[3/3] Evaluating (best_dev_loss checkpoint)..."
EVAL_CMD_LOSS="python $REPO_PATH/llm_distillation/benchmark/benchmarkfairytaleQAbasellama.py \
  --model_id $OUTPUT_DIR/best_dev_loss \
  --model_tokenizer $STUDENT_PATH \
  --dataset_id $DATASET_ID \
  --split_name validation \
  --mapping $REPO_PATH/llm_distillation/benchmark/mapping/fairytaleqa_uld_loss.json \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path ${EVAL_DIR}_loss/"

CUDA_VISIBLE_DEVICES=0 eval "$EVAL_CMD_LOSS"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "FairyTaleQA Experiments Complete"
echo "Results saved to:"
echo "  - $EVAL_DIR/"
echo "  - ${EVAL_DIR}_loss/"
echo "════════════════════════════════════════════════════════════════"
