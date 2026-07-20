# Quick Start: SpanOT-KD Experiments

This guide walks you through downloading teacher models and running experiments on QED, FairyTaleQA, and DialogSum.

## Step 1: Download Teacher Models

### Download Qwen2-8B

```bash
HF_TOKEN=your_token bash $HOME/SpanOT-KD/download_qwen8b.sh
```

Model saved to: `$HOME/models/Qwen2-8B`

### Download BLOOMZ-560M

```bash
HF_TOKEN=your_token bash $HOME/SpanOT-KD/download_bloomz560m.sh
```

Model saved to: `$HOME/models/bloomz-560M`

### Verify Teacher Models

```bash
ls -la $HOME/models/
# Should show: Llama-2-7b-chat-hf, Qwen2-8B, bloomz-560M
```

---

## Step 2: Run Experiments

### QED Dataset Examples

```bash
# OPT-350m with Qwen2-8B (SpanOT-KD enabled)
bash $HOME/SpanOT-KD/run_experiments_qed.sh \
  /workspace/models/Qwen2-8B opt-350m 42 true

# Pythia-410m with BLOOMZ-560M (baseline, no SpanOT-KD)
bash $HOME/SpanOT-KD/run_experiments_qed.sh \
  /workspace/models/bloomz-560M pythia-410m 4 false

# OPT-350m with Llama-2-7b (SpanOT-KD enabled)
bash $HOME/SpanOT-KD/run_experiments_qed.sh \
  /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
```

### FairyTaleQA Dataset Examples

```bash
# OPT-350m with Qwen2-8B (SpanOT-KD enabled)
bash $HOME/SpanOT-KD/run_experiments_fairytaleqa.sh \
  /workspace/models/Qwen2-8B opt-350m 42 true

# Pythia-410m with BLOOMZ-560M (baseline)
bash $HOME/SpanOT-KD/run_experiments_fairytaleqa.sh \
  /workspace/models/bloomz-560M pythia-410m 4 false
```

### DialogSum Dataset Examples

```bash
# OPT-350m with Qwen2-8B (SpanOT-KD enabled)
bash $HOME/SpanOT-KD/run_experiments_dialogsum.sh \
  /workspace/models/Qwen2-8B opt-350m 42 true

# Pythia-410m with BLOOMZ-560M (baseline)
bash $HOME/SpanOT-KD/run_experiments_dialogsum.sh \
  /workspace/models/bloomz-560M pythia-410m 4 false
```

---

## Step 3: Monitor Results

Each experiment script runs 3 phases:
1. **Training** – distills student from teacher
2. **Evaluation (F1)** – evaluates best_dev_f1 checkpoint
3. **Evaluation (Loss)** – evaluates best_dev_loss checkpoint

Results are saved to:
```
$HOME/SpanOT-KD/eval_results/
├── qed_opt_teacher_seed42/
├── qed_pythia_teacher_seed4_loss/
├── fairytaleqa_opt_teacher_seed42/
├── dialogsum_pythia_teacher_seed4_loss/
└── ...
```

Each result directory contains:
- `predictions.json` – Model predictions on test set
- `metrics.json` – F1, EM, BLEU, ROUGE scores
- `results.txt` – Human-readable summary

---

## Step 4: Compare Results

Check metrics across teachers:

```bash
# Compare QED results
for dir in $HOME/SpanOT-KD/eval_results/qed_*; do
  echo "=== $(basename $dir) ==="
  cat "$dir/metrics.json" | jq '.f1, .em'
done

# Compare across datasets
grep -r "f1" $HOME/SpanOT-KD/eval_results/*/metrics.json
```

---

## Script Arguments

All three experiment scripts follow the same signature:

```bash
bash run_experiments_<dataset>.sh <teacher_path> <student_model> <seed> [span_kd_enabled]
```

| Argument | Options |
|----------|---------|
| `<teacher_path>` | `/workspace/models/Llama-2-7b-chat-hf`, `/workspace/models/Qwen2-8B`, `/workspace/models/bloomz-560M` |
| `<student_model>` | `opt-350m`, `pythia-410m` |
| `<seed>` | Any integer (e.g., 42, 4, 2024) |
| `[span_kd_enabled]` | `true` or `false` (default: `true`) |

---

## What Each Script Does

### `download_qwen8b.sh` / `download_bloomz560m.sh`
- Authenticates with Hugging Face using `HF_TOKEN`
- Downloads the teacher model
- Saves to `$HOME/models/`

### `run_experiments_qed.sh` / `run_experiments_fairytaleqa.sh` / `run_experiments_dialogsum.sh`
1. **Train**: Distills student from teacher with optional SpanOT-KD
   - Saves checkpoints (best_dev_f1, best_dev_loss)
   - Logs training curves to W&B (if `WANDB_API_KEY` set)
2. **Evaluate (F1)**: Runs best_dev_f1 checkpoint on test set
   - Saves predictions and metrics
3. **Evaluate (Loss)**: Runs best_dev_loss checkpoint on test set
   - Saves predictions and metrics (comparison only)

---

## Troubleshooting

### "Teacher model path does not exist"
```bash
# Verify teacher was downloaded
ls -la $HOME/models/Qwen2-8B
ls -la $HOME/models/bloomz-560M
```

### "Dataset not found"
```bash
# Check dataset is in the processed directory
ls -la $HOME/SpanOT-KD/llm_distillation/datasets/processed/
```

### "Student model not found"
```bash
# Verify student models from setup.sh
ls -la $HOME/SpanOT-KD/EleutherAI/
```

### Out of memory
- Reduce `--batch_size_training` (e.g., 2 → 1)
- Edit the `run_experiments_*.sh` script to pass custom batch size
- Or run individual commands manually

### GPU not found
- Check `CUDA_VISIBLE_DEVICES=0` (or `=1`, `=2`, etc.)
- Or modify the script to use a different GPU

---

## Full Experiment Matrix

Run all combinations (12 total):

```bash
# For each dataset
for dataset in qed fairytaleqa dialogsum; do
  # For each teacher
  for teacher in Llama-2-7b-chat-hf Qwen2-8B bloomz-560M; do
    # For each student
    for student in opt-350m pythia-410m; do
      # For SpanOT-KD enabled and disabled
      for span_kd in true false; do
        echo "Dataset: $dataset, Teacher: $teacher, Student: $student, SpanOT-KD: $span_kd"
        # Uncomment to run:
        # bash $HOME/SpanOT-KD/run_experiments_$dataset.sh \
        #   /workspace/models/$teacher $student 42 $span_kd
      done
    done
  done
done
```

**Total compute time estimate**: 
- ~2-4 hours per experiment × 12 combinations = ~24-48 hours on RTX 5090

---

## Next: Detailed Commands

For more control over training (custom hyperparameters, evaluation tasks), see `EXPERIMENT_COMMANDS.md` for full individual commands.
