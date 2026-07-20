# SpanOT-KD Experiment Commands Reference

This document contains all the commands needed to run experiments with different teacher models (Qwen2-8B, BLOOMZ-560M, Llama-2-7b) across three datasets (QED, FairyTaleQA, DialogSum).

## Prerequisites

1. **Download Teacher Models** (if not already done):

```bash
# Llama-2-7b-chat-hf (already in setup.sh)
hf download meta-llama/Llama-2-7b-chat-hf --local-dir $HOME/models/Llama-2-7b-chat-hf

# Qwen2-8B
HF_TOKEN=<your_hf_token> bash $HOME/SpanOT-KD/download_qwen8b.sh

# BLOOMZ-560M
HF_TOKEN=<your_hf_token> bash $HOME/SpanOT-KD/download_bloomz560m.sh
```

2. **Ensure student models are downloaded**:
   - `$HOME/SpanOT-KD/EleutherAI/opt-350m`
   - `$HOME/SpanOT-KD/EleutherAI/pythia-410m`

3. **Verify datasets are available**:
   - `$HOME/SpanOT-KD/llm_distillation/datasets/processed/qed`
   - `$HOME/SpanOT-KD/llm_distillation/datasets/processed/fairytaleqa`
   - `$HOME/SpanOT-KD/llm_distillation/datasets/processed/dialogsum`

---

## Running Experiments

### Quick Start (Single Command)

Use the provided experiment scripts. Each takes teacher path, student model, seed, and optional SpanOT-KD flag:

```bash
# QED with Qwen2-8B, OPT-350m, seed 42, SpanOT-KD enabled
bash $HOME/SpanOT-KD/run_experiments_qed.sh /workspace/models/Qwen2-8B opt-350m 42 true

# FairyTaleQA with BLOOMZ-560M, Pythia-410m, seed 4, SpanOT-KD disabled (baseline)
bash $HOME/SpanOT-KD/run_experiments_fairytaleqa.sh /workspace/models/bloomz-560M pythia-410m 4 false

# DialogSum with Llama-2-7b, OPT-350m, seed 42, SpanOT-KD enabled
bash $HOME/SpanOT-KD/run_experiments_dialogsum.sh /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
```

---

## QED Dataset

### Train OPT-350m (SpanOT-KD Enabled, Seed 42)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_opt \
  --distillation_config_model_name /workspace/models/Qwen2-8B \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 42 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation mean \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 42
```

### Train Pythia-410m (Vanilla MultiLevelOT Baseline, Seed 4)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_pythia \
  --distillation_config_model_name /workspace/models/bloomz-560M \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 4 \
  --early_stopping_patience 4 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --seed 4
```

### Evaluate QED (OPT-350m, best_dev_f1 checkpoint)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkqedllama.py \
  --model_id $HOME/SpanOT-KD/output_qed_opt/best_dev_f1 \
  --model_tokenizer $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset_id $HOME/SpanOT-KD/llm_distillation/datasets/processed/qed \
  --split_name validation \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path $HOME/SpanOT-KD/eval_results/qed_opt_llama/
```

---

## FairyTaleQA Dataset

### Train OPT-350m (SpanOT-KD Enabled, Seed 42)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/fairytaleQA.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_fairytaleqa_opt \
  --distillation_config_model_name /workspace/models/Qwen2-8B \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 42 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation mean \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 42
```

### Train Pythia-410m (Vanilla MultiLevelOT Baseline, Seed 4)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/fairytaleQA.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_fairytaleqa_pythia \
  --distillation_config_model_name /workspace/models/bloomz-560M \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 4 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --seed 4
```

### Evaluate FairyTaleQA (OPT-350m, best_dev_f1 checkpoint)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkfairytaleQAbasellama.py \
  --model_id $HOME/SpanOT-KD/output_fairytaleqa_opt/best_dev_f1 \
  --model_tokenizer $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset_id $HOME/SpanOT-KD/llm_distillation/datasets/processed/fairytaleqa \
  --split_name validation \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path $HOME/SpanOT-KD/eval_results/fairytaleqa_opt_llama/
```

---

## DialogSum Dataset

### Train OPT-350m (SpanOT-KD Enabled, Seed 42)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/dialogsum.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_dialogsum_opt \
  --distillation_config_model_name /workspace/models/Qwen2-8B \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 42 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation mean \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 42
```

### Train Pythia-410m (Vanilla MultiLevelOT Baseline, Seed 4)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/dialogsum.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_dialogsum_pythia \
  --distillation_config_model_name /workspace/models/bloomz-560M \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 4 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --seed 4
```

### Evaluate DialogSum (OPT-350m, best_dev_f1 checkpoint)

```bash
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkdialogsumllama.py \
  --model_id $HOME/SpanOT-KD/output_dialogsum_opt/best_dev_f1 \
  --model_tokenizer $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset_id $HOME/SpanOT-KD/llm_distillation/datasets/processed/dialogsum \
  --split_name validation \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task summarization \
  --bfloat \
  --save_predictions \
  --output_path $HOME/SpanOT-KD/eval_results/dialogsum_opt_llama/
```

---

## Parameter Guide

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Path to student model | Required |
| `--distillation_config_model_name` | Path to teacher model | Required |
| `--dataset.file` | Path to dataset loader script | Required |
| `--output_dir` | Directory to save training outputs | Required |
| `--seed` | Random seed for reproducibility | 42 |
| `--lr` | Learning rate | 1e-6 |
| `--num_epochs` | Number of training epochs | 5 |
| `--batch_size_training` | Training batch size | 2 |
| `--val_batch_size` | Validation batch size | 2 |
| `--distillation_config_distil_factor` | Weight of distillation loss | 0.15 |
| `--distillation_config_span_kd_enabled` | Enable SpanOT-KD reweighting | false |
| `--distillation_config_span_aggregation` | Span aggregation method | mean or sum |
| `--distillation_config_span_low_delta` | Weight for non-top-r spans | 0.1 |
| `--distillation_config_span_top_r_sweep` | Sweep span_top_r ∈ {0.3,0.5,0.7} | false |

---

## Output Structure

After running experiments, results are organized as:

```
$HOME/SpanOT-KD/
├── output_<dataset>_<student>/      # Training checkpoints
│   ├── best_dev_f1/                 # Best F1 checkpoint
│   └── best_dev_loss/               # Best loss checkpoint
└── eval_results/
    ├── <dataset>_<student>_teacher_seed<N>/
    │   ├── predictions.json         # Model predictions
    │   ├── metrics.json             # Evaluation metrics
    │   └── results.txt              # Summary results
    └── ...
```

---

## Notes

- All commands assume `HOME=/workspace` and GPU 0 is available
- Adjust `--batch_size_training` and `--val_batch_size` based on GPU memory
- Set `CUDA_VISIBLE_DEVICES` to select different GPUs
- Each experiment takes ~2-4 hours on an RTX 5090 with batch size 2
- Use `--dev_split_seed` to create different dev splits for multiple runs
- Results are logged to W&B if `WANDB_API_KEY` is set
