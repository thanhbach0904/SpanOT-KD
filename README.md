# SpanOT-KD: Span-Selective Knowledge Distillation

This repository extends **Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models** (Cui et al., AAAI 2025 oral) with **SpanOT-KD**, a span-selective distillation mechanism that reweights the HAD/SL/SD loss components by a per-span entropy-gap importance weight (see "Citation" below for the base method). When the SpanOT-KD flags are disabled, training is byte-for-byte identical to vanilla MultiLevelOT.

## Download Pre-trained Teacher Model

The teacher model used in this setup is Llama-2-7b-chat-hf. Download it from Hugging Face into `$HOME/models/`:

Llama2-7b-chat-hf: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

```bash
hf download meta-llama/Llama-2-7b-chat-hf --local-dir $HOME/models/meta-llama/Llama-2-7b-chat-hf
mv $HOME/models/meta-llama/Llama-2-7b-chat-hf $HOME/models/Llama-2-7b-chat-hf
```

## Download Pre-trained Student Models

Student models are downloaded from Hugging Face into `$HOME/SpanOT-KD/EleutherAI/`:

opt-350m: [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)

pythia-410m: [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)

## Dataset

This setup uses **QED**. The processed dataset is expected at `$HOME/SpanOT-KD/llm_distillation/datasets/processed/qed`. If you do not have it locally, see [Datasets](#datasets) below for the source and the transfer steps to produce the teacher-labeled `qedllama` variant used by the distillation loader.

## Task-specific Student Model Distillation

For distillation, the relevant parameters are:
- `--model_name`: The path/ID of the student model.
- `--lr`: Learning rate for the training process.
- `--num_epochs`: Number of epochs for training.
- `--batch_size_training`: Batch size for training.
- `--val_batch_size`: Batch size for validation.
- `--dataset.file`: Path to the dataset loader file.
- `--output_dir`: Directory to save the output.
- `--distillation`: Activate distillation.
- `--distillation_config_model_name`: The path/ID of the teacher model.
- `--distillation_config_pure_bf16`: Use pure BF16 precision.
- `--distillation_config_distil_factor`: Weight of the distillation loss term.
- `--dev_split_ratio` / `--dev_split_seed`: Seeded dev carve-out from the on-disk train split (the on-disk `validation` split is the held-out test set and is never used for selection).
- `--early_stopping_patience`: Epochs of no dev-loss improvement before stopping.
- `--dev_gen_batch_size`: Batch size for the in-training dev-F1 generation pass.
- `--f`: Distillation alignment method. `f=1`: ours (fast); `f=2`: ours (greedy).
- `--seed`: Global seed (also drives the BatchSampler shuffle and, unless overridden, `--dev_split_seed`).

SpanOT-KD-specific flags (no-ops unless `--distillation_config_span_kd_enabled` is set):
- `--distillation_config_span_kd_enabled`: Master switch for the span-selective reweighting.
- `--distillation_config_span_aggregation {mean,sum}`: Per-span entropy-gap aggregation.
- `--distillation_config_span_low_delta`: Weight assigned to low-priority (non top-r) spans.
- `--distillation_config_span_top_r_sweep`: Instead of a fixed `--distillation_config_span_top_r`, train 3 students per seed with `span_top_r ∈ {0.3, 0.5, 0.7}` and keep the best by dev F1. Requires `--distillation` and `--distillation_config_span_kd_enabled`.

## Example Commands

Below are the reference commands for this setup: OPT-350m and Pythia-410m students distilled from Llama-2-7b-chat-hf on QED, plus the mean-vs-sum span-aggregation ablation.

### OPT-350m

```bash
# Train (SpanOT-KD enabled, seed 42)
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_opt \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
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

# Train (vanilla MultiLevelOT baseline, seed 4)
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_opt \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 4 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --seed 4

# Evaluate best_dev_f1 checkpoint
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

# Evaluate best_dev_loss checkpoint
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkqedllama.py \
  --model_id $HOME/SpanOT-KD/output_qed_opt/best_dev_loss \
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

### Pythia-410m

```bash
# Train (SpanOT-KD enabled, seed 4; SPANOT_TRACE* env vars enable the
# per-sample tracer described in change_logs.md, optional)
SPANOT_TRACE=1 SPANOT_TRACE_STEP=0 SPANOT_TRACE_B=0 SPANOT_TRACE_SEED=4 CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_pythia \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 4 \
  --early_stopping_patience 4 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation mean \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 4

# Evaluate best_dev_f1 checkpoint
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkqedllama.py \
  --model_id $HOME/SpanOT-KD/output_qed_pythia/best_dev_f1 \
  --model_tokenizer $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset_id $HOME/SpanOT-KD/llm_distillation/datasets/processed/qed \
  --split_name validation \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path $HOME/SpanOT-KD/eval_results/qed_pythia_llama/

# Evaluate best_dev_loss checkpoint
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/llm_distillation/benchmark/benchmarkqedllama.py \
  --model_id $HOME/SpanOT-KD/output_qed_pythia/best_dev_loss \
  --model_tokenizer $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset_id $HOME/SpanOT-KD/llm_distillation/datasets/processed/qed \
  --split_name validation \
  --batch_size 4 \
  --num_workers 2 \
  --context_length 1024 \
  --from_disk \
  --task qa \
  --bfloat \
  --save_predictions \
  --output_path $HOME/SpanOT-KD/eval_results/qed_pythia_llama/
```

### Ablation study (mean vs. sum span aggregation)

```bash
# OPT-350m, span_aggregation=sum, seed 94
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_opt \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 94 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation sum \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 94

# OPT-350m, span_aggregation=sum, seed 42 (with per-sample tracer)
SPANOT_TRACE=1 SPANOT_TRACE_STEP=0 SPANOT_TRACE_B=0 SPANOT_TRACE_SEED=42 CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/opt-350m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_opt \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
  --distillation \
  --distillation_config_pure_bf16 \
  --distillation_config_distil_factor 0.15 \
  --dev_split_ratio 0.1 \
  --dev_split_seed 42 \
  --early_stopping_patience 3 \
  --dev_gen_batch_size 2 \
  --f 1 \
  --distillation_config_span_kd_enabled \
  --distillation_config_span_aggregation sum \
  --distillation_config_span_low_delta 0.1 \
  --distillation_config_span_top_r_sweep \
  --seed 42

# Pythia-410m, vanilla MultiLevelOT baseline, seed 4
CUDA_VISIBLE_DEVICES=0 python $HOME/SpanOT-KD/finetuning.py \
  --model_name $HOME/SpanOT-KD/EleutherAI/pythia-410m \
  --dataset.file $HOME/SpanOT-KD/llm_distillation/datasets/loader/qed.py \
  --lr 1e-6 \
  --num_epochs 5 \
  --batch_size_training 2 \
  --val_batch_size 2 \
  --output_dir $HOME/SpanOT-KD/output_qed_pythia \
  --distillation_config_model_name /workspace/models/Llama-2-7b-chat-hf \
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

## Datasets

QED has been uploaded to Google Drive: [https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing](https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing). You need to download it yourself — it is too large to push to git.

Most dataset files are given in `SpanOT-KD/llm_distillation/datasets/hf/` and `SpanOT-KD/llm_distillation/datasets/hf/processed/`. Raw splits need to be converted to Arrow datasets (`transfer.py` in `llm_distillation/datasets/hf/` shows an example).

To add the teacher's answer as the student's distillation label (the `qedllama` variant consumed by `qed.py`), generate teacher predictions with `benchmarkqedllama.py`/`benchmark.py`, save them to a JSON file, then run the corresponding `transfer.py` to merge them into a new Arrow dataset.

## Student Checkpoints

The distilled student checkpoints reported in the base MultiLevelOT paper can be downloaded here:
[https://drive.google.com/drive/folders/1O6k6THm_PjqNybDixppXhad0Nyk-xIjB?usp=drive_link](https://drive.google.com/drive/folders/1O6k6THm_PjqNybDixppXhad0Nyk-xIjB?usp=drive_link) &
[https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing](https://drive.google.com/drive/folders/1ZE_wu0Ey2KpKrjq3NA0VgAvyhynOR6a4?usp=sharing)

## Environmental statement

All these files use `{os.getenv('HOME')}`:

llm_distillation/datasets/generator.py

llm_distillation/datasets/loader/*

llm_distillation/prompt/prompt.py

llm_distillation/benchtestfairy.py

llm_distillation/benchmark/*

If you hit environment errors on your machine, try changing these into direct paths.

## Citation

This repository builds on the Multi-Level Optimal Transport distillation method. If you use this code, please cite the base method:

```
@article{cui2024multi,
  title={Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models},
  author={Cui, Xiao and Zhu, Mo and Qin, Yulei and Xie, Liang and Zhou, Wengang and Li, Houqiang},
  journal={arXiv preprint arXiv:2412.14528},
  year={2024}
}
```
