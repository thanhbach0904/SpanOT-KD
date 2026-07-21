# SpanOT-KD Scripts and Setup Summary

This document provides a high-level overview of all the scripts and documentation added to the repository.

---

## 📋 What Was Added

### 1. **Modular Setup Scripts** (5 files)

Refactored the original monolithic `setup.sh` into 3 independent phases:

| Script | Purpose | Size | Time |
|--------|---------|------|------|
| `setup_repo_and_deps.sh` | Clone repo + install Python deps | 4.1 KB | ~3-5 min |
| `setup_teacher.sh` | Download teacher model | 3.2 KB | 5-30 min |
| `setup_students.sh` | Download student models | 4.2 KB | 5-15 min |
| `setup_all.sh` | Orchestrate all 3 phases (convenience wrapper) | 5.1 KB | ~20-50 min total |
| `verify_setup.sh` | Check all components installed correctly | ~8 KB | ~30 sec |

**Original** `setup.sh` (4.5 KB) — Still works, runs all phases in sequence.

---

### 2. **Experiment Runner Scripts** (3 files)

End-to-end training + evaluation on different datasets:

| Script | Dataset | What it Does |
|--------|---------|--------------|
| `run_experiments_qed.sh` | QED | Train student on QED, evaluate both best_dev_f1 and best_dev_loss checkpoints |
| `run_experiments_fairytaleqa.sh` | FairyTaleQA | Same pipeline for FairyTaleQA |
| `run_experiments_dialogsum.sh` | DialogSum | Same pipeline for DialogSum |

Each accepts: `<teacher_path> <student_model> <seed> [span_kd_enabled]`

Example: `bash run_experiments_qed.sh /workspace/models/Qwen2-8B opt-350m 42 true`

---

### 3. **Teacher Model Download Scripts** (2 files)

Standalone scripts to download specific teachers:

| Script | Model | Default Path |
|--------|-------|--------------|
| `download_qwen8b.sh` | Qwen2-8B | `$HOME/models/Qwen2-8B` |
| `download_bloomz560m.sh` | BLOOMZ-560M | `$HOME/models/bloomz-560M` |

Requires: `HF_TOKEN` environment variable

---

### 4. **Documentation** (4 files)

| Document | Purpose | Length |
|----------|---------|--------|
| `SETUP.md` | Complete setup guide (modular phases, use cases, troubleshooting) | 258 lines |
| `QUICKSTART.md` | Quick-start guide with examples for all 3 datasets | 158 lines |
| `EXPERIMENT_COMMANDS.md` | Full reference of all train/eval commands, parameters, output structure | 264 lines |
| `SCRIPTS_SUMMARY.md` | This file — high-level overview |

---

## 🚀 Common Workflows

### Scenario 1: First-Time Setup (Complete)

```bash
# One command: everything
HF_TOKEN=your_token bash setup_all.sh

# Or step-by-step with control:
bash setup_repo_and_deps.sh
HF_TOKEN=your_token bash setup_teacher.sh              # Llama-2-7b by default
bash setup_students.sh                                 # OPT-350m + Pythia-410m by default

# Verify everything is in place
bash verify_setup.sh

# Run an experiment
bash run_experiments_qed.sh /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
```

**Time**: ~30-50 minutes (network-dependent)

---

### Scenario 2: Add a New Teacher Model

```bash
# Already did setup? Just download the teacher
HF_TOKEN=your_token bash setup_teacher.sh Qwen/Qwen2-8B /workspace/models/Qwen2-8B

# Use it immediately
bash run_experiments_qed.sh /workspace/models/Qwen2-8B opt-350m 42 true
```

**Time**: 10-20 minutes

---

### Scenario 3: Retry a Failed Phase

```bash
# Teacher download timed out? Re-run just that phase:
HF_TOKEN=your_token bash setup_teacher.sh

# Student download interrupted? Re-run:
bash setup_students.sh

# Scripts skip already-installed components automatically
```

**Time**: Only the phase that failed

---

### Scenario 4: Run Experiments on Multiple Datasets

```bash
# Single teacher, multiple datasets:
TEACHER=/workspace/models/Qwen2-8B
SEED=42

bash run_experiments_qed.sh $TEACHER opt-350m $SEED true
bash run_experiments_fairytaleqa.sh $TEACHER pythia-410m $SEED true
bash run_experiments_dialogsum.sh $TEACHER opt-350m $SEED true
```

**Time**: ~6-12 hours total (depends on dataset and model size)

---

### Scenario 5: Compare Multiple Teachers

```bash
# Same dataset/student, different teachers:
STUDENT=opt-350m
SEED=42
DATASET=qed

for teacher in "Llama-2-7b-chat-hf" "Qwen2-8B" "bloomz-560M"; do
  bash run_experiments_$DATASET.sh /workspace/models/$teacher $STUDENT $SEED true
done

# Results in: $HOME/SpanOT-KD/eval_results/
```

**Time**: ~6-12 hours total

---

## 📂 Directory Structure After Setup

```
/workspace/
├── models/
│   └── Llama-2-7b-chat-hf/     ← Phase 2 (Teacher)
│       └── (model files...)
│
├── SpanOT-KD/                  ← Phase 1 (Repo)
│   ├── EleutherAI/             ← Phase 3 (Students)
│   │   ├── opt-350m/
│   │   └── pythia-410m/
│   │
│   ├── llm_distillation/       ← Dataset loaders, benchmarks
│   │   ├── datasets/
│   │   │   ├── loader/
│   │   │   │   ├── qed.py
│   │   │   │   ├── fairytaleQA.py
│   │   │   │   └── dialogsum.py
│   │   │   └── processed/      ← Datasets (user downloads)
│   │   └── benchmark/
│   │       ├── benchmarkqedllama.py
│   │       ├── benchmarkfairytaleQAbasellama.py
│   │       └── benchmarkdialogsumllama.py
│   │
│   ├── finetuning.py           ← Training entry point
│   │
│   ├── Setup scripts (5)
│   │   ├── setup_repo_and_deps.sh
│   │   ├── setup_teacher.sh
│   │   ├── setup_students.sh
│   │   ├── setup_all.sh
│   │   └── verify_setup.sh
│   │
│   ├── Experiment runners (3)
│   │   ├── run_experiments_qed.sh
│   │   ├── run_experiments_fairytaleqa.sh
│   │   └── run_experiments_dialogsum.sh
│   │
│   ├── Teacher downloaders (2)
│   │   ├── download_qwen8b.sh
│   │   └── download_bloomz560m.sh
│   │
│   ├── Documentation (4)
│   │   ├── SETUP.md                    ← Complete setup guide
│   │   ├── QUICKSTART.md               ← Quick-start examples
│   │   ├── EXPERIMENT_COMMANDS.md      ← Full command reference
│   │   └── SCRIPTS_SUMMARY.md          ← This file
│   │
│   ├── output_qed_opt/                 ← Training outputs (created by run_experiments_qed.sh)
│   │   ├── best_dev_f1/
│   │   └── best_dev_loss/
│   │
│   └── eval_results/                   ← Evaluation results (created by run_experiments_*.sh)
│       ├── qed_opt_teacher_seed42/
│       │   ├── predictions.json
│       │   ├── metrics.json
│       │   └── results.txt
│       └── (other datasets/students...)
│
└── Multi-Level-OT -> SpanOT-KD/        ← Symlink (created by Phase 1)
```

---

## 🔧 Script Reference

### Setup Scripts

**`setup_repo_and_deps.sh`**
```bash
bash setup_repo_and_deps.sh [GITHUB_TOKEN]
```
- Clones repo to `$HOME/SpanOT-KD`
- Creates symlink `$HOME/Multi-Level-OT`
- Installs dependencies
- No HF_TOKEN required

**`setup_teacher.sh`**
```bash
HF_TOKEN=token bash setup_teacher.sh [model_id] [local_dir]
```
- Downloads teacher model
- Default: Llama-2-7b-chat-hf
- Customizable with args: `Qwen/Qwen2-8B` or `bigscience/bloomz-560m`

**`setup_students.sh`**
```bash
bash setup_students.sh [model_pairs...]
# or
STUDENT_MODELS="id1|dir1 id2|dir2" bash setup_students.sh
```
- Downloads student models
- Default: OPT-350m + Pythia-410m
- Customizable

**`setup_all.sh`**
```bash
HF_TOKEN=token bash setup_all.sh [teacher_id] [teacher_dir]
```
- Runs all 3 phases in sequence
- Convenience wrapper for first-time users

**`verify_setup.sh`**
```bash
bash verify_setup.sh
```
- Checks all components are installed
- Color-coded output (✓ ✗ ⚠)
- Non-blocking warnings

---

### Experiment Scripts

**`run_experiments_qed.sh`** / **`run_experiments_fairytaleqa.sh`** / **`run_experiments_dialogsum.sh`**

```bash
bash run_experiments_<dataset>.sh <teacher_path> <student_model> <seed> [span_kd_enabled]
```

Arguments:
- `teacher_path` — e.g., `/workspace/models/Qwen2-8B`
- `student_model` — `opt-350m` or `pythia-410m`
- `seed` — any integer (e.g., 42)
- `span_kd_enabled` — `true` or `false` (default: `true`)

What it does:
1. Trains student on dataset with distillation
2. Evaluates best_dev_f1 checkpoint
3. Evaluates best_dev_loss checkpoint
4. Saves results to `$HOME/SpanOT-KD/eval_results/`

Example:
```bash
bash run_experiments_qed.sh /workspace/models/Qwen2-8B opt-350m 42 true
```

---

### Teacher Download Scripts

**`download_qwen8b.sh`** / **`download_bloomz560m.sh`**

```bash
HF_TOKEN=token bash download_qwen8b.sh
HF_TOKEN=token bash download_bloomz560m.sh
```

These are standalone (don't require Phase 1 or 3 to be done). Download to `$HOME/models/`.

---

## 📚 Documentation Quick Links

- **Getting started**: Read `SETUP.md` first
- **Quick examples**: See `QUICKSTART.md`
- **Full commands**: See `EXPERIMENT_COMMANDS.md`
- **All this**: `SCRIPTS_SUMMARY.md` (this file)

---

## 🎯 Key Features

### Modularity
- Each phase is independent
- Can run in any order (Phase 1 before others)
- Can skip phases (e.g., use existing teacher)
- Can retry failed phases

### Idempotent
- Safe to re-run scripts
- Skips already-downloaded components
- Clear error messages if something's missing

### Flexible
- Customize teacher/student models
- Choose any seed for reproducibility
- Enable/disable SpanOT-KD
- Run on any dataset (QED, FairyTaleQA, DialogSum)

### Clear Error Messages
- "Repository not found, did you run Phase 1 first?"
- "HF_TOKEN is not set. Run: HF_TOKEN=<token> bash setup_teacher.sh"
- "No symlink at /workspace/Multi-Level-OT (needed by training code)"

---

## 🔄 Backward Compatibility

- Original `setup.sh` still works (unchanged)
- New modular scripts are opt-in
- No breaking changes to training code

---

## 📝 Environment Variables

| Variable | Used By | Example |
|----------|---------|---------|
| `HF_TOKEN` | setup_teacher.sh, download_*.sh | `hf_api_......` |
| `GITHUB_TOKEN` | setup_repo_and_deps.sh | (optional) |
| `STUDENT_MODELS` | setup_students.sh | `"facebook/opt-125m\|dir" "model\|dir"` |
| `REPO_PATH` | setup_students.sh, run_experiments_*.sh | `/workspace/SpanOT-KD` |
| `HOME` | All scripts | `/workspace` (default on Vast.ai) |

---

## 🐛 Troubleshooting Quick Links

See `SETUP.md` for detailed troubleshooting, but quick links:

- **"HF_TOKEN is not set"** → Pass it: `HF_TOKEN=token bash setup_teacher.sh`
- **"Repository not found"** → Run Phase 1 first: `bash setup_repo_and_deps.sh`
- **"No module named transformers"** → Install deps: `bash setup_repo_and_deps.sh`
- **Teacher download interrupted** → Re-run Phase 2: `HF_TOKEN=token bash setup_teacher.sh`
- **Need to accept license** → Visit model card on HuggingFace, then retry

---

## 📊 Experiment Output

After running `run_experiments_<dataset>.sh`, results are saved to:

```
$HOME/SpanOT-KD/eval_results/<dataset>_<student>_teacher_seed<N>/
```

Each directory contains:
- `predictions.json` — Model predictions on test set
- `metrics.json` — F1, EM, BLEU, ROUGE scores
- `results.txt` — Human-readable summary

Compare metrics across teachers:
```bash
for dir in $HOME/SpanOT-KD/eval_results/*/; do
  echo "=== $(basename $dir) ==="
  cat "$dir/metrics.json" | jq '.f1, .em'
done
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Phase 1 (repo + deps) | 3-5 minutes |
| Phase 2 (Llama-2-7b) | 15-30 minutes |
| Phase 2 (Qwen2-8B) | 20-40 minutes |
| Phase 3 (OPT-350m + Pythia-410m) | 5-15 minutes |
| **Full setup (all phases)** | **~30-50 minutes** |
| Run one experiment (train + eval) | 2-4 hours |
| Run 3 experiments (3 datasets) | 6-12 hours |

---

## 💡 Best Practices

1. **Always run `verify_setup.sh`** after setup to catch missing components
2. **Run Phase 1 first** (it creates the repo directory that Phase 2-3 expect)
3. **Use meaningful seeds** (e.g., 42 for reproducibility, 4 for ablation)
4. **Run with SpanOT-KD enabled** for main experiments (add `true` flag)
5. **Run baseline with SpanOT-KD disabled** for comparison (add `false` flag)
6. **Check eval results after training** to detect issues early
7. **Document your setup and results** in `change_logs.md`

---

## 🆘 Need Help?

1. **Setup issues** → Read `SETUP.md` (Troubleshooting section)
2. **Example commands** → See `QUICKSTART.md`
3. **Full reference** → See `EXPERIMENT_COMMANDS.md`
4. **Script descriptions** → This file (`SCRIPTS_SUMMARY.md`)

---

## 📄 Files Summary

```
Setup (5 scripts):
  - setup_repo_and_deps.sh (4.1 KB)
  - setup_teacher.sh (3.2 KB)
  - setup_students.sh (4.2 KB)
  - setup_all.sh (5.1 KB)
  - verify_setup.sh (~8 KB)

Experiments (3 scripts):
  - run_experiments_qed.sh (5.2 KB)
  - run_experiments_fairytaleqa.sh (5.4 KB)
  - run_experiments_dialogsum.sh (5.4 KB)

Teachers (2 scripts):
  - download_qwen8b.sh (842 B)
  - download_bloomz560m.sh (835 B)

Documentation (4 files):
  - SETUP.md (258 lines)
  - QUICKSTART.md (158 lines)
  - EXPERIMENT_COMMANDS.md (264 lines)
  - SCRIPTS_SUMMARY.md (this file)

Original (preserved):
  - setup.sh (4.5 KB, still works)
```

**Total new code**: ~14 KB of scripts + ~680 lines of documentation

---

Last updated: 2026-07-21
