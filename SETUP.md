# SpanOT-KD Setup Guide

This guide explains how to set up SpanOT-KD on a fresh machine. Setup is modularized into 3 independent phases so you can run them separately or together.

## Quick Start (All-in-One)

If you want to set up everything at once (recommended for first-time setup):

```bash
HF_TOKEN=your_hf_token bash setup_all.sh
```

This runs all three phases in sequence:
1. Clone repo + install dependencies
2. Download teacher model (Llama-2-7b-chat-hf by default)
3. Download student models (OPT-350m, Pythia-410m)

**Time**: ~10-20 minutes depending on network and disk speed.

---

## Modular Setup (Individual Phases)

Run phases independently to have more control or to skip/retry specific steps.

### Phase 1: Repo and Dependencies

```bash
bash setup_repo_and_deps.sh [GITHUB_TOKEN]
```

**What it does**:
- Clones SpanOT-KD repository to `$HOME/SpanOT-KD`
- Creates symlink `$HOME/Multi-Level-OT -> $HOME/SpanOT-KD` (required by training code)
- Installs Python dependencies (transformers, datasets, peft, etc.)

**Requirements**:
- Git
- Python 3.8+ with pip
- ~5 GB disk space

**Optional**: Pass `GITHUB_TOKEN` for SSH-authenticated clone (useful for private repos):
```bash
GITHUB_TOKEN=your_github_pat bash setup_repo_and_deps.sh your_github_pat
```

**Time**: ~3-5 minutes.

---

### Phase 2: Teacher Model Download

```bash
HF_TOKEN=your_hf_token bash setup_teacher.sh [model_id] [local_dir]
```

**What it does**:
- Authenticates with Hugging Face
- Downloads a teacher model
- Saves to `local_dir` (or `$HOME/models/Llama-2-7b-chat-hf` by default)

**Defaults**:
- Model: `meta-llama/Llama-2-7b-chat-hf` (Llama-2-7b)
- Location: `$HOME/models/Llama-2-7b-chat-hf`

**Examples**:

```bash
# Download Llama-2-7b (default)
HF_TOKEN=your_hf_token bash setup_teacher.sh

# Download Qwen2-8B
HF_TOKEN=your_hf_token bash setup_teacher.sh Qwen/Qwen2-8B /workspace/models/Qwen2-8B

# Download BLOOMZ-560M
HF_TOKEN=your_hf_token bash setup_teacher.sh bigscience/bloomz-560m /workspace/models/bloomz-560M
```

**Requirements**:
- HF_TOKEN for gated models (Llama, Qwen, etc.)
- ~20-80 GB disk space depending on model size

**Time**: 5-30 minutes depending on model size and network.

---

### Phase 3: Student Models Download

```bash
bash setup_students.sh [model_pairs...]
```

**What it does**:
- Downloads student models to `$HOME/SpanOT-KD/EleutherAI/`
- Default: OPT-350m and Pythia-410m

**Defaults**:
- `facebook/opt-350m` → `EleutherAI/opt-350m`
- `EleutherAI/pythia-410m` → `EleutherAI/pythia-410m`

**Examples**:

```bash
# Download default students
bash setup_students.sh

# Download custom students
bash setup_students.sh \
  "facebook/opt-125m|EleutherAI/opt-125m" \
  "EleutherAI/pythia-70m|EleutherAI/pythia-70m"

# Override via environment variable
STUDENT_MODELS="facebook/opt-1.3b|EleutherAI/opt-1.3b" bash setup_students.sh
```

**Requirements**:
- ~10 GB disk space for default models
- HF_TOKEN already set (via `hf auth login` or env var)

**Time**: 5-15 minutes.

---

## Verification

After setup completes, verify all components are in place:

```bash
bash verify_setup.sh
```

This checks:
- Repository exists and symlink is correct
- Dependencies are installed
- Teacher model is present
- Student models are present
- Dataset loaders exist
- Benchmark scripts exist

Example output:
```
✓ Repository exists at /workspace/SpanOT-KD
✓ Symlink /workspace/Multi-Level-OT -> /workspace/SpanOT-KD is correct
✓ transformers is installed (version: 4.40.2)
✓ Models directory exists at /workspace/models
✓ Found 1 teacher model(s): Llama-2-7b-chat-hf
✓ Student models directory exists at /workspace/SpanOT-KD/EleutherAI
✓ Found 2 student model(s): opt-350m, pythia-410m
...
✓ Setup verification passed!
```

---

## Use Cases

### Scenario 1: First-time setup on Vast.ai

```bash
# 1. One command: full setup
HF_TOKEN=your_token bash setup_all.sh

# 2. Verify
bash verify_setup.sh

# 3. Run experiment
bash run_experiments_qed.sh /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
```

### Scenario 2: Add a new teacher model later

```bash
# Just download the teacher (phases 1-3 already done)
HF_TOKEN=your_token bash setup_teacher.sh Qwen/Qwen2-8B /workspace/models/Qwen2-8B

# Use it in an experiment
bash run_experiments_qed.sh /workspace/models/Qwen2-8B opt-350m 42 true
```

### Scenario 3: Download more student models

```bash
# Phase 1-3 already done; just add more students
bash setup_students.sh "facebook/opt-1.3b|EleutherAI/opt-1.3b"

# Now train with the new student
bash run_experiments_qed.sh /workspace/models/Llama-2-7b-chat-hf opt-1.3b 42 true
```

### Scenario 4: Retry a failed phase (network timeout, etc.)

```bash
# Repository clone failed? Re-run phase 1
bash setup_repo_and_deps.sh

# Teacher download interrupted? Re-run phase 2
HF_TOKEN=your_token bash setup_teacher.sh

# Student download failed? Re-run phase 3
bash setup_students.sh
```

Scripts detect existing components and skip redundant work (or fail gracefully with clear errors).

---

## Directory Structure After Setup

After all phases complete:

```
/workspace/
├── models/
│   ├── Llama-2-7b-chat-hf/     ← Teacher model (Phase 2)
│   ├── Qwen2-8B/               ← (optional, if downloaded)
│   └── bloomz-560M/            ← (optional, if downloaded)
│
├── SpanOT-KD/                  ← Repository (Phase 1)
│   ├── EleutherAI/             ← Student models (Phase 3)
│   │   ├── opt-350m/
│   │   └── pythia-410m/
│   ├── llm_distillation/
│   ├── finetuning.py
│   ├── setup_all.sh
│   ├── setup_repo_and_deps.sh
│   ├── setup_teacher.sh
│   ├── setup_students.sh
│   ├── verify_setup.sh
│   ├── run_experiments_qed.sh
│   ├── run_experiments_fairytaleqa.sh
│   └── run_experiments_dialogsum.sh
│
└── Multi-Level-OT -> SpanOT-KD  ← Symlink (Phase 1, required by training code)
```

---

## Environment Variables

### Required

- **HF_TOKEN** — Hugging Face API token (needed for Phase 2)
  - Get it from: https://huggingface.co/settings/tokens
  - For gated models (Llama), you must accept the license on the model card first

### Optional

- **GITHUB_TOKEN** — GitHub personal access token (Phase 1 only)
  - Needed only if repo is private
  - If not set, `git clone` uses HTTPS (you'll be prompted for credentials)

- **STUDENT_MODELS** — Space-separated "id|dir" pairs (Phase 3 only)
  - Override default students
  - Example: `STUDENT_MODELS="facebook/opt-125m|EleutherAI/opt-125m" bash setup_students.sh`

- **REPO_PATH** — Custom repository location
  - Default: `$HOME/SpanOT-KD`
  - Used by Phase 3 and all experiment runners

- **HOME** — User home directory
  - Default: `/workspace` (on Vast.ai)
  - Change if your pod uses a different home

---

## Troubleshooting

### "HF_TOKEN is not set"

```bash
# Fix: Pass HF_TOKEN when running Phase 2
HF_TOKEN=your_hf_token bash setup_teacher.sh

# Or set it globally (in ~/.bashrc or ~/.zshrc)
export HF_TOKEN="your_hf_token"
bash setup_teacher.sh
```

### "Repository not found at $REPO_PATH"

```bash
# Phase 3 requires Phase 1 to complete first. Run:
bash setup_repo_and_deps.sh
bash setup_students.sh
```

### "No module named transformers"

```bash
# Dependencies weren't installed. Run Phase 1:
bash setup_repo_and_deps.sh
```

### Teacher model download fails (network error)

```bash
# Re-run Phase 2 — it will resume from where it failed
HF_TOKEN=your_token bash setup_teacher.sh
```

### "Accept the license to access this model"

```bash
# You need to accept the license on the model card:
# 1. Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# 2. Click "Accept and access repository"
# 3. Re-run Phase 2:
HF_TOKEN=your_token bash setup_teacher.sh
```

### Out of disk space

```bash
# Check available space
df -h /workspace

# If model download is interrupted, clean up and retry:
rm -rf /workspace/models/Llama-2-7b-chat-hf  # or whichever model
HF_TOKEN=your_token bash setup_teacher.sh
```

---

## Next Steps

After setup completes:

1. **Verify**: `bash verify_setup.sh`

2. **Download additional teachers** (optional):
   ```bash
   HF_TOKEN=your_token bash download_qwen8b.sh
   HF_TOKEN=your_token bash download_bloomz560m.sh
   ```

3. **Run an experiment**:
   ```bash
   bash run_experiments_qed.sh /workspace/models/Llama-2-7b-chat-hf opt-350m 42 true
   ```

4. **See all experiment options**: `cat QUICKSTART.md` or `cat EXPERIMENT_COMMANDS.md`

---

## For the Original setup.sh

The original `setup.sh` still works but runs all three phases in one go without modularity. We recommend using the new modular scripts, but if you prefer the original:

```bash
HF_TOKEN=your_token bash setup.sh
```

This is equivalent to:

```bash
bash setup_repo_and_deps.sh
HF_TOKEN=your_token bash setup_teacher.sh
bash setup_students.sh
```

The original `setup.sh` will eventually be deprecated in favor of the modular approach.
