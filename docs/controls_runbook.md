# Controls runbook: selection vs. reduced effective weight

Source plan: `PLAN_matched_weight_controls.md` §3 (steps below verbatim, with the exact commands added).

1. Pull branch, run `pytest tests/` (must pass).
2. Dry-run per arm: 16 samples, 2 steps (agent: find the exact flag in `finetuning.py`; document the command).
3. GPU order (stop early if budget is short): `matched` on QED for all seeds -> `matched` on FairytaleQA ->
   `mlot_scaled` (optional) -> r=1.0 extra point (optional).
4. After runs: `python scripts/aggregate_controls.py ...`, then write a 3-5 line summary into `experiments/log.md`
   (config hash, key metrics, interpretation, next action).
5. Compute cost, stated as arithmetic only: `matched` = 3 trainings (one per r) per seed per (teacher, student, dataset).

---

## Commands

All commands from `$HOME/SpanOT-KD` (vast.ai). `T=/workspace/models/Llama-2-7b-chat-hf`, `S=opt-350m` as examples.

### 1. Tests (CPU, seconds)

```bash
pytest tests/            # or: python -m unittest discover -s tests
```

40 tests: w_eff_valid, linearity, matched select (incl. pre-edit snapshot of entropy/random),
r=1.0 semantics, gradient liveness in matched mode, audit + aggregation scripts.
`tests/fixtures/span_weights_snapshot.json` was generated from the pre-edit code — do NOT regenerate
it unless you intend to change the entropy/random arms.

### 2. Dry run per arm (GPU, minutes)

There was no dry-run flag; `--max_samples N` was added (caps train, dev and dev-gen sets to N rows).
The run scripts expose it as `MAX_SAMPLES=N`, which also sets `--num_epochs 1` and appends `_dryrun`
to the run tag (so dry-run output can never be mistaken for a real run by `run_controls.sh`).
Batch size is 2, so `MAX_SAMPLES=4` = 2 optimizer steps per r; `MAX_SAMPLES=16` = 8 steps per r.

```bash
for ARM in false true random matched; do
  MAX_SAMPLES=4 SKIP_EVAL=1 bash run_experiments_qed.sh         $T $S 42 $ARM || break
  MAX_SAMPLES=4 SKIP_EVAL=1 bash run_experiments_fairytaleqa.sh $T $S 42 $ARM || break
done
```

Check in each dry run's output dir:
- span arms: `r_0.3/ r_0.5/ r_0.7/` each with `w_eff.json` and `run_results.json`, plus `sweep_summary.json`
  whose `per_run.*.w_eff_valid_overall` is filled;
- `[SpanOT-KD] enabled — ... select_mode=matched` in the log for the matched arm;
- `matched` and `true` give similar (not necessarily identical — different models after step 1)
  `diag/w_eff_valid`; `false` gives exactly 1.0.

Optionally drop `SKIP_EVAL=1` once to exercise the eval step (full test set; slower).

### 3. GPU runs (resumable)

```bash
# matched on QED, all seeds of the main table (fill in the real seed list — open question 4)
bash scripts/run_controls.sh $T $S qed "42 4 63" "matched"
# matched on FairytaleQA
bash scripts/run_controls.sh $T $S fairytaleqa "42 4 63" "matched"
# missing arms, if any (skips every (arm, seed) whose final eval file exists)
bash scripts/run_controls.sh $T $S qed "42 4 63"
```

Status: `experiments/controls_status.csv`; per job `<output_dir>/run_meta.json` (commit, dirty flag,
command, exit code) and `controls_stdout.log`. Re-run the same command after an instance death.
Granularity is one (arm, seed) job: a job killed mid-sweep restarts from r=0.3.

Optional `mlot_scaled` (only valid because WP2 found every component linear in omega):

```bash
python scripts/make_scaled_commands.py --repo $HOME/SpanOT-KD --dataset qed --teacher $T --student $S --seeds 42 4 63
# prints DISTIL_FACTOR=... SCALED_R=... bash run_experiments_qed.sh ... scaled   — run those lines
```

Optional r = 1.0 extra point (trained and logged, never used for dev winner selection):

```bash
SPAN_TOP_R_EXTRA=1.0 bash run_experiments_qed.sh $T $S 42 true
```

The test-set eval in the run script only evaluates the dev winner; evaluate `r_1.0` by hand with the
same benchmark command pointed at `$OUTPUT_DIR/r_1.0/best_dev_f1` (FairytaleQA: `best_dev_rouge_l`).
Note this reruns all of 0.3/0.5/0.7 too — to avoid that cost use the 5th (FIXED_R) argument of
`run_experiments_qed.sh` instead: `bash run_experiments_qed.sh $T $S 42 true 1.0`.

### 4. Aggregate

```bash
python scripts/aggregate_controls.py --repo $HOME/SpanOT-KD --dataset qed --teacher $T --student $S \
    --seeds 42 4 63 --out_dir results/controls_qed_${S}
```

Outputs `controls_table.md/.tex`, `controls_raw.csv`. Read the caveat block at the top of the `.md`
before quoting any number. Any `⚠ NOT MATCHED` in the w_eff table means that control is not
weight-matched and cannot answer the reviewer question as is.

Optional span-alignment audit (tokenizer-only, CPU OK):

```bash
python scripts/audit_span_alignment.py --student $HOME/SpanOT-KD/EleutherAI/$S --teacher $T \
    --dataset_file llm_distillation/datasets/loader/qed.py --n 500 --out results/span_audit_qed_${S}
```

### 5. Cost (arithmetic only)

- `matched`, `random`, `true`: 3 trainings each (r ∈ {0.3, 0.5, 0.7}) per seed per (teacher, student, dataset);
  4 with `SPAN_TOP_R_EXTRA=1.0`.
- `false`: 1 training per seed. `mlot_scaled`: 3 trainings per seed.
- Each job also runs 2 test-set evals.
