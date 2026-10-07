import os
import json
import time
import torch
import wandb
os.environ["WANDB_MODE"] = "dryrun"
import torch.distributed as dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from contextlib import nullcontext
from models.memory import MemoryTrace
from train.tools import clear_gpu_cache
from train.evaluations import evaluation
from train.save import save_train_params, save_model
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

# llm_distillation/benchmark is a namespace package with no __init__.py and
# its modules use bare imports (e.g. `import score`), so we add the
# benchmark directory to sys.path before pulling in score. Derived from
# __file__ (not hardcoded to a repo folder name like "Multi-Level-OT")
# so this works regardless of what the clone directory is named.
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BENCHMARK_DIR = os.path.join(_REPO_ROOT, "llm_distillation", "benchmark")
if _BENCHMARK_DIR not in _sys.path:
    _sys.path.append(_BENCHMARK_DIR)
import score as benchmark_score


def _compute_dev_f1(student_model, dev_gen_dataloader, dev_gen_answers, tokenizer,
                    max_new_tokens, device, rank, also_score_rouge_l=False):
    """Greedy generation on dev + token-overlap F1 matching the benchmark driver.

    Returns the average F1 (float). If `also_score_rouge_l` is set, also
    scores ROUGE-L (`benchmark_score.rouge`, same scorer used at test time by
    the fairytaleQA benchmark driver) on the same generations and returns
    (f1, rouge_l) instead — this avoids a second, redundant generation pass.
    Returns None (or (None, None)) on non-zero ranks; only rank 0 holds the
    prediction strings (cheap operation, no DDP gather).
    """
    if rank != 0:
        # F1/ROUGE-L are computed on rank 0 only — predictions are CPU strings
        # and the operation is cheap relative to training. Other ranks wait.
        if dist.is_initialized():
            dist.barrier()
        return (None, None) if also_score_rouge_l else None

    student_model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dev_gen_dataloader, desc="Dev F1 (gen)", colour="cyan", dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = student_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            # Strip the prompt prefix — input_ids is left-padded so prompt
            # length is identical across the batch (= input_ids.shape[1]).
            output = output[:, input_ids.shape[1]:]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            for s in sentences:
                predictions.append(s.split('\n')[0].strip())

    n_pred = len(predictions)
    n_ans = len(dev_gen_answers)
    if n_pred != n_ans:
        print(f"[dev F1] WARNING: {n_pred} predictions vs {n_ans} gold answers — truncating to min.")
    n = min(n_pred, n_ans)
    res = benchmark_score.f1_score(predictions[:n], dev_gen_answers[:n])
    f1 = float(res['f1'])

    rouge_l = None
    if also_score_rouge_l:
        rouge_res = benchmark_score.rouge(predictions[:n], dev_gen_answers[:n])
        rouge_l = float(rouge_res['rougeL'])

    if dist.is_initialized():
        dist.barrier()
    return (f1, rouge_l) if also_score_rouge_l else f1



def run_grad_norm_probe(components, student_params, optimizer):
    """Backward each component separately and return a dict of grad norms.

    ``components`` is a list of ``(name, tensor)``. Grads are zeroed before
    and after each probe so each measurement is clean. The computation graph
    is torn down by the FINAL probe (no ``retain_graph``) — callers must
    therefore re-run the forward pass if they want to backward the full loss
    afterwards. In practice we only run this probe when we can afford to skip
    the real optimiser step for that micro-batch (see train loop).
    """
    norms = {}
    n = len(components)
    for idx, (name, comp) in enumerate(components):
        optimizer.zero_grad(set_to_none=True)
        if comp is None or not comp.requires_grad or comp.grad_fn is None:
            norms[name] = float("nan")
            continue
        is_last = (idx == n - 1)
        if is_last:
            comp.backward()  # tear down graph on final probe
        else:
            comp.backward(retain_graph=True)
        sq_sum = 0.0
        for p in student_params:
            if p.grad is not None:
                g = p.grad.detach()
                sq_sum += float(g.float().norm(2).item()) ** 2
        norms[name] = sq_sum ** 0.5
    optimizer.zero_grad(set_to_none=True)
    return norms


def plot_loss_curves(loss_history, output_path, smoothing_window=50,
                     epoch_train_loss=None, epoch_dev_loss=None, epoch_dev_f1=None):
    """Plot train/dev loss and dev F1 per epoch."""
    has_epoch_data = bool(epoch_train_loss or epoch_dev_loss or epoch_dev_f1)
    if not has_epoch_data:
        print("[plot_loss_curves] No epoch data recorded, skipping plot.")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    if epoch_train_loss:
        epochs_train = list(range(1, len(epoch_train_loss) + 1))
        ax1.plot(epochs_train, epoch_train_loss, color="tab:blue", marker="o",
                 linewidth=1.8, label="train_loss")

    if epoch_dev_loss:
        epochs_dev = list(range(1, len(epoch_dev_loss) + 1))
        ax1.plot(epochs_dev, epoch_dev_loss, color="tab:orange", marker="s",
                 linewidth=1.8, label="dev_loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Dev Loss and Dev F1")
    ax1.grid(True, alpha=0.3)

    if epoch_dev_f1:
        ax2 = ax1.twinx()
        epochs_f1 = list(range(1, len(epoch_dev_f1) + 1))
        ax2.plot(epochs_f1, epoch_dev_f1, color="tab:green", marker="^",
                 linewidth=1.8, linestyle="--", label="dev_F1")
        ax2.set_ylabel("F1")
        ax2.set_ylim(0, 1)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_loss_curves] Saved loss curve plot to: {output_path}")



def plot_grad_norm_curves(grad_norm_history, output_path):
    """Plot per-component gradient-norm probe results.

    ``grad_norm_history`` is a dict with keys ``step``, ``l1``, ``kl``, ``sinkhorn``,
    populated only at steps where the probe ran (see LSD_PROBE_EVERY env var).
    """
    steps = grad_norm_history.get("step", [])
    if len(steps) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for key, label, color in [
        ("l1",       "||grad L1/OT||",   "tab:green"),
        ("kl",       "||grad KL||",      "tab:red"),
        ("sinkhorn", "||grad Sinkhorn||","tab:purple"),
    ]:
        vals = grad_norm_history.get(key, [])
        if len(vals) != len(steps):
            continue
        ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.2,
                color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Training step (global)")
    ax.set_ylabel("Gradient L2 norm over student params (log)")
    ax.set_title("Per-component gradient-norm probe (retain_graph=True backwards)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_grad_norm_curves] Saved grad-norm plot to: {output_path}")

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, teacher_train_dataloader=None, teacher_eval_dataloader=None, fsdp_config=None, local_rank=None, rank=None, f=1, dev_gen_dataloader=None, dev_gen_answers=None, student_tokenizer=None, wandb_name_suffix=None):
    # Weights & Biases tracking system initialization.
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if rank == 0:
        # When run inside a span_top_r sweep, the caller passes a suffix
        # (e.g. "-r0.3") so the three runs are distinguishable in the W&B UI.
        _suffix = wandb_name_suffix or ""
        wandb.init(
            project=f"llm_distillation_{dataset_config.file.split('/')[-1][:-3]}",
            name=(f"{train_config.model_name.split('/')[-1]}-{model.teacher.name_or_path.split('/')[-1]}-d{distil_config.distil_factor}-t{distil_config.teacher_temperature}{distil_config.student_temperature}{_suffix}"
                  if train_config.distillation else f"{train_config.model_name.split('/')[-1]}{_suffix}"),
            config={
                "model_name": train_config.model_name.split('/')[-1],
                "dataset": dataset_config.file.split('/')[-1],
                "batch_size_training": train_config.batch_size_training,
                "val_batch_size": train_config.val_batch_size,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "num_epochs": train_config.num_epochs,
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
                "pct_start": train_config.pct_start,
                "div_factor": train_config.div_factor,
                "final_div_factor": train_config.final_div_factor,
                "seed": train_config.seed,
                "use_fp16": train_config.use_fp16,
                "mixed_precision": train_config.mixed_precision,
                "peft_method": train_config.peft_method,
                "use_peft": train_config.use_peft,
                "freeze_layers": train_config.freeze_layers,
                "num_freeze_layers": train_config.num_freeze_layers,
                "quantization": train_config.quantization,
                "cross_entropy_factor": distil_config.cross_entropy_factor if train_config.distillation else -1,
                "distil_factor": distil_config.distil_factor if train_config.distillation else -1,
                "student_temperature": distil_config.student_temperature if train_config.distillation else -1,
                "teacher_temperature": distil_config.teacher_temperature if train_config.distillation else -1
            }
        )

    # Init distillation loss if distillation is enabled.
    # SpanOT-KD knobs (`span_kd_enabled`, `span_aggregation`, `span_top_r`,
    # `span_low_delta`) are read off `distil_config` and forwarded into
    # the loss. They default to MultiLevelOT-equivalent values, so a run
    # that does not set the SpanOT flags is byte-for-byte identical to
    # the previous behaviour.
    if train_config.distillation:
        distillation_loss = DistillationLoss(
            distillation_weight=distil_config.distil_factor,
            student_temperature=distil_config.student_temperature,
            teacher_temperature=distil_config.teacher_temperature,
            skip_student_eos=True,
            debug=False,
            debug_rank=0,
            tokenizer_student=model.student.name_or_path,
            tokenizer_teacher=model.teacher.name_or_path,
            f=f,
            span_kd_enabled=getattr(distil_config, "span_kd_enabled", False),
            span_aggregation=getattr(distil_config, "span_aggregation", "mean"),
            span_top_r=getattr(distil_config, "span_top_r", 0.5),
            span_low_delta=getattr(distil_config, "span_low_delta", 0.1),
            span_select_mode=getattr(distil_config, "span_select_mode", "entropy"),
            span_random_pool=getattr(distil_config, "span_random_pool", "active"),
            span_random_seed=train_config.seed,
        )

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_ppl = []
    val_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}

    # Per-step loss history (rank 0 only) for plotting the loss curve at the end of training.
    # The `sinkhorn_raw`/`sinkhorn_per_sample`/`cost_*` keys are the diagnostics
    # captured *before* any weight scaling, so a collapse visible in `sinkhorn`
    # (the weighted component) can be traced to either scaling or an upstream
    # problem in the cost matrix.
    loss_history = {
        "step": [],
        "ce": [],
        "distil": [],
        "l1": [],
        "kl": [],
        "sinkhorn": [],
        "sinkhorn_raw": [],            # diag: Sinkhorn_seq.forward output, pre-gamma, pre-w
        "sinkhorn_per_sample": [],     # diag: mean over batch of raw sum(P*C), no scaling
        "cost_mean": [],               # diag: mean of L1 cost matrix C
        "cost_max": [],                # diag: max  of L1 cost matrix C
        "cost_min": [],                # diag: min  of L1 cost matrix C
    }
    # Per-component gradient-norm probes (rank 0 only, opt-in via env var).
    grad_norm_history = {"step": [], "l1": [], "kl": [], "sinkhorn": []}

    # Opt-in per-component gradient-norm probe.
    # Set LSD_PROBE_EVERY=N to run the probe every N optimiser steps (N=0 disables).
    # The probe does 3 extra backward passes with retain_graph=True, so it is
    # O(4x) the cost of a normal step and should NOT be left on by default.
    grad_probe_every = int(os.environ.get("LSD_PROBE_EVERY", "0"))
    if rank == 0 and grad_probe_every > 0:
        print(f"[train] Gradient-norm probe ENABLED (every {grad_probe_every} steps). "
              f"Expect ~4x step cost on probe steps.")
        if train_config.use_fp16:
            print("[train] WARNING: use_fp16=True and probe does not apply the "
                  "gradient scaler. Reported grad norms may underflow; interpret "
                  "with caution.")
    steps_per_eval = len(eval_dataloader)
    steps_per_epoch = len(train_dataloader)
    best_dev_loss = float("inf")
    best_dev_f1 = -1.0
    best_dev_rouge_l = -1.0
    # fairytaleQA selects its best checkpoint by dev ROUGE-L instead of dev
    # F1 (dataset_config.file is the loader path, e.g. .../fairytaleQA.py).
    # Other datasets (QED, DialogSum) are unaffected and keep using dev F1.
    select_by_rouge_l = "fairytaleqa" in os.path.basename(dataset_config.file).lower()
    epochs_since_dev_loss_improved = 0
    early_stop_triggered = False
    val_loss = []
    val_ppl = []
    dev_f1_history = []
    dev_rouge_l_history = []

    # Persist the dev-split metadata so two runs with the same --seed can be
    # audited for determinism after the fact.
    if rank == 0 and dev_gen_dataloader is not None:
        try:
            os.makedirs(train_config.output_dir, exist_ok=True)
            dev_ds = dev_gen_dataloader.dataset
            # dev_ds carries 'example_id' since it's the raw HF dataset with
            # added prompt/tokenization columns. Use it as a stable identifier
            # (HF row indices change if upstream re-shuffles).
            try:
                example_ids = [int(x) for x in dev_ds['example_id']]
            except Exception:
                example_ids = list(range(len(dev_ds)))
            split_meta = {
                "seed": int(getattr(dataset_config, "dev_split_seed", train_config.seed)),
                "ratio": float(getattr(dataset_config, "dev_split_ratio", 0.1)),
                "n_dev": len(dev_ds),
                "example_ids_first10": example_ids[:10],
                "example_ids_all": example_ids,
            }
            with open(os.path.join(train_config.output_dir, "dev_split_indices.json"), "w") as fh:
                json.dump(split_meta, fh)
            print(f"[dev split] seed={split_meta['seed']} ratio={split_meta['ratio']} "
                  f"n_dev={split_meta['n_dev']} first10={split_meta['example_ids_first10']}")
        except Exception as _e:
            print(f"[dev split] failed to write dev_split_indices.json: {_e}")

    # Hard guard against accidentally pulling the held-out 1355-sample test set.
    if rank == 0 and eval_dataloader is not None:
        dev_len = len(eval_dataloader.dataset) if hasattr(eval_dataloader, 'dataset') else None
        if dev_len is not None:
            assert dev_len != 1355, (
                f"[guard] eval_dataloader has 1355 rows — that's the QED held-out test set, "
                "not the dev carve-out. Aborting before any training.")
            print(f"[dev/test guard] dev (eval) size = {dev_len}")

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        total_length = steps_per_epoch//gradient_accumulation_steps
        model.student.train() if train_config.distillation else model.train()
        with MemoryTrace() as memtrace:
            total_loss = 0.0
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader if not train_config.distillation else zip(train_dataloader, teacher_train_dataloader)):
                if train_config.distillation: batch = preprocess_distillation_batch(batch)
                for key in batch.keys():
                    if train_config.enable_fsdp or distil_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')

                with autocast():
                    if train_config.distillation:
                        student_output, teacher_output = model(**batch)
                        (
                            loss,
                            cross_loss,
                            dist_loss,
                            l1_comp,
                            kl_comp,
                            sinkhorn_comp,
                            diagnostics,
                        ) = distillation_loss(epoch, student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank, step=step)
                    else:
                        loss = model(**batch).loss
                        diagnostics = None

                # Per-component gradient-norm probe.
                # We can only run this cleanly at the START of an accumulation
                # cycle (so the gradient buffers are empty and the probe doesn't
                # contaminate the real accumulated gradient). Rank 0 only.
                ran_grad_probe = False
                if (
                    train_config.distillation
                    and rank == 0
                    and grad_probe_every > 0
                    and (step % grad_probe_every == 0)
                    and (step % gradient_accumulation_steps == 0)
                ):
                    student_module = model.student if hasattr(model, "student") else model
                    # NOTE: the probe tears down the autograd graph on its last
                    # backward, so afterwards we must re-run the forward pass
                    # before doing the real optimisation backward. To keep the
                    # step logically equivalent, we skip the real backward for
                    # this micro-batch (grad buffers are zero afterwards), then
                    # re-run forward + real backward below.
                    grad_norms = run_grad_norm_probe(
                        components=[
                            ("l1", l1_comp),
                            ("kl", kl_comp),
                            ("sinkhorn", sinkhorn_comp),
                        ],
                        student_params=list(student_module.parameters()),
                        optimizer=optimizer,
                    )
                    global_step_probe = epoch * steps_per_epoch + step
                    grad_norm_history["step"].append(global_step_probe)
                    grad_norm_history["l1"].append(grad_norms.get("l1", float("nan")))
                    grad_norm_history["kl"].append(grad_norms.get("kl", float("nan")))
                    grad_norm_history["sinkhorn"].append(grad_norms.get("sinkhorn", float("nan")))
                    print(f"[grad-probe step={global_step_probe}] "
                          + ", ".join(f"||grad {k}||={v:.3e}" for k, v in grad_norms.items()))
                    wandb.log({
                        "grad_norm/l1":       grad_norms.get("l1", float("nan")),
                        "grad_norm/kl":       grad_norms.get("kl", float("nan")),
                        "grad_norm/sinkhorn": grad_norms.get("sinkhorn", float("nan")),
                    })
                    # Re-run forward so the real backward has a fresh graph.
                    with autocast():
                        student_output, teacher_output = model(**batch)
                        (
                            loss,
                            cross_loss,
                            dist_loss,
                            l1_comp,
                            kl_comp,
                            sinkhorn_comp,
                            diagnostics,
                        ) = distillation_loss(epoch, student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank, step=step)
                    ran_grad_probe = True

                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update()
                else:
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update()

                if rank == 0:
                    if train_config.distillation:
                        global_step = epoch * steps_per_epoch + step
                        ce_val = float(cross_loss.detach().float().item())
                        dist_val = float(dist_loss.detach().float().item())
                        l1_val = float(l1_comp.detach().float().item())
                        kl_val = float(kl_comp.detach().float().item())
                        sink_val = float(sinkhorn_comp.detach().float().item())

                        loss_history["step"].append(global_step)
                        loss_history["ce"].append(ce_val)
                        loss_history["distil"].append(dist_val)
                        loss_history["l1"].append(l1_val)
                        loss_history["kl"].append(kl_val)
                        loss_history["sinkhorn"].append(sink_val)

                        diag = diagnostics if diagnostics is not None else {}
                        sinkhorn_raw        = float(diag.get("sinkhorn_raw_value", 0.0))
                        sinkhorn_per_sample = float(diag.get("sinkhorn_per_sample_raw_mean", 0.0))
                        cost_mean           = float(diag.get("cost_mean", 0.0))
                        cost_max            = float(diag.get("cost_max", 0.0))
                        cost_min            = float(diag.get("cost_min", 0.0))
                        loss_history["sinkhorn_raw"].append(sinkhorn_raw)
                        loss_history["sinkhorn_per_sample"].append(sinkhorn_per_sample)
                        loss_history["cost_mean"].append(cost_mean)
                        loss_history["cost_max"].append(cost_max)
                        loss_history["cost_min"].append(cost_min)

                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "cross_loss": cross_loss.detach().float(),
                            "distil_loss": dist_loss.detach().float(),
                            "distil_l1_component": l1_val,
                            "distil_kl_component": kl_val,
                            "distil_sinkhorn_component": sink_val,
                            # Diagnostics: pre-scaling Sinkhorn + cost stats.
                            "diag/sinkhorn_raw_pre_weight":     sinkhorn_raw,
                            "diag/sinkhorn_per_sample_raw":     sinkhorn_per_sample,
                            "diag/sinkhorn_epsilon":            float(diag.get("sinkhorn_epsilon", 0.1)),
                            "diag/sinkhorn_gamma":              float(diag.get("sinkhorn_gamma", 0.1)),
                            "diag/cost_mean":                   cost_mean,
                            "diag/cost_max":                    cost_max,
                            "diag/cost_min":                    cost_min,
                            # SpanOT-KD diagnostics: tells you what fraction of
                            # token positions were down-weighted (delta) vs at
                            # unit weight. If `span_low_weight_frac` is ~0 then
                            # `top_r` is too permissive; if it's ~1 then
                            # `top_r` is too aggressive and hardly any spans
                            # are getting full weight.
                            "diag/span_kd_enabled":             int(diag.get("span_kd_enabled", False)),
                            "diag/span_low_weight_frac":        float(diag.get("span_low_weight_frac", 0.0)),
                            "diag/span_position_weight_mean":   float(diag.get("span_position_weight_mean", 1.0)),
                            "teacher_loss": teacher_output.loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr'],
                            "grad_probe_ran": int(ran_grad_probe),
                        })
                    else:
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']
                        })

                lr_scheduler.step()
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

                # In-epoch save_step-driven checkpointing was removed: model
                # selection is now per-epoch on dev loss + dev F1 (see below).
            pbar.close()

        if epoch == 0:
            distillation_loss.on_epoch_end()

        if rank == 0: print(memtrace)
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / steps_per_epoch
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if rank == 0:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            wandb.log({
                "train_perplexity": train_perplexity,
                "train_epoch_loss": train_epoch_loss,
                "train_epoch_time": epoch_end_time
            })

        # ============================================================
        # End-of-epoch dev evaluation: dev loss (model selection +
        # early-stop signal) and dev F1 (best-model selection for
        # final test-set evaluation).
        # ============================================================
        if train_config.run_validation and eval_dataloader is not None:
            # --- Dev loss (teacher-forced; matches existing evaluation()). ---
            model.student.eval() if train_config.distillation else model.eval()
            eval_iter = (eval_dataloader if not train_config.distillation
                         else zip(eval_dataloader, teacher_eval_dataloader))
            eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss = evaluation(
                epoch, model, train_config, distil_config,
                eval_iter, steps_per_eval, local_rank,
            )
            model.student.train() if train_config.distillation else model.train()
            val_loss.append(eval_epoch_loss)
            val_ppl.append(eval_ppl)
            dev_loss_scalar = float(eval_epoch_loss.detach().float().item()) if torch.is_tensor(eval_epoch_loss) else float(eval_epoch_loss)

            if rank == 0:
                print(f"[dev] epoch {epoch+1}: dev_loss={dev_loss_scalar:.4f}, ppl={float(eval_ppl):.4f}")
                wandb_log = {
                    "dev/loss": dev_loss_scalar,
                    "dev/ppl": float(eval_ppl),
                    "epoch": epoch + 1,
                }
                if train_config.distillation:
                    wandb_log["dev/cross_loss"] = float(eval_cross_loss)
                    wandb_log["dev/distil_loss"] = float(eval_dist_loss)
                wandb.log(wandb_log)

            # --- Dev F1 (and, for fairytaleQA, dev ROUGE-L) — generative,
            # matches the benchmark driver. ---
            dev_f1 = None
            dev_rouge_l = None
            if dev_gen_dataloader is not None and student_tokenizer is not None:
                # Generation runs on the student. Determine its device.
                if train_config.enable_fsdp or distil_config.enable_fsdp:
                    gen_device = torch.device(f"cuda:{local_rank}")
                else:
                    gen_device = torch.device("cuda:0")
                student_for_gen = model.student if train_config.distillation else model
                if select_by_rouge_l:
                    dev_f1, dev_rouge_l = _compute_dev_f1(
                        student_for_gen, dev_gen_dataloader, dev_gen_answers,
                        student_tokenizer, train_config.dev_eval_max_new_tokens,
                        gen_device, rank, also_score_rouge_l=True,
                    )
                else:
                    dev_f1 = _compute_dev_f1(
                        student_for_gen, dev_gen_dataloader, dev_gen_answers,
                        student_tokenizer, train_config.dev_eval_max_new_tokens,
                        gen_device, rank,
                    )
                # Restore train mode regardless of who ran generation.
                if train_config.distillation:
                    model.student.train()
                else:
                    model.train()
                if rank == 0 and dev_f1 is not None:
                    print(f"[dev] epoch {epoch+1}: dev_f1={dev_f1:.4f}")
                    dev_f1_history.append(dev_f1)
                    wandb.log({"dev/f1": dev_f1, "epoch": epoch + 1})
                if rank == 0 and dev_rouge_l is not None:
                    print(f"[dev] epoch {epoch+1}: dev_rouge_l={dev_rouge_l:.4f}")
                    dev_rouge_l_history.append(dev_rouge_l)
                    wandb.log({"dev/rouge_l": dev_rouge_l, "epoch": epoch + 1})

            # --- Best-checkpoint saves: best_dev_loss, plus the generative
            # selection metric — best_dev_f1, or for fairytaleQA best_dev_rouge_l. ---
            if dev_loss_scalar < best_dev_loss:
                best_dev_loss = dev_loss_scalar
                epochs_since_dev_loss_improved = 0
                if train_config.save_model:
                    save_model(
                        model if not train_config.distillation else model.student,
                        optimizer, (steps_per_epoch * (epoch + 1)) - 1,
                        train_config, distil_config, fsdp_config, rank,
                        subdir_name="best_dev_loss",
                    )
                    if rank == 0:
                        print(f"[dev] new best dev_loss={best_dev_loss:.4f} → saved best_dev_loss/")
                        wandb.log({"dev/best_loss": best_dev_loss, "epoch": epoch + 1})
            else:
                epochs_since_dev_loss_improved += 1
                if rank == 0:
                    print(f"[dev] no dev_loss improvement ({epochs_since_dev_loss_improved}/{train_config.early_stopping_patience})")

            selection_value = dev_rouge_l if select_by_rouge_l else dev_f1
            selection_best_so_far = best_dev_rouge_l if select_by_rouge_l else best_dev_f1
            if selection_value is not None and selection_value > selection_best_so_far:
                if select_by_rouge_l:
                    best_dev_rouge_l = selection_value
                else:
                    best_dev_f1 = selection_value
                subdir_name = "best_dev_rouge_l" if select_by_rouge_l else "best_dev_f1"
                metric_label = "dev_rouge_l" if select_by_rouge_l else "dev_f1"
                if train_config.save_model:
                    save_model(
                        model if not train_config.distillation else model.student,
                        optimizer, (steps_per_epoch * (epoch + 1)) - 1,
                        train_config, distil_config, fsdp_config, rank,
                        subdir_name=subdir_name,
                    )
                    if rank == 0:
                        print(f"[dev] new best {metric_label}={selection_value:.4f} → saved {subdir_name}/")
                        wandb.log({f"dev/best_{metric_label}": selection_value, "epoch": epoch + 1})

            clear_gpu_cache(rank)

            # --- Early stopping on dev loss (epochs without improvement). ---
            if (train_config.early_stopping_patience > 0
                    and epochs_since_dev_loss_improved >= train_config.early_stopping_patience):
                if rank == 0:
                    print(f"[dev] early stopping triggered after epoch {epoch+1} "
                          f"({epochs_since_dev_loss_improved} epochs without dev_loss improvement)")
                early_stop_triggered = True
                break

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.run_validation and len(val_loss) > 0:
        results["best_dev_loss"] = best_dev_loss
        results["dev_loss_history"] = [float(x.detach().float().item()) if torch.is_tensor(x) else float(x) for x in val_loss]
        if best_dev_f1 > -1.0:
            results["best_dev_f1"] = best_dev_f1
            results["dev_f1_history"] = dev_f1_history
        if best_dev_rouge_l > -1.0:
            results["best_dev_rouge_l"] = best_dev_rouge_l
            results["dev_rouge_l_history"] = dev_rouge_l_history
        results["early_stop_triggered"] = early_stop_triggered

    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    # Save loss curve plot (rank 0 only).
    if rank == 0 and (len(train_loss) > 0 or len(val_loss) > 0):
        plot_dir = train_config.output_dir if train_config.output_dir else "."
        plot_path = os.path.join(plot_dir, "loss_curve.png")
        try:
            _to_float = lambda lst: [float(x.detach().float().item()) if torch.is_tensor(x) else float(x) for x in lst]
            plot_loss_curves(
                loss_history, plot_path,
                epoch_train_loss=_to_float(train_loss) if train_loss else None,
                epoch_dev_loss=_to_float(val_loss) if val_loss else None,
                epoch_dev_f1=dev_f1_history if dev_f1_history else None,
            )
        except Exception as e:
            print(f"[train] Failed to save loss curve plot: {e}")

        if len(grad_norm_history["step"]) > 0:
            grad_plot_path = os.path.join(plot_dir, "grad_norm_probe.png")
            try:
                plot_grad_norm_curves(grad_norm_history, grad_plot_path)
            except Exception as e:
                print(f"[train] Failed to save grad-norm plot: {e}")

    # Close the W&B run cleanly so the next train() call (e.g. inside a
    # span_top_r sweep) starts a fresh run instead of appending to this one.
    if rank == 0:
        try:
            wandb.finish()
        except Exception as _e:
            print(f"[train] wandb.finish() raised: {_e}")

    return results