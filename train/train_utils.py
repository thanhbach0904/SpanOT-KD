import os
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
from train.span_match import SpanMatchEvaluator


def _moving_average(values, window):
    """Simple centered moving average for smoothing noisy loss curves."""
    if window <= 1 or len(values) < window:
        return list(values)
    import numpy as np
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def _component_grad_norm(component_loss, params):
    """Compute the L2 norm of gradients produced by a single loss component.

    IMPORTANT caveats:
      * Caller is responsible for ensuring ``param.grad`` is zero before this
        call — otherwise the norm reflects accumulation from prior backwards.
      * Uses ``retain_graph=True`` so subsequent backwards on OTHER components
        of the same forward pass are still possible. The caller must eventually
        do one backward *without* ``retain_graph`` (or free the graph) to avoid
        leaking activations.
      * For FSDP the returned norm is the LOCAL (sharded) norm. That's fine for
        ratio-style comparisons between components on the same rank.
    """
    component_loss.backward(retain_graph=True)
    sq_sum = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            # .float() guards against fp16 overflow when squaring small values.
            sq_sum += float(g.float().norm(2).item()) ** 2
    return sq_sum ** 0.5


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


def plot_loss_curves(loss_history, output_path, smoothing_window=50):
    """Plot CE/distillation losses, components, and Sinkhorn diagnostics.

    Panels (top to bottom):
      1. CE vs total distillation loss.
      2. Three weighted components (L1/OT, KL, Sinkhorn) that sum to panel 1's
         distillation curve.
      3. Sinkhorn raw value (before the outer * 0.1 gamma and before
         distillation_weight) on a log y-axis alongside the Sinkhorn component
         and, if they were recorded, the cost-matrix statistics.
    """
    steps = loss_history["step"]
    if len(steps) == 0:
        print(f"[plot_loss_curves] No steps recorded, skipping plot.")
        return

    have_diag = len(loss_history.get("sinkhorn_raw", [])) == len(steps)
    n_panels = 3 if have_diag else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: CE vs total distillation ---
    ax = axes[0]
    for key, label, color in [
        ("ce", "CE loss", "tab:blue"),
        ("distil", "Distillation loss (total)", "tab:orange"),
    ]:
        raw = loss_history[key]
        ax.plot(steps, raw, alpha=0.25, color=color, linewidth=0.8)
        smooth = _moving_average(raw, smoothing_window)
        ax.plot(steps, smooth, color=color, linewidth=1.8,
                label=f"{label} (MA{smoothing_window})")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-entropy vs. distillation loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: 3 components of distillation loss ---
    ax = axes[1]
    for key, label, color in [
        ("l1", "L1/OT component", "tab:green"),
        ("kl", "KL component (x0.1)", "tab:red"),
        ("sinkhorn", "Sinkhorn component (x0.1)", "tab:purple"),
    ]:
        raw = loss_history[key]
        ax.plot(steps, raw, alpha=0.25, color=color, linewidth=0.8)
        smooth = _moving_average(raw, smoothing_window)
        ax.plot(steps, smooth, color=color, linewidth=1.8,
                label=f"{label} (MA{smoothing_window})")
    ax.set_ylabel("Loss")
    ax.set_title("Distillation loss components (weighted, sum to total above)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Sinkhorn raw + cost-matrix diagnostics ---
    if have_diag:
        ax = axes[2]
        diag_curves = [
            ("sinkhorn_raw",         "Sinkhorn raw (pre-gamma, pre-w)", "tab:purple"),
            ("sinkhorn_per_sample",  "Sinkhorn per-sample raw mean",    "tab:pink"),
            ("cost_mean",            "C mean",                          "tab:brown"),
            ("cost_max",             "C max",                           "tab:olive"),
            ("cost_min",             "C min",                           "tab:gray"),
        ]
        plotted_any = False
        for key, label, color in diag_curves:
            raw = loss_history.get(key, [])
            if len(raw) != len(steps):
                continue
            # Guard: log-scale blows up on <=0 values; clip for display only.
            import numpy as np
            raw_arr = np.asarray(raw, dtype=float)
            raw_arr = np.where(raw_arr > 0, raw_arr, np.nan)
            ax.plot(steps, raw_arr, alpha=0.25, color=color, linewidth=0.8)
            smooth = _moving_average(np.nan_to_num(raw_arr, nan=0.0).tolist(), smoothing_window)
            ax.plot(steps, smooth, color=color, linewidth=1.6,
                    label=f"{label} (MA{smoothing_window})")
            plotted_any = True
        if plotted_any:
            ax.set_yscale("log")
        ax.set_ylabel("Value (log scale)")
        ax.set_title("Sinkhorn diagnostics: raw LSD value and cost-matrix stats")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Training step (global)")

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

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, teacher_train_dataloader=None, teacher_eval_dataloader=None, fsdp_config=None, local_rank=None, rank=None, f=1):
    # Weights & Biases tracking system initialization.
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if rank == 0:
        wandb.init(
            project=f"llm_distillation_{dataset_config.file.split('/')[-1][:-3]}",
            name=f"{train_config.model_name.split('/')[-1]}-{model.teacher.name_or_path.split('/')[-1]}-d{distil_config.distil_factor}-t{distil_config.teacher_temperature}{distil_config.student_temperature}" if train_config.distillation else f"{train_config.model_name.split('/')[-1]}",
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

    # Init distillation loss if distillation is enabled
    if train_config.distillation:
        distillation_loss = DistillationLoss(distillation_weight=distil_config.distil_factor, student_temperature=distil_config.student_temperature, teacher_temperature=distil_config.teacher_temperature, skip_student_eos=True, debug=False, debug_rank=0, tokenizer_student=model.student.name_or_path, tokenizer_teacher=model.teacher.name_or_path, f=f)

    # Span-match evaluator (opt-in via SPAN_MATCH_EVAL=1).
    # Set SPAN_MATCH_MAX_BATCHES=N to cap the number of batches per evaluation
    # pass (default 500).  Each pass iterates over the train dataloader from the
    # beginning, so it does not disturb the outer training loop's iterator.
    _span_eval_enabled = int(os.environ.get("SPAN_MATCH_EVAL", "0")) > 0
    _span_max_batches  = int(os.environ.get("SPAN_MATCH_MAX_BATCHES", "500"))
    span_evaluator = None
    if train_config.distillation and _span_eval_enabled and rank == 0:
        span_evaluator = SpanMatchEvaluator(
            student_tokenizer_path=model.student.name_or_path,
            teacher_tokenizer_path=model.teacher.name_or_path,
            output_dir=train_config.output_dir or ".",
            max_eval_batches=_span_max_batches,
            rank=rank,
        )
        print(f"[train] Span-match evaluator ENABLED (max_eval_batches={_span_max_batches}). "
              "Runs at pre_training and post_training.")
    elif train_config.distillation and rank == 0:
        print("[train] Span-match evaluator disabled (set SPAN_MATCH_EVAL=1 to enable).")

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
    best_val_loss = float("inf")

    # Phase 1: pre-training span-match baseline (vanilla student, no gradient).
    if span_evaluator is not None and teacher_train_dataloader is not None:
        try:
            span_evaluator.run_evaluation(
                model, train_dataloader, teacher_train_dataloader,
                phase_label="pre_training", global_step=0, local_rank=local_rank,
            )
        except Exception as _e:
            print(f"[train] Span-match pre_training eval failed: {_e}")

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
                        ) = distillation_loss(epoch, student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank)
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
                        ) = distillation_loss(epoch, student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank)
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

                if (train_config.run_validation and ((step+1) % train_config.save_step == 0 or step+1 == steps_per_epoch)):
                    if rank == 0: print("Running evaluation...")
                    # model.eval()
                    # eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss = evaluation(
                    #     model, train_config, distil_config, 
                    #     eval_dataloader if not train_config.distillation else zip(eval_dataloader, teacher_eval_dataloader),
                    #     steps_per_eval, local_rank, epoch)
                    # model.student.train() if train_config.distillation else model.train()
                    # val_loss.append(eval_epoch_loss)
                    # val_ppl.append(eval_ppl)
                    
                    # if rank == 0:
                    #     print(f"Perplexity {eval_ppl}, loss {eval_epoch_loss}")
                    #     if train_config.distillation:
                    #         wandb.log({
                    #             "eval_ppl": eval_ppl,
                    #             "eval_epoch_loss": eval_epoch_loss,
                    #             "eval_cross_loss": eval_cross_loss,
                    #             "eval_dist_loss": eval_dist_loss
                    #         })
                    #     else:
                    #         wandb.log({
                    #             "eval_ppl": eval_ppl,
                    #             "eval_epoch_loss": eval_epoch_loss,
                    #         })

                    # if eval_epoch_loss < best_val_loss or train_config.save_all:
                    if True:
                        # if eval_epoch_loss < best_val_loss:
                        #     best_val_loss = eval_epoch_loss
                            # if rank == 0:
                            #     print(f"best eval loss is {best_val_loss}")
                        if train_config.save_model:
                            checkpoint_start_time = time.perf_counter()
                            save_model(
                                model if not train_config.distillation else model.student, 
                                optimizer, ((steps_per_epoch*epoch)+step), train_config, distil_config, fsdp_config, rank
                            )
                            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                            checkpoint_times.append(checkpoint_end_time)
                    clear_gpu_cache(rank)
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

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    # if train_config.run_validation:
    #     avg_eval_prep = sum(val_ppl)/len(val_ppl)
    #     avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    # if train_config.run_validation:
    #     results['avg_eval_prep'] = avg_eval_prep
    #     results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    # Phase 3: post-training span-match evaluation and history plot.
    if span_evaluator is not None and teacher_train_dataloader is not None:
        try:
            span_evaluator.run_evaluation(
                model, train_dataloader, teacher_train_dataloader,
                phase_label="post_training",
                global_step=train_config.num_epochs * steps_per_epoch - 1,
                local_rank=local_rank,
            )
        except Exception as _e:
            print(f"[train] Span-match post_training eval failed: {_e}")
        try:
            span_evaluator.plot_history()
        except Exception as _e:
            print(f"[train] Span-match plot_history failed: {_e}")

    # Save loss curve plot (rank 0 only, distillation runs only).
    if rank == 0 and train_config.distillation and len(loss_history["step"]) > 0:
        plot_dir = train_config.output_dir if train_config.output_dir else "."
        plot_path = os.path.join(plot_dir, "loss_curve.png")
        try:
            plot_loss_curves(loss_history, plot_path)
        except Exception as e:
            print(f"[train] Failed to save loss curve plot: {e}")

        if len(grad_norm_history["step"]) > 0:
            grad_plot_path = os.path.join(plot_dir, "grad_norm_probe.png")
            try:
                plot_grad_norm_curves(grad_norm_history, grad_plot_path)
            except Exception as e:
                print(f"[train] Failed to save grad-norm plot: {e}")

    return results