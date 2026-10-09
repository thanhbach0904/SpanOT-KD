import os
import gc
import json
import shutil
import argparse
import random
import torch

from configs import dataset as DATA_CONFIG
from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from configs import distillation_config as DISTIL_CONFIG

from train.train_utils import train
from configs.configs_utils import update_config
from data.data_utils import (get_dataloader, get_distillation_dataloader)
from train.tools import (setup, setup_environ_flags, clear_gpu_cache)
from models.models_utils import (get_model, get_distillation_models, get_optimizer)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Discrete grid for the optional span_top_r sweep. The methodology paper
# only recommends a small set of plausible top-r values, so we pin the
# grid here rather than continuous search.
SPAN_TOP_R_SWEEP_VALUES = [0.3, 0.5, 0.7]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset.file", type=str, required=True, help="Path to the dataset loader")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size_training", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--distillation_config_model_name", type=str, help="Model name for distillation")
    parser.add_argument("--distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--distillation_config_enable_fsdp", action="store_true", help="Enable FSDP for distillation")
    parser.add_argument("--distillation_config_pure_bf16", action="store_true", help="Use pure BF16 for distillation")
    parser.add_argument("--distillation_config_distil_factor", type=float, default=1.5, help="Distillation factor")
    parser.add_argument("--save_step", type=int, default=100,
                        help="DEPRECATED — no longer used; in-epoch checkpointing was removed in favor of per-epoch dev-loss/dev-F1 selection.")
    parser.add_argument("--f", type=int, default=1, help="method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dev-set carve-out and early stopping.
    parser.add_argument("--dev_split_ratio", type=float, default=0.1,
                        help="Fraction of on-disk train set held out as dev (seeded). The on-disk validation split is reserved as final test.")
    parser.add_argument("--dev_split_seed", type=int, default=None,
                        help="Seed for dev carve-out. Defaults to --seed if unset, so two runs with the same --seed get identical dev rows.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Epochs of no dev-loss improvement before stopping. 0 disables.")
    parser.add_argument("--dev_eval_max_new_tokens", type=int, default=64,
                        help="max_new_tokens for greedy generation on dev (F1).")
    parser.add_argument("--dev_gen_batch_size", type=int, default=4,
                        help="Batch size for dev generation dataloader.")
    # SpanOT-KD: span-selective extension of MultiLevelOT (see
    # train/span_ot.py). When `--span_kd_enabled` is omitted the run is
    # byte-for-byte identical to vanilla MultiLevelOT.
    parser.add_argument("--distillation_config_span_kd_enabled", action="store_true",
                        help="Enable SpanOT-KD: span-selective per-token weighting of HAD/SL/SD (default: off, recovers MultiLevelOT)")
    parser.add_argument("--distillation_config_span_aggregation", type=str, default="mean", choices=["mean", "sum"],
                        help="Per-span entropy-gap aggregation: 'mean' (Eq. 10) or 'sum' (Eq. 11)")
    parser.add_argument("--distillation_config_span_top_r", type=float, default=0.5,
                        help="Fraction of positive-gap spans (sorted by entropy gap) deemed high-priority. r=1.0 is NOT MultiLevelOT: gap<=0 spans still get delta")
    parser.add_argument("--distillation_config_span_low_delta", type=float, default=0.1,
                        help="Down-weight applied to non-top-r spans. Methodology recommends (0, 0.1]. delta=1.0 collapses to MultiLevelOT")
    parser.add_argument("--distillation_config_span_select_mode", type=str, default="entropy", choices=["entropy", "random", "matched"],
                        help="'entropy': SpanOT-KD. 'random': control with the same per-sample token mass at weight 1.0 but random span positions. "
                             "'matched': control with the same per-sample mean weight as 'entropy', spread uniformly over the answer")
    parser.add_argument("--distillation_config_span_random_pool", type=str, default="active", choices=["active", "pos"],
                        help="Only for select_mode=random: draw from all aligned spans ('active') or only spans with gap>0 ('pos')")
    # Optional span_top_r sweep (per seed). When enabled the script trains 3
    # students consecutively with span_top_r in SPAN_TOP_R_SWEEP_VALUES, then
    # selects the best by dev F1 and by dev CE loss (these can differ) and
    # copies both winners to the canonical {output_dir}/best_dev_f1 and
    # {output_dir}/best_dev_loss directories for downstream test evaluation.
    parser.add_argument("--distillation_config_span_top_r_sweep", action="store_true",
                        help=("Sweep span_top_r over {0.3, 0.5, 0.7} (per seed) and pick the best by dev F1 "
                              "and the best by dev CE loss — possibly two different r values. Requires "
                              "--distillation and --distillation_config_span_kd_enabled."))
    # Optional extra sweep points appended to SPAN_TOP_R_SWEEP_VALUES (e.g. 1.0,
    # which isolates the gap>0 filter from the top-r ranking). Default off.
    # Extras are trained and reported in sweep_summary.json but never compete
    # in dev winner selection (see _run_sweep._best).
    parser.add_argument("--distillation_config_span_top_r_extra", type=float, nargs="*", default=[],
                        help="Extra span_top_r values appended to the sweep grid (default: none)")
    # Dry-run only: cap train / dev / dev-gen sets to the first N rows. 0 = off
    # (default, identical to before).
    parser.add_argument("--max_samples", type=int, default=0,
                        help="DRY RUN ONLY: keep the first N rows of train, dev and dev-gen sets (0 = off)")
    return parser.parse_args()


def _sweep_values(args):
    extra = [float(r) for r in getattr(args, "distillation_config_span_top_r_extra", []) or []]
    return SPAN_TOP_R_SWEEP_VALUES + [r for r in extra if r not in SPAN_TOP_R_SWEEP_VALUES]


def _seed_everything(seed: int):
    """Re-apply the global seeds. Called at the start of every training run
    inside the sweep so that, modulo span_top_r, the runs are deterministic."""
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def _free_after_run(*objs):
    """Aggressively drop references and free GPU memory between sweep runs."""
    for _ in objs:
        pass  # references go out of caller frame after `del` there
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _copytree_overwrite(src: str, dst: str):
    """Copy `src` to `dst`, overwriting any existing `dst`."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _execute_training_run(args, train_config, fsdp_config, distil_config, data_config,
                          rank, local_rank, run_output_dir, r_override=None,
                          sweep_tag=None, wandb_name_suffix=None):
    """Build models + data + optimizer, then call train(). Returns results dict.

    The caller decides where checkpoints live (`run_output_dir`) and which
    span_top_r the run uses (`r_override`). When `r_override` is None the
    value supplied via CLI is used unchanged — i.e. single-run behaviour is
    untouched.
    """
    train_config.output_dir = run_output_dir
    if rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)

    # Re-seed before each run so the only thing that varies across sweep
    # iterations is span_top_r. Without this the second/third runs would
    # inherit a perturbed RNG state from the first.
    _seed_everything(train_config.seed)

    if train_config.distillation:
        # Re-apply distillation knobs each run; span_top_r is the only one
        # that may legitimately change between sweep iterations.
        distil_config.model_name = args.distillation_config_model_name
        distil_config.pure_bf16 = args.distillation_config_pure_bf16
        distil_config.enable_fsdp = args.distillation_config_enable_fsdp
        distil_config.distil_factor = args.distillation_config_distil_factor
        distil_config.span_kd_enabled = args.distillation_config_span_kd_enabled
        distil_config.span_aggregation = args.distillation_config_span_aggregation
        distil_config.span_top_r = (
            r_override if r_override is not None else args.distillation_config_span_top_r
        )
        distil_config.span_low_delta = args.distillation_config_span_low_delta
        distil_config.span_select_mode = args.distillation_config_span_select_mode
        distil_config.span_random_pool = args.distillation_config_span_random_pool
        if rank == 0:
            tag = f" [{sweep_tag}]" if sweep_tag else ""
            if distil_config.span_kd_enabled:
                print(f"[SpanOT-KD]{tag} ENABLED", flush=True)
                print(f"  span_aggregation : {distil_config.span_aggregation}", flush=True)
                print(f"  span_top_r       : {distil_config.span_top_r}", flush=True)
                print(f"  span_low_delta   : {distil_config.span_low_delta}", flush=True)
                print(f"  span_select_mode : {distil_config.span_select_mode}", flush=True)
                print(f"  span_random_pool : {distil_config.span_random_pool}", flush=True)
            else:
                print(f"[SpanOT-KD]{tag} disabled (vanilla MultiLevelOT)", flush=True)
        student_tokenizer, teacher_tokenizer, model = get_distillation_models(
            train_config, distil_config, fsdp_config, rank, vars(args)
        )
        tokenizer = None
    else:
        tokenizer, model = get_model(train_config, fsdp_config, rank, vars(args))
        student_tokenizer = teacher_tokenizer = None

    if rank == 0: print(model)
    if rank == 0: print("[checkpoint] models loaded OK", flush=True)

    # Load Data
    data_config.encoder_decoder = train_config.encoder_decoder
    if rank == 0: print("[checkpoint] starting data loading...", flush=True)
    dev_gen_dataloader = None
    dev_gen_answers = None
    teacher_train_dataloader = None
    teacher_eval_dataloader = None
    if train_config.distillation:
        (train_dataloader, teacher_train_dataloader,
         eval_dataloader, teacher_eval_dataloader,
         dev_gen_dataloader, dev_gen_answers) = get_distillation_dataloader(
            data_config, train_config, distil_config,
            student_tokenizer, teacher_tokenizer, rank)
    else:
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)
    if rank == 0: print(f"[checkpoint] data loaded OK — train batches: {len(train_dataloader)}", flush=True)

    optimizer = get_optimizer(model, train_config, fsdp_config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_config.lr,
        epochs=train_config.num_epochs, steps_per_epoch=len(train_dataloader),
        pct_start=train_config.pct_start, div_factor=train_config.div_factor,
        final_div_factor=train_config.final_div_factor,
    )

    f = train_config.f
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        distil_config,
        data_config,
        teacher_train_dataloader if train_config.distillation else None,
        teacher_eval_dataloader if train_config.distillation else None,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if (train_config.enable_fsdp or distil_config.enable_fsdp) else None,
        rank,
        f,
        dev_gen_dataloader=dev_gen_dataloader,
        dev_gen_answers=dev_gen_answers,
        student_tokenizer=(student_tokenizer if train_config.distillation else tokenizer),
        wandb_name_suffix=wandb_name_suffix,
    )

    # Persist dev metrics + w_eff for every run (sweep or not) so the
    # aggregation script never depends on stdout. Tensors -> float.
    if rank == 0:
        def _jsonable(v):
            if torch.is_tensor(v):
                return float(v.detach().float().item())
            if isinstance(v, dict):
                return {str(k): _jsonable(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_jsonable(x) for x in v]
            return v
        with open(os.path.join(run_output_dir, "run_results.json"), "w") as fh:
            json.dump({"span_top_r": getattr(distil_config, "span_top_r", None),
                       "seed": int(train_config.seed),
                       **_jsonable(results)}, fh, indent=2)

    # Drop heavy refs and reclaim GPU memory before the next sweep run.
    del model, optimizer, scheduler
    del train_dataloader, eval_dataloader
    if teacher_train_dataloader is not None: del teacher_train_dataloader
    if teacher_eval_dataloader is not None: del teacher_eval_dataloader
    if dev_gen_dataloader is not None: del dev_gen_dataloader
    _free_after_run()

    return results


def _run_sweep(args, train_config, fsdp_config, distil_config, data_config,
               rank, local_rank, base_output_dir):
    """Train one student per value in _sweep_values(args), then pick the
    best by dev CE loss and the best by the generative selection metric
    (dev F1, or for fairytaleQA dev ROUGE-L) — these can differ — and copy
    both winning checkpoints to canonical paths under `base_output_dir`."""
    # Mirrors the dataset gate in train.train_utils.train(): fairytaleQA
    # selects checkpoints by dev ROUGE-L, other datasets keep dev F1.
    select_by_rouge_l = "fairytaleqa" in os.path.basename(data_config.file).lower()
    gen_metric_key = "best_dev_rouge_l" if select_by_rouge_l else "best_dev_f1"
    gen_metric_label = "dev_rouge_l" if select_by_rouge_l else "dev_f1"

    if not train_config.distillation:
        raise ValueError(
            "[sweep] --distillation_config_span_top_r_sweep requires --distillation.")
    if not args.distillation_config_span_kd_enabled:
        # span_top_r is a no-op without SpanOT-KD enabled — running 3 identical
        # trainings would just burn compute.
        raise ValueError(
            "[sweep] --distillation_config_span_top_r_sweep requires "
            "--distillation_config_span_kd_enabled (otherwise span_top_r has no effect).")

    sweep_values = _sweep_values(args)
    if rank == 0:
        print(f"\n[sweep] === Span-top-r sweep enabled — values: {sweep_values} ===\n", flush=True)
        os.makedirs(base_output_dir, exist_ok=True)

    sweep_results = {}
    for r in sweep_values:
        run_dir = os.path.join(base_output_dir, f"r_{r:.1f}")
        if rank == 0:
            print(f"\n[sweep] >>> Starting run for span_top_r = {r}  →  {run_dir}\n", flush=True)
        results = _execute_training_run(
            args, train_config, fsdp_config, distil_config, data_config,
            rank, local_rank,
            run_output_dir=run_dir, r_override=r,
            sweep_tag=f"sweep r={r}",
            wandb_name_suffix=f"-r{r:.1f}",
        )
        sweep_results[r] = results
        if rank == 0:
            best_dev_loss = results.get("best_dev_loss", float("nan"))
            best_dev_gen = results.get(gen_metric_key, None)
            print(f"\n[sweep] <<< Finished run for span_top_r = {r}  →  "
                  f"best_dev_loss={best_dev_loss}, {gen_metric_label}={best_dev_gen}\n", flush=True)

    # Winner selection is rank-0 only (it's pure file I/O on cached metrics).
    if rank == 0:
        def _best(metric_key, mode):
            assert mode in ("min", "max")
            cand = []
            for r, res in sweep_results.items():
                # Extra points (--distillation_config_span_top_r_extra) are
                # analysis-only: excluding them keeps winner selection
                # best-of-the-same-3 for every arm, with or without extras.
                if r not in SPAN_TOP_R_SWEEP_VALUES: continue
                v = res.get(metric_key, None)
                if v is None: continue
                try:
                    cand.append((r, float(v)))
                except (TypeError, ValueError):
                    continue
            if not cand:
                return None, None
            picker = min if mode == "min" else max
            return picker(cand, key=lambda x: x[1])

        winner_r_loss, best_dev_loss = _best("best_dev_loss", "min")
        winner_r_gen, best_dev_gen = _best(gen_metric_key, "max")

        # Build a human-readable / machine-readable summary.
        per_run = {}
        for r in sweep_values:
            res = sweep_results.get(r, {}) or {}
            bdl = res.get("best_dev_loss", None)
            bdg = res.get(gen_metric_key, None)
            per_run[f"{r:.1f}"] = {
                "run_dir": os.path.join(base_output_dir, f"r_{r:.1f}"),
                "best_dev_loss": float(bdl) if bdl is not None else None,
                gen_metric_key:  float(bdg) if bdg is not None else None,
                "w_eff_valid_overall": res.get("w_eff_valid_overall", None),
            }

        winner_dev_loss_src = (
            os.path.join(base_output_dir, f"r_{winner_r_loss:.1f}", "best_dev_loss")
            if winner_r_loss is not None else None
        )
        winner_dev_gen_src = (
            os.path.join(base_output_dir, f"r_{winner_r_gen:.1f}", gen_metric_key)
            if winner_r_gen is not None else None
        )
        winner_dev_loss_dst = os.path.join(base_output_dir, "best_dev_loss")
        winner_dev_gen_dst = os.path.join(base_output_dir, gen_metric_key)

        summary = {
            "sweep_r_values": sweep_values,
            "seed": int(train_config.seed),
            "per_run": per_run,
            "winner_dev_loss": {
                "r": winner_r_loss,
                "best_dev_loss": best_dev_loss,
                "src": winner_dev_loss_src,
                "dst": winner_dev_loss_dst,
            },
            f"winner_{gen_metric_label}": {
                "r": winner_r_gen,
                gen_metric_key: best_dev_gen,
                "src": winner_dev_gen_src,
                "dst": winner_dev_gen_dst,
            },
        }

        print("\n[sweep] === Sweep summary ===", flush=True)
        for r in sweep_values:
            entry = per_run[f"{r:.1f}"]
            print(f"  r={r}: best_dev_loss={entry['best_dev_loss']}, "
                  f"{gen_metric_label}={entry[gen_metric_key]}", flush=True)
        print(f"  winner dev_loss: r={winner_r_loss} (dev_loss={best_dev_loss})", flush=True)
        print(f"  winner {gen_metric_label}  : r={winner_r_gen} ({gen_metric_label}={best_dev_gen})", flush=True)

        # Copy winners to canonical paths so downstream test eval (which
        # expects {output_dir}/best_dev_loss and /{gen_metric_key}) just works.
        # The per-r subdirectories are kept on disk for auditing.
        if train_config.save_model:
            if winner_dev_loss_src and os.path.isdir(winner_dev_loss_src):
                _copytree_overwrite(winner_dev_loss_src, winner_dev_loss_dst)
                print(f"[sweep] Copied dev-loss winner: {winner_dev_loss_src} → {winner_dev_loss_dst}", flush=True)
            else:
                print(f"[sweep] WARNING: dev-loss winner src missing ({winner_dev_loss_src}) — no canonical copy made.", flush=True)
            if winner_dev_gen_src and os.path.isdir(winner_dev_gen_src):
                _copytree_overwrite(winner_dev_gen_src, winner_dev_gen_dst)
                print(f"[sweep] Copied {gen_metric_label} winner: {winner_dev_gen_src} → {winner_dev_gen_dst}", flush=True)
            else:
                print(f"[sweep] WARNING: {gen_metric_label} winner src missing ({winner_dev_gen_src}) — no canonical copy made.", flush=True)

        summary_path = os.path.join(base_output_dir, "sweep_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[sweep] Wrote sweep summary: {summary_path}", flush=True)

    return sweep_results


def main():
    args = parse_args()

    train_config, fsdp_config, distil_config, data_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG(), DATA_CONFIG()
    update_config((train_config, fsdp_config, data_config), **vars(args))
    update_config((distil_config), isSubmodule=True, **vars(args))

    # Dev-split seed: default to train seed so determinism is single-knob.
    if args.dev_split_seed is None:
        data_config.dev_split_seed = train_config.seed
    else:
        data_config.dev_split_seed = args.dev_split_seed
    data_config.dev_split_ratio = args.dev_split_ratio

    _seed_everything(train_config.seed)

    if train_config.enable_fsdp or distil_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else:
        rank = 0
        local_rank = None

    if rank == 0:
        print(f"[seed] {train_config.seed}", flush=True)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    base_output_dir = train_config.output_dir
    sweep_enabled = bool(getattr(args, "distillation_config_span_top_r_sweep", False))

    if sweep_enabled:
        _run_sweep(args, train_config, fsdp_config, distil_config, data_config,
                   rank, local_rank, base_output_dir)
    else:
        results = _execute_training_run(
            args, train_config, fsdp_config, distil_config, data_config,
            rank, local_rank,
            run_output_dir=base_output_dir, r_override=None,
        )
        if rank == 0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    main()
