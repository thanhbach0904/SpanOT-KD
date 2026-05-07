import os
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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset.file", type=str, required=True, help="Path to the dataset loader")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size_training", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--distillation_config_model_name", type=str, help="Model name for distillation")
    parser.add_argument("--distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--distillation_config_enable_fsdp", action="store_true", help="Enable FSDP for distillation")
    parser.add_argument("--distillation_config_pure_bf16", action="store_true", help="Use pure BF16 for distillation")
    parser.add_argument("--distillation_config_distil_factor", type=float, default=1.5, help="Distillation factor")
    parser.add_argument("--save_step", type=int, default=100, help="Save step")
    parser.add_argument("--f", type=int, default=1, help="method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug_tokenization", action="store_true", help="Run tokenization alignment check on a few batches then exit")
    parser.add_argument("--debug_max_batches", type=int, default=2, help="Number of batches to check when --debug_tokenization is set")

    # SpanOT-KD: span-selective extension of MultiLevelOT (see
    # train/span_ot.py). When `--span_kd_enabled` is omitted the run is
    # byte-for-byte identical to vanilla MultiLevelOT.
    parser.add_argument("--distillation_config_span_kd_enabled", action="store_true",
                        help="Enable SpanOT-KD: span-selective per-token weighting of HAD/SL/SD (default: off, recovers MultiLevelOT)")
    parser.add_argument("--distillation_config_span_aggregation", type=str, default="mean", choices=["mean", "sum"],
                        help="Per-span entropy-gap aggregation: 'mean' (Eq. 10) or 'sum' (Eq. 11)")
    parser.add_argument("--distillation_config_span_top_r", type=float, default=0.5,
                        help="Fraction of spans (sorted by entropy gap) deemed high-priority. r=1.0 collapses to MultiLevelOT")
    parser.add_argument("--distillation_config_span_low_delta", type=float, default=0.1,
                        help="Down-weight applied to non-top-r spans. Methodology recommends (0, 0.1]. delta=1.0 collapses to MultiLevelOT")
    return parser.parse_args()

def main():
    args = parse_args()

    train_config, fsdp_config, distil_config, data_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG(), DATA_CONFIG()
    update_config((train_config, fsdp_config, data_config), **vars(args))
    update_config((distil_config), isSubmodule=True, **vars(args))
    #print(train_config)
    #print(fsdp_config)
    #print(data_config)

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or distil_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else: rank = 0

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load Model and Tokenizer
    if train_config.distillation:
        distil_config.model_name = args.distillation_config_model_name
        distil_config.pure_bf16 = args.distillation_config_pure_bf16
        distil_config.enable_fsdp = args.distillation_config_enable_fsdp
        distil_config.distil_factor = args.distillation_config_distil_factor
        # SpanOT-KD knobs are surfaced via CLI but applied here so they show
        # up in the same place as the rest of the distillation config.
        distil_config.span_kd_enabled = args.distillation_config_span_kd_enabled
        distil_config.span_aggregation = args.distillation_config_span_aggregation
        distil_config.span_top_r = args.distillation_config_span_top_r
        distil_config.span_low_delta = args.distillation_config_span_low_delta
        if rank == 0:
            if distil_config.span_kd_enabled:
                print("[SpanOT-KD] ENABLED", flush=True)
                print(f"  span_aggregation : {distil_config.span_aggregation}", flush=True)
                print(f"  span_top_r       : {distil_config.span_top_r}", flush=True)
                print(f"  span_low_delta   : {distil_config.span_low_delta}", flush=True)
            else:
                print("[SpanOT-KD] disabled (vanilla MultiLevelOT)", flush=True)
        student_tokenizer, teacher_tokenizer, model = get_distillation_models(
            train_config, distil_config, fsdp_config, rank, vars(args)
        )
    else:
        tokenizer, model = get_model(train_config, fsdp_config, rank, vars(args))
    if rank == 0: print(model)
    if rank == 0: print("[checkpoint] models loaded OK", flush=True)

    # Load Data
    data_config.encoder_decoder = train_config.encoder_decoder
    if rank == 0: print("[checkpoint] starting data loading...", flush=True)
    if train_config.distillation:
        train_dataloader, teacher_train_dataloader, eval_dataloader, teacher_eval_dataloader = get_distillation_dataloader(data_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank)
    else:
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)
    if rank == 0: print(f"[checkpoint] data loaded OK — train batches: {len(train_dataloader)}", flush=True)

    if args.debug_tokenization and train_config.distillation:
        import sys
        from models.distillation_model import preprocess_distillation_batch
        from train.span_match import compute_single_sample_rates, SpanMatchEvaluator

        device = f"cuda:{local_rank}" if (train_config.enable_fsdp or distil_config.enable_fsdp) else "cuda:0"
        model.student.eval()
        print(f"[debug_tokenization] Checking {args.debug_max_batches} batch(es) then exiting.", flush=True)
        with torch.no_grad():
            for batch_idx, batch_pair in enumerate(zip(train_dataloader, teacher_train_dataloader)):
                if batch_idx >= args.debug_max_batches:
                    break
                batch = preprocess_distillation_batch(batch_pair)
                batch = {k: v.to(device) for k, v in batch.items()}
                student_out, teacher_out = model(**batch)
                s_starts, s_sizes = SpanMatchEvaluator._extract_answer_spans(batch["student_labels"])
                t_starts, t_sizes = SpanMatchEvaluator._extract_answer_spans(batch["teacher_labels"])
                print(f"[batch={batch_idx}] student_labels shape={list(batch['student_labels'].shape)}, "
                      f"s_sizes={s_sizes}, t_sizes={t_sizes}", flush=True)
                for i in range(batch["student_labels"].size(0)):
                    ss, se = s_starts[i], s_sizes[i]
                    ts, te = t_starts[i], t_sizes[i]
                    if se <= 0 or te <= 0:
                        print(f"  sample={i}: se={se}, te={te} — skipping (no answer tokens)", flush=True)
                        continue
                    s_pred_ids = torch.argmax(student_out.logits[i, ss:ss + se, :], dim=-1).cpu().tolist()
                    t_pred_ids = torch.argmax(teacher_out.logits[i, ts:ts + te, :], dim=-1).cpu().tolist()
                    raw_labels = batch["student_labels"][i]
                    answer_token_ids = raw_labels[raw_labels != -100].cpu().tolist()
                    answer_text = student_tokenizer.decode(answer_token_ids, skip_special_tokens=True)
                    print(f"\n[batch={batch_idx} sample={i}] answer_text={answer_text!r}", flush=True)
                    compute_single_sample_rates(
                        s_pred_ids, t_pred_ids, answer_text,
                        student_tokenizer, teacher_tokenizer,
                        debug=True,
                    )
        print("[debug_tokenization] Done. Exiting.", flush=True)
        sys.exit(0)

    # Get the optimizer and learning rate scheduler
    optimizer = get_optimizer(model, train_config, fsdp_config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config.lr, epochs=train_config.num_epochs, steps_per_epoch=len(train_dataloader),
                                                    pct_start=train_config.pct_start, div_factor=train_config.div_factor, final_div_factor=train_config.final_div_factor)

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
        local_rank if train_config.enable_fsdp or distil_config.enable_fsdp else None,
        rank,
        f,
    )
    if rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    main()