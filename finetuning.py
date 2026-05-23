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
                        help="Fraction of spans (sorted by entropy gap) deemed high-priority. r=1.0 collapses to MultiLevelOT")
    parser.add_argument("--distillation_config_span_low_delta", type=float, default=0.1,
                        help="Down-weight applied to non-top-r spans. Methodology recommends (0, 0.1]. delta=1.0 collapses to MultiLevelOT")
    return parser.parse_args()

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

    if rank == 0:
        print(f"[seed] {train_config.seed}", flush=True)

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
    dev_gen_dataloader = None
    dev_gen_answers = None
    if train_config.distillation:
        (train_dataloader, teacher_train_dataloader,
         eval_dataloader, teacher_eval_dataloader,
         dev_gen_dataloader, dev_gen_answers) = get_distillation_dataloader(
            data_config, train_config, distil_config,
            student_tokenizer, teacher_tokenizer, rank)
    else:
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)
    if rank == 0: print(f"[checkpoint] data loaded OK — train batches: {len(train_dataloader)}", flush=True)

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
        dev_gen_dataloader=dev_gen_dataloader,
        dev_gen_answers=dev_gen_answers,
        student_tokenizer=(student_tokenizer if train_config.distillation else tokenizer),
    )
    if rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    main()