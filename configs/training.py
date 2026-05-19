from dataclasses import dataclass

@dataclass
class train_config:
    project_name: str=None
    model_name: str="meta-llama/Llama-2-7b-chat-hf"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="padding"
    context_length: int=None
    gradient_accumulation_steps: int=1
    num_epochs: int=1
    num_workers_dataloader: int=2
    lr: float=1e-6
    weight_decay: float=0.1
    pct_start=0.1
    div_factor=2
    final_div_factor=5
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    peft_method: str = "lora"
    use_peft: bool=False
    output_dir: str = ""
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    save_model: bool = True
    save_step: int = 1000  # deprecated: no longer drives in-epoch checkpointing
    save_optimizer: bool=False
    use_fast_kernels: bool = False
    distillation: bool = False
    save_all: bool = False
    training_size: int = 1
    encoder_decoder: bool = False
    f : int = 1

    # Dev-loss early stopping (epochs without dev-loss improvement; 0 disables).
    early_stopping_patience: int = 3
    # Generative dev F1: max new tokens for student.generate during dev eval.
    dev_eval_max_new_tokens: int = 64
    # Batch size for the dev generation dataloader.
    dev_gen_batch_size: int = 4