from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class distillation_config:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    quantization: bool = False
    use_fast_kernels: bool = False
    use_peft: bool = False
    freeze_layers: bool = False
    num_freeze_layers: int = 0
    cross_entropy_factor: float = 1
    distil_factor: float = 1.5
    student_temperature: float = 1
    teacher_temperature: float = 1
    encoder_decoder: bool = False
    
    # FSDP Config
    mixed_precision: bool = False
    use_fp16: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"

    # SpanOT-KD config (extension of MultiLevelOT, see train/span_ot.py).
    # When `span_kd_enabled=False` the loss is the original MultiLevelOT
    # objective from Cui et al. 2025 — byte-for-byte identical numerics.
    # When True, the three distillation components (HAD, SL, SD) get
    # multiplied by per-(teacher-)position weights derived from a
    # character-aligned span partition + per-span entropy gap.
    span_kd_enabled: bool = False
    # Aggregation strategy for per-token entropy gaps within a span:
    #   "mean" — Eq. 10 (length-normalised, span-size invariant)
    #   "sum"  — Eq. 11 (accumulated, biases toward long spans)
    span_aggregation: str = "mean"
    # Fraction of spans (sorted by gap, descending) that receive weight 1.0.
    # The rest receive `span_low_delta`. r=1.0 collapses to MultiLevelOT.
    span_top_r: float = 0.5
    # Down-weight for low-priority spans. Methodology recommends (0, 0.1].
    # delta=1.0 also collapses to MultiLevelOT.
    span_low_delta: float = 0.1