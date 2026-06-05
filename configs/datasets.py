from dataclasses import dataclass
 
@dataclass
class dataset:
    file: str = None
    training_size: float = 1
    encoder_decoder: bool = False

    # Distillation
    generated_by: str = None

    # Dev-set carve-out (seeded 90/10 split of the on-disk train).
    # The on-disk "validation" split is held out as the final test set and
    # is NEVER touched during training; loaders raise if asked for it.
    dev_split_ratio: float = 0.1
    dev_split_seed: int = 42