import os
import torch
import importlib

from pathlib import Path
from data.concatenator import ConcatDataset
from configs.configs_utils import get_dataloader_kwargs

sort_index = []
sort_index_val = []

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)
    return module


def _load_dataset_module(dataset_config):
    if not dataset_config.file:
        raise ValueError(
            f"Dataset not specified. Please select a dataset path with the parameter '--dataset.file'.")

    if dataset_config.file.endswith('.py'):
        module_path = Path(dataset_config.file)
    else:
        module_path = Path(dataset_config.file + "/load.py")

    if not os.path.isfile(module_path):
        raise ValueError(
            f"The load.py file in the dataset folder or the path to a python loading file doesn't exist. {module_path}")
    return load_module_from_py_file(module_path.as_posix())


def get_dataset(dataset_config, tokenizer, split: str) -> torch.utils.data.Dataset:
    module = _load_dataset_module(dataset_config)
    try:
        return getattr(module, "get_split")(dataset_config, tokenizer, split)
    except AttributeError:
        raise ValueError(f"Method 'get_split' not found in {dataset_config.file}.")


def get_dev_generation_dataset(dataset_config, tokenizer):
    """Return the dev set prepared for greedy generation (raw prompts,
    left-padded input_ids, gold answers preserved). Requires the dataset
    loader to expose ``get_dev_for_generation``.
    """
    module = _load_dataset_module(dataset_config)
    if not hasattr(module, "get_dev_for_generation"):
        raise AttributeError(
            f"Dataset loader {dataset_config.file} does not implement "
            "'get_dev_for_generation' — required for generative dev F1.")
    return module.get_dev_for_generation(dataset_config, tokenizer)


def get_dataloader(dataset_config, train_config, tokenizer, rank, distil_config=None):
    global sort_index
    global sort_index_val
    
    dataset_train = get_dataset(
        dataset_config,
        tokenizer,
        split="train",
    )
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(
            dataset_train, chunk_size=train_config.context_length)
    
    if train_config.context_length and not sort_index:
        sort_index = [idx for idx, ex in enumerate(dataset_train) if len(ex['input_ids']) <= train_config.context_length]
    if train_config.context_length and sort_index:
        dataset_train = dataset_train.select(sort_index)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train", distil_config)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
        **train_dl_kwargs,
    )
    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if (train_config.run_validation):
        # Note: 'dev' is a seeded 10% carve-out of the on-disk train set; the
        # on-disk 'validation' split is the held-out final test set and the
        # loader will RAISE if asked for it. See loader docstrings.
        dataset_val = get_dataset(
            dataset_config,
            tokenizer,
            split="dev",
        )

        if train_config.context_length and not sort_index_val:
            sort_index_val = [idx for idx, ex in enumerate(dataset_val) if len(ex['input_ids']) <= train_config.context_length]
        if sort_index_val:
            dataset_val = dataset_val.select(sort_index_val)

        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(
                dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val", distil_config)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            **val_dl_kwargs,
        )
        if rank == 0:
            print(f"--> Dev Set Length (seeded carve-out from train) = {len(dataset_val)}")
            # Hard guard: dev must never equal the on-disk held-out test size.
            assert len(dataset_val) != 1355, (
                f"[guard] Dev set has 1355 rows — this is the QED held-out test set. "
                "The training pipeline must not load it. Check qed loader split routing.")
        return train_dataloader, eval_dataloader
    else:
        return train_dataloader, None


def get_dev_gen_dataloader(dataset_config, train_config, tokenizer, rank):
    """Build a dataloader for greedy generation on the dev split.

    The dataset has left-padded input_ids/attention_mask; the gold answers
    column ``original_nq_answers`` is preserved separately on the dataset
    (the caller pulls it out before set_format strips non-tensor columns).
    """
    dataset = get_dev_generation_dataset(dataset_config, tokenizer)

    # Pull the gold answers out before we lock the format to tensors.
    answers = []
    for sublist in dataset['original_nq_answers']:
        if sublist and isinstance(sublist[0], dict) and 'string' in sublist[0]:
            answers.append(sublist[0]['string'])
        else:
            answers.append("")

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config.dev_gen_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
    )
    if rank == 0:
        print(f"--> Dev Generation Set Length = {len(dataset)} "
              f"(batch_size={train_config.dev_gen_batch_size})")
    return dataloader, answers


def get_distillation_dataloader(dataset_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank):
    dataset_config.generated_by = teacher_tokenizer.name_or_path

    student_train_dataloader, student_eval_dataloader = get_dataloader(dataset_config, train_config, student_tokenizer, rank, distil_config)
    dataset_config.encoder_decoder = True if distil_config.encoder_decoder else False
    teacher_train_dataloader, teacher_eval_dataloader = get_dataloader(dataset_config, train_config, teacher_tokenizer, rank, distil_config)
    dataset_config.encoder_decoder = train_config.encoder_decoder

    # Dev generation dataloader is built against the STUDENT tokenizer — the
    # student is what's being selected by F1.
    dev_gen_dataloader, dev_gen_answers = get_dev_gen_dataloader(
        dataset_config, train_config, student_tokenizer, rank)

    return (student_train_dataloader, teacher_train_dataloader,
            student_eval_dataloader, teacher_eval_dataloader,
            dev_gen_dataloader, dev_gen_answers)