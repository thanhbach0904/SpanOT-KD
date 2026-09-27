import os
import sys
from datasets import load_from_disk

from datasets import __file__ as datasets_file
print("datasets package path:", datasets_file)


sys.path.append(f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation")
from llm_distillation.prompt.prompt import create_chat_prompt
from llm_distillation.prompt.prompt import create_prompt

def tokenize(item, tokenizer):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa_generative"

    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 2
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 4
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 2
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Qwen-7B-Chat":
        # Must equal the shot used by prepare_fairytaleqa_dataset.py when it
        # generated this teacher's labels (2, same as Llama-2).
        shot = 2
    else:
        # Unlisted chat teacher: `shot` was previously unbound here
        # (UnboundLocalError as soon as is_chat=True). Mirrors qed.py.
        shot = 0

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            context = item['context'],
            question = item['question'],
            sys_user = True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['context'],
            question = item['question'],
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    
    if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
        context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
        if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
            answer_tokens = tokenizer.encode(f" {item['answers_generated']}", add_special_tokens=False)
        else:
            answer_tokens = tokenizer.encode(f"{item['answers_generated']}", add_special_tokens=False)
    else:
        context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
        answer_tokens = tokenizer.encode(f" {item['answers_generated']}{tokenizer.eos_token}", add_special_tokens=False)

    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


FAIRYTALEQA_HF_DIR = f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation/datasets/hf"


def _fairytaleqa_disk_path(dataset_config):
    """Folder of teacher-generated labels for the teacher of this run.

    The labels (answers_generated) are the teacher's own outputs, so the folder
    is keyed by the teacher basename — same rule as prepare_fairytaleqa_dataset.py
    and run_experiments_fairytaleqa.sh. generated_by is set to the teacher's
    path in data_utils.get_distillation_dataloader; it is None only outside
    distillation, where we keep the legacy Llama-2 folder.
    """
    generated_by = getattr(dataset_config, "generated_by", None) or ""
    teacher = os.path.basename(generated_by.rstrip("/")) or "Llama-2-7b-chat-hf"
    return f"{FAIRYTALEQA_HF_DIR}/uld_loss_{teacher}-FairytaleQA/fairytaleQA"


def _load_raw_train_dev(dataset_config):
    """Load the on-disk train split and carve a seeded dev slice out of it.

    Mirrors qed.py's contract (see configs/datasets.py docstring): the
    on-disk 'train' is split 90/10 (seeded) into train/dev, and the on-disk
    'validation' split is the held-out final test set — never touched here.
    Returns (raw_train, raw_dev), both RAW (not tokenized).
    """
    disk_path = _fairytaleqa_disk_path(dataset_config)
    print(f"[fairytaleQA loader] labels from {disk_path}")
    full_train = load_from_disk(disk_path)["train"]
    splits = full_train.train_test_split(
        test_size=dataset_config.dev_split_ratio,
        seed=dataset_config.dev_split_seed,
        shuffle=True,
    )
    return splits["train"], splits["test"]


def get_split(dataset_config, tokenizer, split):
    """Return a tokenized split for the training loop.

    split:
      - "train" : 90% of on-disk train (after seeded carve-out)
      - "dev"   : 10% of on-disk train (seeded carve-out — used for dev loss)
      - "validation" : REJECTED. The on-disk validation set is the held-out
        test set and must never be loaded by the training pipeline.
    """
    if split == "validation":
        raise RuntimeError(
            "[fairytaleQA loader] split='validation' is the held-out test set "
            "and must not be loaded from the training pipeline. "
            "Use split='dev' for in-training evaluation."
        )
    if split not in ("train", "dev"):
        raise ValueError(f"[fairytaleQA loader] unknown split '{split}'")

    raw_train, raw_dev = _load_raw_train_dev(dataset_config)
    dataset = raw_train if split == "train" else raw_dev
    print(f"[fairytaleQA loader] split='{split}' size={len(dataset)} "
          f"(ratio={dataset_config.dev_split_ratio}, seed={dataset_config.dev_split_seed})")

    # training_size truncation only applies to train — dev is a fixed reference.
    if split == "train" and dataset_config.training_size < 1:
        dataset = dataset.select(range(int(len(dataset) * dataset_config.training_size)))

    dataset = dataset.map(lambda item: tokenize(item, tokenizer), remove_columns=list(dataset.features))
    return dataset


def get_dev_for_generation(dataset_config, tokenizer):
    """Return the dev split prepared for greedy generation.

    The returned dataset has left-padded input_ids/attention_mask and a
    QED-shaped 'original_nq_answers' column ([{'string': answer}]) so
    data_utils.get_dev_gen_dataloader (written against QED's schema) can
    pull the gold answer out the same way for every dataset.
    """
    _, raw_dev = _load_raw_train_dev(dataset_config)

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_chat = 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower()
    task = "qa_generative"
    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 2
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 4
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 2
    else:
        shot = 0

    def _add_prompt(item):
        if is_chat:
            item['prompt'] = create_chat_prompt(
                task, shot,
                context=item['context'],
                question=item['question'],
                sys_user=True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" in tokenizer.name_or_path else False,
                chat_template=tokenizer.apply_chat_template,
            )
        else:
            item['prompt'] = create_prompt(
                task, 0,
                context=item['context'],
                question=item['question'],
            )
        item['original_nq_answers'] = [{'string': item['answers_generated']}]
        return item

    def _tok(items):
        return tokenizer(items['prompt'], padding='longest')

    dataset = raw_dev.map(_add_prompt)
    dataset = dataset.map(_tok, batched=True, batch_size=64)

    tokenizer.padding_side = prev_padding_side
    return dataset