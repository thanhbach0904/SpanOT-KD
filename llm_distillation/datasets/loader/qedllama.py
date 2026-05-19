import os
import sys
from datasets import load_from_disk
from datasets import __file__ as datasets_file
print("datasets package path:", datasets_file)

sys.path.append(f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation")
from llm_distillation.prompt.prompt import create_chat_prompt
from llm_distillation.prompt.prompt import create_prompt
#import traceback


def tokenize(item, tokenizer, encoder_decoder=False):
    #traceback.print_stack()

    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa"

    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 5
        title = False
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 5
        title = item['title']
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 3
        title = False

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            title = title,
            context = item['paragraph_text'],
            question = item['question'],
            sys_user = True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['paragraph_text'],
            question = item['question'],
        )
    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    # 'example_id'
    
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {item['llama2_7b_chat_answers']}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{item['llama2_7b_chat_answers']}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {item['llama2_7b_chat_answers']}{tokenizer.eos_token}", add_special_tokens=False)
        #print(item['original_nq_answers'][:][0]['string'])
        #print(answer_tokens)
        #input()
        prompt_tokens = context_tokens+answer_tokens
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens


        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(item['llama2_7b_chat_answers'], add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

QEDLLAMA_DISK_PATH = f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation/datasets/hf/qedllama"


def _load_raw_train_dev(dataset_config):
    """Same seeded 90/10 carve-out as qed.py — must use the same seed/ratio
    on the same on-disk dataset to keep student and teacher dev sets aligned.
    """
    full_train = load_from_disk(QEDLLAMA_DISK_PATH)["train"]
    splits = full_train.train_test_split(
        test_size=dataset_config.dev_split_ratio,
        seed=dataset_config.dev_split_seed,
        shuffle=True,
    )
    return splits["train"], splits["test"]


def get_split(dataset_config, tokenizer, split):
    if split == "validation":
        raise RuntimeError(
            "[qedllama loader] split='validation' is the held-out test set "
            "and must not be loaded from the training pipeline. "
            "Use split='dev' for in-training evaluation."
        )
    if split not in ("train", "dev"):
        raise ValueError(f"[qedllama loader] unknown split '{split}'")

    print("这就是目录")
    print(dataset_config.generated_by.split('/')[-1])

    raw_train, raw_dev = _load_raw_train_dev(dataset_config)
    dataset = raw_train if split == "train" else raw_dev
    print(f"[qedllama loader] split='{split}' size={len(dataset)} "
          f"(ratio={dataset_config.dev_split_ratio}, seed={dataset_config.dev_split_seed})")

    if split == "train" and dataset_config.training_size < 1:
        dataset = dataset.select(range(int(len(dataset) * dataset_config.training_size)))

    dataset = dataset.map(
        lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder),
        remove_columns=list(dataset.features),
    )
    return dataset


def _build_generation_prompt(item, tokenizer):
    is_chat = ('chat' in tokenizer.name_or_path.lower()
               or "instruct" in tokenizer.name_or_path.lower())
    task = "qa"

    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 5
        title = False
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 5
        title = item['title_text']
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 3
        title = False
    else:
        shot = 0
        title = False

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            title=title,
            context=item['paragraph_text'],
            question=item['question'],
            sys_user=True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" in tokenizer.name_or_path else False,
            chat_template=tokenizer.apply_chat_template,
        )
    else:
        prompt = create_prompt(
            task, 0,
            context=item['paragraph_text'],
            question=item['question'],
        )
    return prompt


def get_dev_for_generation(dataset_config, tokenizer):
    """Dev split prepared for greedy generation; mirrors qed.py."""
    _, raw_dev = _load_raw_train_dev(dataset_config)

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _add_prompt(item):
        item['prompt'] = _build_generation_prompt(item, tokenizer)
        return item

    def _tok(items):
        return tokenizer(items['prompt'], padding='longest')

    dataset = raw_dev.map(_add_prompt)
    dataset = dataset.map(_tok, batched=True, batch_size=64)

    tokenizer.padding_side = prev_padding_side
    return dataset