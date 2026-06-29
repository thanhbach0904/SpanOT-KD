"""
Visualize QED dataset samples exactly as seen by the training pipeline.

For each sample this script shows:
  1. Raw dataset fields: question, paragraph_text, gold answers
  2. The constructed prompt (identical to what qed.py passes to the tokenizer)
  3. The full token sequence split into: masked context vs. supervised answer
  4. A reminder of how input_ids / labels are constructed

Usage:
    python inspect_qed_samples.py [--n N] [--split train|validation] [--seed S]
"""
import io
import os
import sys
import argparse
import textwrap

# Force UTF-8 on Windows consoles that default to cp1252.
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datasets import load_from_disk

# Derive paths relative to this script so the file runs from any working dir.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../Multi-Level-OT

# prompt.py resolves few_shot modules via os.getenv('HOME')/Multi-Level-OT/...
# Override HOME to the project root's parent so that path resolves correctly
# on both Linux (server) and Windows (local).
os.environ['HOME'] = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, _SCRIPT_DIR)

from llm_distillation.prompt.prompt import create_prompt

QED_DISK_PATH = os.path.join(
    _SCRIPT_DIR, "llm_distillation", "datasets", "processed", "qed"
)

DIVIDER = "=" * 80
SUBDIV  = "-" * 80
WRAP_W  = 74


def _gold_answer(item):
    """Return the first gold answer string from original_nq_answers."""
    for entry in item.get("original_nq_answers", []):
        if isinstance(entry, dict) and "string" in entry:
            return entry["string"]
    return "<no answer>"


def _gold_answer_word_count(item):
    """Number of whitespace-separated words in the sample's gold answer."""
    answer = _gold_answer(item)
    if answer == "<no answer>":
        return 0
    return len(answer.split())


def _gold_answer_char_count(item):
    """Number of characters in the sample's gold answer."""
    answer = _gold_answer(item)
    if answer == "<no answer>":
        return 0
    return len(answer)


def compute_avg_gold_answer_length(dataset):
    """Average word and character count of gold answers across an entire dataset split."""
    word_counts = [_gold_answer_word_count(item) for item in dataset]
    char_counts = [_gold_answer_char_count(item) for item in dataset]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0.0
    return avg_words, avg_chars


def _build_prompt(item):
    """Construct the prompt exactly as qed.py does for a non-chat / 0-shot model."""
    return create_prompt(
        "qa", 0,
        context=item["paragraph_text"],
        question=item["question"],
    )


def _wrap(label, text, indent=4):
    pad = " " * indent
    lines = textwrap.fill(text, width=WRAP_W,
                          initial_indent=pad, subsequent_indent=pad)
    return f"  {label}:\n{lines}"


def display_sample(idx, item):
    answer = _gold_answer(item)
    prompt = _build_prompt(item)

    # Exact text boundaries that become input_ids vs labels in qed.tokenize()
    # (non-chat, decoder-only path):
    #   context_tokens = tokenizer.encode(f"{bos}{prompt}", add_special_tokens=False)
    #   answer_tokens  = tokenizer.encode(f" {answer}{eos}",  add_special_tokens=False)
    #   input_ids = context_tokens + answer_tokens
    #   labels    = [-100] * len(context_tokens) + answer_tokens
    masked_text     = f"<bos>{prompt}"
    supervised_text = f" {answer}<eos>"

    print(DIVIDER)
    print(f"  SAMPLE {idx + 1}")
    print(SUBDIV)

    print(_wrap("QUESTION",       item["question"]))
    print()
    print(_wrap("PARAGRAPH TEXT", item["paragraph_text"]))
    print()
    print(_wrap("GOLD ANSWER",    answer))
    print()

    print(SUBDIV)
    print("  TRAINING VIEW  (non-chat, 0-shot, decoder-only)")
    print(SUBDIV)
    print()
    print("  Prompt fed to tokenizer:")
    for line in prompt.splitlines():
        print(f"    {line}")
    print()

    print("  Token sequence with label-masking regions:")
    print()
    print("  ┌─ MASKED (labels = -100) " + "─" * 51 + "┐")
    for line in textwrap.wrap(masked_text, width=WRAP_W):
        print(f"  │  {line}")
    print("  └" + "─" * 77 + "┘")
    print()
    print("  ┌─ SUPERVISED (labels = answer token ids) " + "─" * 35 + "┐")
    for line in textwrap.wrap(supervised_text, width=WRAP_W):
        print(f"  │  {line}")
    print("  └" + "─" * 77 + "┘")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Show QED samples as seen by the training pipeline."
    )
    parser.add_argument("--n",     type=int, default=5,
                        help="Number of samples to display (default: 5)")
    parser.add_argument("--split", choices=["train", "validation"], default="train",
                        help="Dataset split to sample from (default: train)")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Shuffle seed (default: 42)")
    args = parser.parse_args()

    print(f"\nLoading QED dataset from: {QED_DISK_PATH}")
    dataset = load_from_disk(QED_DISK_PATH)[args.split]
    print(f"Split '{args.split}': {len(dataset):,} samples")
    print(f"Columns: {dataset.column_names}")

    avg_words, avg_chars = compute_avg_gold_answer_length(dataset)
    print(f"Average gold answer length: {avg_words:.2f} words / {avg_chars:.2f} characters "
          f"(over {len(dataset):,} samples in split '{args.split}')")

    print(f"\nDisplaying {args.n} sample(s)  (split='{args.split}', seed={args.seed})\n")

    samples = dataset.shuffle(seed=args.seed).select(range(min(args.n, len(dataset))))

    for i, item in enumerate(samples):
        display_sample(i, item)

    print(DIVIDER)
    print("\nLabel construction (non-chat, decoder-only) — from qed.py tokenize():")
    print("  context_tokens = tokenizer.encode('<bos>' + prompt)")
    print("  answer_tokens  = tokenizer.encode(' ' + answer + '<eos>')")
    print("  input_ids      = context_tokens + answer_tokens")
    print("  labels         = [-100] * len(context_tokens) + answer_tokens")
    print("  attention_mask = [1] * len(input_ids)")
    print()
    print("The model is supervised only on the answer span; the prompt is masked.\n")


if __name__ == "__main__":
    main()
