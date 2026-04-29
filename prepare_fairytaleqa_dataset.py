#!/usr/bin/env python3
"""
Prepare FairytaleQA teacher-labeled dataset for distillation.

Input:  $HOME/FairytaleQAData  (cloned GitHub repo)
        Structure expected:
          story_meta.csv                              (filename, origin, split, ...)
          data-by-train-split/questions/              (one CSV per story)
          data-by-train-split/section-stories/        (one text file per story)

Output: $HOME/Multi-Level-OT/llm_distillation/datasets/hf/
            uld_loss_Llama-2-7b-chat-hf-FairytaleQA/fairytaleQA

Output DatasetDict splits: train, validation
Output columns per row:    context, question, answers_generated

Run:
    python prepare_fairytaleqa_dataset.py --batch_size 4
"""
import os
import sys
import csv
import torch
import argparse
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

HOME = os.getenv("HOME")
TEACHER_PATH = f"{HOME}/models/Llama-2-7b-chat-hf"
RAW_DATA_DIR = f"{HOME}/FairytaleQAData"
OUTPUT_PATH = (
    f"{HOME}/Multi-Level-OT/llm_distillation/datasets/hf"
    "/uld_loss_Llama-2-7b-chat-hf-FairytaleQA/fairytaleQA"
)

sys.path.append(f"{HOME}/Multi-Level-OT/llm_distillation")

SPLIT_DIR = os.path.join(RAW_DATA_DIR, "data-by-train-split")
QUESTIONS_DIR = os.path.join(SPLIT_DIR, "questions")
SECTION_DIR = os.path.join(SPLIT_DIR, "section-stories")
META_PATH = os.path.join(RAW_DATA_DIR, "story_meta.csv")


def _stories_for_split(split_label: str) -> list[str]:
    """Return story filenames that belong to `split_label` (train/val/test)."""
    stories = []
    with open(META_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"].strip() == split_label:
                stories.append(row["filename"].strip())
    if not stories:
        with open(META_PATH, encoding="utf-8") as f:
            available = sorted({r["split"].strip() for r in csv.DictReader(f)})
        raise ValueError(
            f"No stories found for split '{split_label}'. "
            f"Available splits in story_meta.csv: {available}"
        )
    return stories


def _find_story_file(story_name: str, base_dir: str) -> str | None:
    """Search recursively under base_dir for a file whose name starts with story_name."""
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        for fname in sorted(files):
            if fname.startswith(story_name):
                return os.path.join(root, fname)
    return None


def _load_sections(story_name: str) -> dict[str, str]:
    """
    Return a dict mapping section-id → section-text for one story.
    Searches recursively under SECTION_DIR.
    Handles CSV files and plain-text files (sections split by blank lines).
    """
    path = _find_story_file(story_name, SECTION_DIR)
    if path is None:
        raise FileNotFoundError(
            f"No section-stories file found for story '{story_name}' "
            f"under {SECTION_DIR}."
        )

    if path.endswith(".csv"):
        sections: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            id_col = next(
                (c for c in fields if "section" in c.lower() and "id" in c.lower()),
                next((c for c in fields if "id" in c.lower()), fields[0] if fields else None),
            )
            text_col = next(
                (c for c in fields if c.lower() in ("text", "content", "section", "passage")),
                next((c for c in fields if c != id_col), None),
            )
            if id_col is None or text_col is None:
                raise KeyError(
                    f"Cannot detect id/text columns in {path}. Fields: {fields}"
                )
            for row in reader:
                sections[row[id_col].strip()] = row[text_col].strip()
        return sections

    # Plain-text: split on blank lines
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return {str(i + 1): p for i, p in enumerate(paragraphs)}


def load_split(split_label: str) -> list[dict]:
    """
    Load all (context, question) pairs for the given split label
    (e.g. 'train', 'val', 'test').
    """
    stories = _stories_for_split(split_label)
    print(f"  [{split_label}] {len(stories)} stories found in story_meta.csv")

    items: list[dict] = []
    missing_q, missing_s = [], []

    # Peek at column names from the first available questions CSV
    q_col_detected: str | None = None
    section_col_detected: str | None = None

    for story_name in stories:
        # ── Locate questions CSV (search recursively) ───────────────────────
        q_path_found = _find_story_file(story_name, QUESTIONS_DIR)
        q_matches = [q_path_found] if q_path_found else []
        if not q_matches:
            missing_q.append(story_name)
            continue
        q_path = q_matches[0]

        # ── Load section texts for this story ───────────────────────────────
        try:
            sections = _load_sections(story_name)
        except FileNotFoundError:
            missing_s.append(story_name)
            sections = {}

        with open(q_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

            # Detect which column holds the section reference (id or text)
            if section_col_detected is None:
                for candidate in ("cor_section", "section_name", "section", "section_idx",
                                  "section_id", "content", "context", "passage"):
                    if candidate in fields:
                        section_col_detected = candidate
                        break
                if section_col_detected is None:
                    raise KeyError(
                        f"Cannot find section column in {q_path}. Fields: {fields}\n"
                        "Add the correct column name to the candidates list above."
                    )

            if q_col_detected is None:
                q_col_detected = "question" if "question" in fields else fields[-1]

            for row in reader:
                q = row.get(q_col_detected, "").strip()
                if not q:
                    continue

                raw_section = row.get(section_col_detected, "").strip()

                # If the column contains actual prose (>40 chars), use it directly.
                # Otherwise treat it as an id to look up in the section-stories file.
                if len(raw_section) > 40:
                    ctx = raw_section
                elif raw_section and sections:
                    ctx = sections.get(raw_section, "")
                    if not ctx:
                        # Try numeric lookup (some datasets use 1-based int index)
                        ctx = sections.get(str(raw_section), "")
                else:
                    ctx = ""

                if not ctx:
                    continue
                items.append({"context": ctx, "question": q})

    if missing_q:
        print(f"  Warning: no questions file for {len(missing_q)} stories: {missing_q[:5]}")
    if missing_s:
        print(f"  Warning: no section-stories file for {len(missing_s)} stories: {missing_s[:5]}")
    if not items:
        raise RuntimeError(
            f"No items loaded for split '{split_label}'. "
            f"Detected section_col='{section_col_detected}', q_col='{q_col_detected}'.\n"
            f"Run with --peek to print raw CSV rows for inspection."
        )

    print(f"  [{split_label}] {len(items)} (context, question) pairs loaded")
    return items


def build_prompt(item: dict, tokenizer) -> str:
    from llm_distillation.prompt.prompt import create_chat_prompt
    return create_chat_prompt(
        "qa_generative",
        2,  # shot=2 matches fairytaleQA.py for Llama-2-7b-chat-hf
        context=item["context"],
        question=item["question"],
        sys_user=False,
        chat_template=tokenizer.apply_chat_template,
    )


@torch.no_grad()
def run_inference(
    items: list[dict],
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    answers = []

    for i in tqdm(range(0, len(items), batch_size), desc="teacher inference"):
        batch = items[i : i + batch_size]
        prompts = [build_prompt(x, tokenizer) for x in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        # strip prompt tokens, keep only generated portion
        out = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        answers.extend(d.split("\n")[0].strip() for d in decoded)

    assert len(answers) == len(items), f"{len(answers)} != {len(items)}"
    return answers


def make_dataset(items: list[dict], answers: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "context": [x["context"] for x in items],
            "question": [x["question"] for x in items],
            "answers_generated": answers,
        }
    )


def _list_dir_tree(path: str, depth: int = 2, prefix: str = "  ") -> None:
    """Print directory tree up to `depth` levels."""
    try:
        entries = sorted(os.listdir(path))
    except NotADirectoryError:
        return
    for entry in entries[:20]:
        full = os.path.join(path, entry)
        print(f"{prefix}{entry}{'/' if os.path.isdir(full) else ''}")
        if depth > 1 and os.path.isdir(full):
            _list_dir_tree(full, depth - 1, prefix + "  ")


def _first_csv_in(directory: str):
    """Return the path to the first CSV found at any depth under directory."""
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".csv"):
                return os.path.join(root, fname)
    return None


def _first_file_in(directory: str):
    """Return the first regular file found at any depth under directory."""
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fname in sorted(files):
            return os.path.join(root, fname)
    return None


def _peek():
    """Print raw structure so you can identify the right column names."""
    print("=== story_meta.csv (first 5 rows) ===")
    with open(META_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            print(" ", line.rstrip())

    print(f"\n=== data-by-train-split/ tree (2 levels) ===")
    _list_dir_tree(SPLIT_DIR, depth=2)

    print("\n=== questions/ — first CSV found (columns + 3 rows) ===")
    q_csv = _first_csv_in(QUESTIONS_DIR)
    if q_csv:
        print("  file:", q_csv)
        with open(q_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print("  columns:", reader.fieldnames)
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(" ", dict(row))
    else:
        print("  No CSV files found anywhere under", QUESTIONS_DIR)

    print("\n=== section-stories/ — first file found (first 400 chars) ===")
    s_file = _first_file_in(SECTION_DIR)
    if s_file:
        print("  file:", s_file)
        with open(s_file, encoding="utf-8", errors="replace") as f:
            print(f.read(400))
    else:
        print("  No files found anywhere under", SECTION_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument(
        "--val_split",
        default="val",
        help="Split label used in story_meta.csv for validation (usually 'val')",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Print raw data structure and exit (no model loading, no inference)",
    )
    args = parser.parse_args()

    if args.peek:
        _peek()
        return

    # ── Load teacher ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "left"

    print(f"Loading model from {TEACHER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # ── Load raw FairytaleQA ─────────────────────────────────────────────────
    print("Loading raw FairytaleQA splits …")
    train_items = load_split("train")
    val_items = load_split(args.val_split)
    print(f"  train: {len(train_items)} rows, val: {len(val_items)} rows")
    print(f"  sample: {train_items[0]}")

    # ── Teacher inference ────────────────────────────────────────────────────
    print("\nRunning teacher inference on train split …")
    train_answers = run_inference(
        train_items, tokenizer, model, args.batch_size, args.max_new_tokens
    )

    print("\nRunning teacher inference on val split …")
    val_answers = run_inference(
        val_items, tokenizer, model, args.batch_size, args.max_new_tokens
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ds = DatasetDict(
        {
            "train": make_dataset(train_items, train_answers),
            "validation": make_dataset(val_items, val_answers),
        }
    )
    ds.save_to_disk(OUTPUT_PATH)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(ds)
    print("Train sample:", ds["train"][0])


if __name__ == "__main__":
    main()
