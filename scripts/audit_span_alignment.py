#!/usr/bin/env python3
"""WP6: measure span-alignment health for a (student, teacher) pair.

Tokenizer-only (no model forward): mirrors exactly what the training loss sees
for each sample — answer sizes from the label rows (with skip_student_eos=True
as in train_utils), the student-label decode, `_safe_offset_map` on both sides,
clipping to min(size, len(offsets)), then `find_parent_token`.

Reports, over N training samples:
  * fraction of student / teacher answer tokens with span id -1
  * parent spans per sample (mean / median)
  * fraction of samples hitting each fallback of compute_position_weights_one_sample
    that is decidable without logits: no offsets, no parent spans, no active spans.
    "no positive-gap spans" needs model entropies -> reported as NOT COMPUTED.
  * fraction of spans inactive on one side (empty s_idx or t_idx)
  * end-offset mismatch (student_offsets[-1][-1] != teacher_offsets[-1][-1]) and
    what the last parent span covers in those cases (find_parent_token builds
    parent boundaries from student ends only, with n = max(a, b))
  * samples where teacher_size > student_size (rows beyond min(s, t) are outside
    the L1 support but inside KL/Sinkhorn's)

Usage (on vast.ai, from the repo root):
  python scripts/audit_span_alignment.py \
      --student $HOME/SpanOT-KD/EleutherAI/opt-350m \
      --teacher /workspace/models/Llama-2-7b-chat-hf \
      --dataset_file llm_distillation/datasets/loader/qed.py \
      --n 500 --out results/span_audit_qed_opt_llama
Writes <out>.json and <out>.md.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.span_match import find_parent_token  # noqa: E402
from train.span_ot import _safe_offset_map  # noqa: E402


def answer_size(labels, ignore_index=-100):
    """Same count as DistillationLoss.__get_start_and_size_answers."""
    return sum(1 for x in labels if x != ignore_index)


def audit_sample(s_labels, t_labels, s_tok_span, t_tok_span, ignore_index=-100):
    s_size = answer_size(s_labels) - 1  # skip_student_eos=True in train_utils
    t_size = answer_size(t_labels)      # skip_teacher_eos=False
    rec = {"s_size": s_size, "t_size": t_size, "fallback": None}
    ids = [x for x in s_labels if x != ignore_index]
    text = s_tok_span.decode(ids, skip_special_tokens=True) if ids else ""
    s_off = _safe_offset_map(text, s_tok_span)
    t_off = _safe_offset_map(text, t_tok_span)
    if not s_off or not t_off:
        rec["fallback"] = "no_offsets"
        return rec
    s_len, t_len = min(s_size, len(s_off)), min(t_size, len(t_off))
    if s_len <= 0 or t_len <= 0:
        rec["fallback"] = "no_offsets"
        return rec
    s_off, t_off = s_off[:s_len], t_off[:t_len]
    s_dict, t_dict = find_parent_token(s_off, t_off)
    if not s_dict or not t_dict:
        rec["fallback"] = "no_parent_spans"
        return rec

    s_cov, t_cov = set(), set()
    n_active = n_inactive = 0
    for key, t_idx in t_dict.items():
        s_idx = [i for i in s_dict.get(key, []) if i < s_len]
        t_idx = [i for i in t_idx if i < t_len]
        s_cov.update(s_idx)
        t_cov.update(t_idx)
        if s_idx and t_idx:
            n_active += 1
        else:
            n_inactive += 1
    rec.update(
        n_spans=len(t_dict), n_active=n_active, n_inactive=n_inactive,
        s_unaligned=s_len - len(s_cov), t_unaligned=t_len - len(t_cov), s_len=s_len, t_len=t_len,
        end_mismatch=s_off[-1][-1] != t_off[-1][-1],
    )
    if rec["end_mismatch"]:
        last = list(t_dict.keys())[-1]
        rec["mismatch_detail"] = {
            "text": text, "student_end": s_off[-1][-1], "teacher_end": t_off[-1][-1],
            "last_span": list(last), "last_span_text": text[last[0]:last[1]],
        }
    if n_active == 0:
        rec["fallback"] = "no_active_spans"
    return rec


def summarise(recs):
    N = len(recs)
    fb = {k: sum(r["fallback"] == k for r in recs) / max(N, 1)
          for k in ("no_offsets", "no_parent_spans", "no_active_spans")}
    ok = [r for r in recs if "n_spans" in r]
    tot_s = sum(r["s_len"] for r in ok) or 1
    tot_t = sum(r["t_len"] for r in ok) or 1
    spans = [r["n_spans"] for r in ok]
    tot_spans = sum(spans) or 1
    mism = [r for r in ok if r["end_mismatch"]]
    return {
        "n_samples": N,
        "fallback_frac": {**fb, "no_positive_gap_spans": "NOT COMPUTED (needs model logits)"},
        "student_token_unaligned_frac": sum(r["s_unaligned"] for r in ok) / tot_s,
        "teacher_token_unaligned_frac": sum(r["t_unaligned"] for r in ok) / tot_t,
        "spans_per_sample_mean": statistics.mean(spans) if spans else None,
        "spans_per_sample_median": statistics.median(spans) if spans else None,
        "inactive_span_frac": sum(r["n_inactive"] for r in ok) / tot_spans,
        "end_offset_mismatch_frac": len(mism) / max(N, 1),
        "teacher_longer_than_student_frac": sum(r["t_size"] > r["s_size"] for r in recs) / max(N, 1),
        "end_mismatch_examples": [r["mismatch_detail"] for r in mism[:20]],
    }


def to_markdown(summary, args):
    lines = [f"# Span alignment audit", "",
             f"student `{args.student}` / teacher `{args.teacher}` / dataset `{args.dataset_file}` "
             f"/ split `{args.split}` / N = {summary['n_samples']}", "",
             "| quantity | value |", "|---|---|"]
    for k, v in summary.items():
        if k == "end_mismatch_examples":
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"| {k}.{kk} | {vv if isinstance(vv, str) else f'{vv:.4f}'} |")
        else:
            lines.append(f"| {k} | {v if v is None or isinstance(v, int) else f'{v:.4f}'} |")
    if summary["end_mismatch_examples"]:
        lines += ["", "## End-offset mismatch examples (first 20)", ""]
        for ex in summary["end_mismatch_examples"]:
            lines.append(f"- student_end={ex['student_end']} teacher_end={ex['teacher_end']} "
                         f"last_span={ex['last_span']} `{ex['last_span_text']!r}` in `{ex['text']!r}`")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--dataset_file", required=True)
    p.add_argument("--split", default="train", choices=["train", "dev"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42, help="dev carve-out seed (same as training)")
    p.add_argument("--out", required=True, help="output path prefix (.json / .md appended)")
    args = p.parse_args()

    # Heavy imports here so audit_sample() stays importable in unit tests.
    from transformers import AutoTokenizer
    from configs.datasets import dataset as DatasetConfig
    from data.data_utils import get_dataset
    from models.models_utils import load_tokenizer

    cfg = DatasetConfig()
    cfg.file = args.dataset_file
    cfg.generated_by = args.teacher
    cfg.dev_split_seed = args.seed
    # Label tokenisation: same loader as training (get_distillation_models).
    s_tok, t_tok = load_tokenizer(args.student, False), load_tokenizer(args.teacher, False)
    s_ds = get_dataset(cfg, s_tok, args.split)
    t_ds = get_dataset(cfg, t_tok, args.split)
    # Offsets: same tokenizers as DistillationLoss (use_fast=True).
    s_span = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True, use_fast=True)
    t_span = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True, use_fast=True)

    n = min(args.n, len(s_ds))
    recs = [audit_sample(s_ds[i]["labels"], t_ds[i]["labels"], s_span, t_span) for i in range(n)]
    summary = summarise(recs)
    summary["config"] = vars(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.out + ".md", "w") as fh:
        fh.write(to_markdown({k: v for k, v in summary.items() if k != "config"}, args))
    print(to_markdown({k: v for k, v in summary.items() if k != "config"}, args))


if __name__ == "__main__":
    main()
