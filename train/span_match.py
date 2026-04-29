"""Span-level and token-level match-rate diagnostics for student–teacher KD.

A *parent span* is the minimal character range whose boundaries are respected
by both the student and teacher tokenisations of the same text.  For every
parent span we ask: do student and teacher produce the same decoded text for
the tokens that land in that span?  Aggregating over all spans and samples
gives the *span-match rate* — a direct measure of how well the student mimics
the teacher AT THE SPAN LEVEL rather than the raw-token level.

Public API
----------
find_parent_token()            – core alignment primitive (paper code)
compute_span_assignments()     – per-token span-ID vectors
compute_single_sample_rates()  – span + token match rates for one sample
SpanMatchEvaluator             – runs evaluation passes and saves results

Usage (inside train_utils.py)
------------------------------
    evaluator = SpanMatchEvaluator(
        student_tokenizer_path, teacher_tokenizer_path, output_dir
    )
    # before training:
    evaluator.run_evaluation(model, train_dl, teacher_dl, "pre_training", 0)
    # mid / end of each epoch:
    evaluator.run_evaluation(model, train_dl, teacher_dl, f"epoch_{e}_mid", step)
    evaluator.run_evaluation(model, train_dl, teacher_dl, f"epoch_{e}_end", step)
    # after all epochs:
    evaluator.plot_history()

Enable via environment variable:
    SPAN_MATCH_EVAL=1          activate the evaluator (default: off)
    SPAN_MATCH_MAX_BATCHES=N   cap evaluation pass at N batches (default: 500)
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Core span-alignment primitives
# ---------------------------------------------------------------------------

def find_parent_token(
    student_offset_map: List[Tuple[int, int]],
    teacher_offset_map: List[Tuple[int, int]],
) -> Tuple[Dict, Dict]:
    """Identify *parent spans* and map every token to its containing span.

    A parent span is a minimal character range whose boundaries are respected
    by both tokenisations.  Returns two dicts keyed by ``(char_start, char_end)``
    tuples; values are lists of token indices within the respective tokenisation.
    The dicts share the same set of keys so you can zip over them.
    """
    if not student_offset_map or not teacher_offset_map:
        return {}, {}

    a = student_offset_map[-1][-1]
    b = teacher_offset_map[-1][-1]
    n = max(a, b)
    if n == 0:
        return {}, {}

    cnt = np.zeros(n + 1, dtype=np.int32)
    for s, e in student_offset_map:
        cnt[s] += 1
    for s, e in teacher_offset_map:
        cnt[s] += 1
    cnt[n] = 2

    parent_index = [0]
    for _, e in student_offset_map:
        if cnt[e] == 2 and (not parent_index or e != parent_index[-1]):
            parent_index.append(int(e))

    parents = [
        (parent_index[i], parent_index[i + 1])
        for i in range(len(parent_index) - 1)
    ]

    def _build_dict(
        offset_map: List[Tuple[int, int]],
        parents: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], List[int]]:
        d: Dict[Tuple[int, int], List[int]] = {}
        idx = 0
        for p in parents:
            d[p] = []
            while idx < len(offset_map):
                if offset_map[idx][1] <= p[1]:
                    d[p].append(idx)
                if offset_map[idx][1] == p[1]:
                    idx += 1
                    break
                idx += 1
        return d

    return _build_dict(student_offset_map, parents), _build_dict(teacher_offset_map, parents)


def compute_span_assignments(
    student_offsets: List[Tuple[int, int]],
    teacher_offsets: List[Tuple[int, int]],
    student_len: int,
    teacher_len: int,
) -> Tuple[List[int], List[int]]:
    """Return per-token span-id vectors for student and teacher.

    Tokens not covered by any parent span get id ``-1``.
    """
    s_dict, t_dict = find_parent_token(student_offsets, teacher_offsets)

    s_spans = [-1] * student_len
    t_spans = [-1] * teacher_len

    for span_idx, (_, indices) in enumerate(s_dict.items()):
        for i in indices:
            if i < student_len:
                s_spans[i] = span_idx
    for span_idx, (_, indices) in enumerate(t_dict.items()):
        for i in indices:
            if i < teacher_len:
                t_spans[i] = span_idx

    return s_spans, t_spans


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase — used for normalized match metric."""
    return re.sub(r'\s+', ' ', s).strip().lower()


# ---------------------------------------------------------------------------
# Per-sample match-rate computation
# ---------------------------------------------------------------------------

def compute_single_sample_rates(
    student_pred_ids: List[int],
    teacher_pred_ids: List[int],
    answer_text: str,
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, float]:
    """Compute span-level and token-level match rates for a single sample.

    Parameters
    ----------
    student_pred_ids:
        Predicted token IDs from the student for the answer positions,
        i.e. ``argmax(student_logits[answer_start : answer_start + answer_size])``.
    teacher_pred_ids:
        Same for the teacher.
    answer_text:
        Ground-truth answer decoded as a string.  Used ONLY to obtain
        character-level offset maps for span alignment.

    Returns
    -------
    dict with keys:
      ``span_match_rate``   – fraction of parent spans where decoded student
                              text == decoded teacher text.  NaN if no spans.
      ``token_match_rate``  – fraction of aligned positions (min length) where
                              individually decoded tokens agree.
      ``num_spans``         – total parent spans found.
      ``avg_tokens_per_span_student`` / ``…_teacher``.
    """
    nan_result = {
        "span_match_rate": float("nan"),
        "span_normalized_match_rate": float("nan"),
        "span_boundary_match_rate": float("nan"),
        "span_content_match_rate": float("nan"),
        "span_content_normalized_match_rate": float("nan"),
        "token_match_rate": float("nan"),
        "num_spans": 0,
        "active_spans": 0,
        "avg_tokens_per_span_student": 0.0,
        "avg_tokens_per_span_teacher": 0.0,
        "span_details": [],
    }

    if not student_pred_ids or not teacher_pred_ids or not answer_text.strip():
        return nan_result

    s_encoding = student_tokenizer(
        answer_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    t_encoding = teacher_tokenizer(
        answer_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    s_offsets = [(int(s), int(e)) for s, e in s_encoding["offset_mapping"] if s != e]
    t_offsets = [(int(s), int(e)) for s, e in t_encoding["offset_mapping"] if s != e]

    if not s_offsets or not t_offsets:
        return nan_result

    # Clip to the shorter of (offset map length, prediction length) to stay in-bounds.
    s_len = min(len(student_pred_ids), len(s_offsets))
    t_len = min(len(teacher_pred_ids), len(t_offsets))

    s_dict, t_dict = find_parent_token(s_offsets[:s_len], t_offsets[:t_len])
    if not s_dict:
        return nan_result

    # ---------- span-level match ----------
    num_spans = len(s_dict)
    active_spans = 0      # spans with tokens present on both student and teacher sides
    span_matches = 0
    span_norm_matches = 0
    span_details: List[Dict] = []
    for (span_key, s_indices), (_, t_indices) in zip(s_dict.items(), t_dict.items()):
        s_ids = [student_pred_ids[j] for j in s_indices if j < s_len]
        t_ids = [teacher_pred_ids[j] for j in t_indices if j < t_len]
        if not s_ids or not t_ids:
            span_details.append({
                "char_span": list(span_key),
                "student_text": None,
                "teacher_text": None,
                "matched_strict": False,
                "matched_normalized": False,
            })
            continue
        active_spans += 1
        s_text = student_tokenizer.decode(s_ids, skip_special_tokens=True)
        t_text = teacher_tokenizer.decode(t_ids, skip_special_tokens=True)
        matched_strict = s_text.strip() == t_text.strip()
        matched_normalized = _normalize(s_text) == _normalize(t_text)
        if matched_strict:
            span_matches += 1
        if matched_normalized:
            span_norm_matches += 1
        span_details.append({
            "char_span": list(span_key),
            "student_text": s_text,
            "teacher_text": t_text,
            "matched_strict": matched_strict,
            "matched_normalized": matched_normalized,
        })

    # span_boundary_match_rate: what fraction of parent spans have tokens on BOTH sides
    # span_content_match_rate:  of those active spans, what fraction have matching decoded text
    span_boundary_match_rate = active_spans / num_spans if num_spans > 0 else float("nan")
    span_content_match_rate = span_matches / active_spans if active_spans > 0 else float("nan")
    span_content_normalized_match_rate = span_norm_matches / active_spans if active_spans > 0 else float("nan")
    # Legacy metrics (denominator = total spans) kept for backwards compatibility
    span_match_rate = span_matches / num_spans if num_spans > 0 else float("nan")
    span_normalized_match_rate = span_norm_matches / num_spans if num_spans > 0 else float("nan")

    # ---------- token-level match (position-aligned, capped at min length) ----------
    min_len = min(s_len, t_len)
    token_matches = 0
    for j in range(min_len):
        s_tok = student_tokenizer.decode([student_pred_ids[j]], skip_special_tokens=True)
        t_tok = teacher_tokenizer.decode([teacher_pred_ids[j]], skip_special_tokens=True)
        if s_tok.strip() == t_tok.strip():
            token_matches += 1
    token_match_rate = token_matches / min_len if min_len > 0 else float("nan")

    return {
        "span_match_rate": span_match_rate,
        "span_normalized_match_rate": span_normalized_match_rate,
        "span_boundary_match_rate": span_boundary_match_rate,
        "span_content_match_rate": span_content_match_rate,
        "span_content_normalized_match_rate": span_content_normalized_match_rate,
        "token_match_rate": token_match_rate,
        "num_spans": num_spans,
        "active_spans": active_spans,
        "avg_tokens_per_span_student": s_len / num_spans if num_spans > 0 else 0.0,
        "avg_tokens_per_span_teacher": t_len / num_spans if num_spans > 0 else 0.0,
        "span_details": span_details,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _clean_for_json(obj):
    """Recursively replace float NaN/Inf with None for valid JSON output."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


class SpanMatchEvaluator:
    """Evaluate span/token match rates between student and teacher predictions.

    Designed to be called at three training phases:

    1. **Before training** – measures the vanilla (untrained) student.
    2. **During training** – at mid-epoch and end-of-epoch checkpoints.
    3. **After training** – measures the final converged model.

    Results are appended to ``<output_dir>/span_match_history.json`` after
    each call and logged to W&B under the ``span_match/`` prefix.  A summary
    plot is written by :meth:`plot_history`.

    Parameters
    ----------
    student_tokenizer_path / teacher_tokenizer_path:
        Paths or Hub IDs of the respective tokenisers.
    output_dir:
        Directory for ``span_match_history.json`` and the plot.
    max_eval_batches:
        Upper bound on batches consumed per evaluation pass.  Default 500;
        set lower via SPAN_MATCH_MAX_BATCHES to trade coverage for speed.
    rank:
        Distributed rank.  Only rank 0 logs to W&B and writes files.
    """

    def __init__(
        self,
        student_tokenizer_path: str,
        teacher_tokenizer_path: str,
        output_dir: str,
        max_eval_batches: int = 500,
        rank: int = 0,
    ) -> None:
        self.output_dir = output_dir or "."
        self.max_eval_batches = max_eval_batches
        self.rank = rank
        self._history: List[Dict] = []

        if rank == 0:
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                student_tokenizer_path, trust_remote_code=True
            )
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_tokenizer_path, trust_remote_code=True
            )
            print(
                f"[SpanMatchEvaluator] Initialised. "
                f"max_eval_batches={max_eval_batches}, output_dir={output_dir}"
            )
        else:
            self.student_tokenizer = None
            self.teacher_tokenizer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer_spans(
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> Tuple[List[int], List[int]]:
        """Return (start_positions, sizes) for the answer part of each row.

        Mirrors the logic in ``DistillationLoss.__get_start_and_size_answers``.
        """
        starts, sizes = [], []
        for row in labels:
            is_pad = row.eq(ignore_index)
            sizes.append(int((~is_pad).sum().item()))
            indices = is_pad.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                starts.append(0)
            else:
                diff = indices[1:] - indices[:-1]
                brk = (diff != 1).nonzero()
                length = (brk[0].item() + 1) if len(brk) > 0 else len(indices)
                starts.append(length - 1)
        return starts, sizes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_evaluation(
        self,
        model,
        train_dataloader,
        teacher_train_dataloader,
        phase_label: str,
        global_step: int,
        local_rank: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run a capped evaluation pass and return aggregate match-rate stats.

        Parameters
        ----------
        model:
            The ``DistillationModel`` instance.
        train_dataloader / teacher_train_dataloader:
            Training dataloaders; only the first ``max_eval_batches`` batches
            are consumed.  Creating a new ``for`` loop here does not affect
            the outer training loop's iterator.
        phase_label:
            Human-readable label, e.g. ``"pre_training"``, ``"epoch_2_end"``.
        global_step:
            Current training step for W&B x-axis.
        local_rank:
            Device to place tensors on.  ``None`` → ``cuda:0``.
        """
        if self.rank != 0:
            return {}

        from models.distillation_model import preprocess_distillation_batch  # avoid circular import

        device = f"cuda:{local_rank}" if local_rank is not None else "cuda:0"

        was_training = model.student.training
        model.student.eval()

        span_rates: List[float] = []
        span_norm_rates: List[float] = []
        span_boundary_rates: List[float] = []
        span_content_rates: List[float] = []
        span_content_norm_rates: List[float] = []
        token_rates: List[float] = []
        num_spans_per_sample: List[int] = []
        sample_match_details: List[Dict] = []
        unmatched_pairs: List[Dict] = []  # capped at 50 for manual inspection

        for batch_idx, batch_pair in enumerate(zip(train_dataloader, teacher_train_dataloader)):
            if batch_idx >= self.max_eval_batches:
                break

            batch = preprocess_distillation_batch(batch_pair)
            batch = {k: v.to(device) for k, v in batch.items()}

            student_out, teacher_out = model(**batch)

            s_starts, s_sizes = self._extract_answer_spans(batch["student_labels"])
            t_starts, t_sizes = self._extract_answer_spans(batch["teacher_labels"])

            for i in range(batch["student_labels"].size(0)):
                ss, se = s_starts[i], s_sizes[i]
                ts, te = t_starts[i], t_sizes[i]
                if se <= 0 or te <= 0:
                    continue

                s_pred_ids = torch.argmax(
                    student_out.logits[i, ss: ss + se, :], dim=-1
                ).cpu().tolist()
                t_pred_ids = torch.argmax(
                    teacher_out.logits[i, ts: ts + te, :], dim=-1
                ).cpu().tolist()

                # Strip EOS from predictions: answer_text is decoded with
                # skip_special_tokens=True so re-tokenization yields one fewer
                # token, causing the offset-map length assertion to fail.
                if s_pred_ids and s_pred_ids[-1] == self.student_tokenizer.eos_token_id:
                    s_pred_ids = s_pred_ids[:-1]
                if t_pred_ids and t_pred_ids[-1] == self.teacher_tokenizer.eos_token_id:
                    t_pred_ids = t_pred_ids[:-1]

                # Decode ground-truth answer text from student labels (used only
                # to produce offset maps — predictions are free to differ).
                raw_labels = batch["student_labels"][i]
                answer_token_ids = raw_labels[raw_labels != -100].cpu().tolist()
                answer_text = self.student_tokenizer.decode(
                    answer_token_ids, skip_special_tokens=True
                )

                rates = compute_single_sample_rates(
                    s_pred_ids,
                    t_pred_ids,
                    answer_text,
                    self.student_tokenizer,
                    self.teacher_tokenizer,
                )
                if not math.isnan(rates["span_match_rate"]):
                    span_rates.append(rates["span_match_rate"])
                if not math.isnan(rates["span_normalized_match_rate"]):
                    span_norm_rates.append(rates["span_normalized_match_rate"])
                if not math.isnan(rates["span_boundary_match_rate"]):
                    span_boundary_rates.append(rates["span_boundary_match_rate"])
                if not math.isnan(rates["span_content_match_rate"]):
                    span_content_rates.append(rates["span_content_match_rate"])
                if not math.isnan(rates["span_content_normalized_match_rate"]):
                    span_content_norm_rates.append(rates["span_content_normalized_match_rate"])
                if not math.isnan(rates["token_match_rate"]):
                    token_rates.append(rates["token_match_rate"])
                num_spans_per_sample.append(rates["num_spans"])
                sample_match_details.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "answer_text": answer_text,
                    "span_match_rate": rates["span_match_rate"],
                    "span_normalized_match_rate": rates["span_normalized_match_rate"],
                    "span_boundary_match_rate": rates["span_boundary_match_rate"],
                    "span_content_match_rate": rates["span_content_match_rate"],
                    "span_content_normalized_match_rate": rates["span_content_normalized_match_rate"],
                    "token_match_rate": rates["token_match_rate"],
                    "num_spans": rates["num_spans"],
                    "active_spans": rates["active_spans"],
                    "span_details": rates["span_details"],
                })
                # Collect unmatched span pairs for manual inspection (cap at 50)
                if len(unmatched_pairs) < 50:
                    for detail in rates["span_details"]:
                        if len(unmatched_pairs) >= 50:
                            break
                        if detail.get("student_text") is not None and not detail.get("matched_strict", True):
                            unmatched_pairs.append({
                                "answer_text": answer_text,
                                "char_span": detail["char_span"],
                                "student_text": detail["student_text"],
                                "teacher_text": detail["teacher_text"],
                                "matched_normalized": detail.get("matched_normalized", False),
                            })

        if was_training:
            model.student.train()

        if not span_rates:
            agg: Dict = {
                "phase": phase_label,
                "global_step": global_step,
                "num_samples": 0,
                "span_match_rate_mean": float("nan"),
                "span_normalized_match_rate_mean": float("nan"),
                "span_boundary_match_rate_mean": float("nan"),
                "span_content_match_rate_mean": float("nan"),
                "span_content_normalized_match_rate_mean": float("nan"),
                "token_match_rate_mean": float("nan"),
            }
        else:
            agg = {
                "phase": phase_label,
                "global_step": global_step,
                "num_samples": len(span_rates),
                "span_match_rate_mean": float(np.mean(span_rates)),
                "span_match_rate_std": float(np.std(span_rates)),
                "span_normalized_match_rate_mean": float(np.mean(span_norm_rates)) if span_norm_rates else float("nan"),
                "span_normalized_match_rate_std": float(np.std(span_norm_rates)) if span_norm_rates else float("nan"),
                "span_boundary_match_rate_mean": float(np.mean(span_boundary_rates)) if span_boundary_rates else float("nan"),
                "span_boundary_match_rate_std": float(np.std(span_boundary_rates)) if span_boundary_rates else float("nan"),
                "span_content_match_rate_mean": float(np.mean(span_content_rates)) if span_content_rates else float("nan"),
                "span_content_match_rate_std": float(np.std(span_content_rates)) if span_content_rates else float("nan"),
                "span_content_normalized_match_rate_mean": float(np.mean(span_content_norm_rates)) if span_content_norm_rates else float("nan"),
                "span_content_normalized_match_rate_std": float(np.std(span_content_norm_rates)) if span_content_norm_rates else float("nan"),
                "token_match_rate_mean": float(np.mean(token_rates)) if token_rates else float("nan"),
                "token_match_rate_std": float(np.std(token_rates)) if token_rates else float("nan"),
                "avg_spans_per_sample": float(np.mean(num_spans_per_sample)),
                "samples": sample_match_details,
            }

        # Write up to 50 unmatched span pairs for manual inspection
        if unmatched_pairs:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                dump_path = os.path.join(
                    self.output_dir, f"unmatched_spans_{phase_label}.json"
                )
                with open(dump_path, "w") as fh:
                    json.dump(_clean_for_json(unmatched_pairs), fh, indent=2, ensure_ascii=False)
                print(
                    f"[SpanMatchEvaluator] Dumped {len(unmatched_pairs)} unmatched span pairs "
                    f"to: {dump_path}"
                )
            except Exception as e:
                print(f"[SpanMatchEvaluator] Failed to write unmatched spans dump: {e}")

        self._log_and_save(agg)
        return agg

    def _log_and_save(self, results: Dict) -> None:
        phase = results.get("phase", "unknown")
        n = results.get("num_samples", 0)
        smr   = results.get("span_match_rate_mean", float("nan"))
        snmr  = results.get("span_normalized_match_rate_mean", float("nan"))
        sbmr  = results.get("span_boundary_match_rate_mean", float("nan"))
        scmr  = results.get("span_content_match_rate_mean", float("nan"))
        scnmr = results.get("span_content_normalized_match_rate_mean", float("nan"))
        tmr   = results.get("token_match_rate_mean", float("nan"))

        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not math.isnan(v) else "nan"

        print(
            f"[SpanMatchEvaluator] {phase} "
            f"(step={results.get('global_step', '?')}, n={n}): "
            f"span_boundary={_fmt(sbmr)}, span_content={_fmt(scmr)}, "
            f"span_content_norm={_fmt(scnmr)}, token={_fmt(tmr)} "
            f"[legacy span_match={_fmt(smr)}]"
        )

        if n > 0:
            try:
                wandb.log({
                    "span_match/span_match_rate": smr,
                    "span_match/span_normalized_match_rate": snmr,
                    "span_match/span_boundary_match_rate": sbmr,
                    "span_match/span_content_match_rate": scmr,
                    "span_match/span_content_normalized_match_rate": scnmr,
                    "span_match/token_match_rate": tmr,
                    f"span_match/{phase}/span_match_rate": smr,
                    f"span_match/{phase}/span_normalized_match_rate": snmr,
                    f"span_match/{phase}/span_boundary_match_rate": sbmr,
                    f"span_match/{phase}/span_content_match_rate": scmr,
                    f"span_match/{phase}/span_content_normalized_match_rate": scnmr,
                    f"span_match/{phase}/token_match_rate": tmr,
                })
            except Exception:
                pass

        self._history.append(results)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, "span_match_history.json")
            with open(out_path, "w") as fh:
                json.dump(_clean_for_json(self._history), fh, indent=2)
        except Exception as e:
            print(f"[SpanMatchEvaluator] Failed to write history: {e}")

    def plot_history(self, output_path: Optional[str] = None) -> None:
        """Save a line plot of span/token match rate across training phases."""
        records = [r for r in self._history if r.get("num_samples", 0) > 0]
        if not records:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            phases = [r["phase"] for r in records]
            steps  = [r["global_step"] for r in records]
            smr    = [r.get("span_match_rate_mean", float("nan")) for r in records]
            snmr   = [r.get("span_normalized_match_rate_mean", float("nan")) for r in records]
            tmr    = [r.get("token_match_rate_mean", float("nan")) for r in records]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, smr,  marker="o", linewidth=1.6, label="Span match rate (strict)")
            ax.plot(steps, snmr, marker="^", linewidth=1.6, linestyle="-.",
                    label="Span match rate (normalized)")
            ax.plot(steps, tmr,  marker="s", linestyle="--", linewidth=1.6,
                    label="Token match rate")
            for i, phase in enumerate(phases):
                if not math.isnan(smr[i]):
                    ax.annotate(
                        phase, (steps[i], smr[i]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="tab:blue",
                    )
            ax.set_xlabel("Global training step")
            ax.set_ylabel("Match rate")
            ax.set_title("Student–teacher span/token match rate across training phases")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            out = output_path or os.path.join(self.output_dir, "span_match_history.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[SpanMatchEvaluator] Saved history plot to: {out}")
        except Exception as e:
            print(f"[SpanMatchEvaluator] Failed to plot history: {e}")
