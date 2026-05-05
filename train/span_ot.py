"""SpanOT-KD: span-selective distillation weights on top of MultiLevelOT.

This module implements the **per-sample position-weight construction** described
in Sections 3-5 of the SpanOT-KD methodology. The numerical loss math itself
(HAD / SL / SD) lives in ``models.distillation_model.DistillationLoss``; this
file is responsible only for turning a (truncated, ranked) batch of teacher /
student probability distributions into a position-weight tensor that can be
multiplied into those losses.

Pipeline (per sample, per training step):

1.  Identify *aligned spans* between the student-side and teacher-side
    tokenisations of the **ground-truth answer string**, by reusing
    :func:`train.span_match.find_parent_token`. A span is the minimal
    character range whose boundaries are respected by both tokenisations
    (Definition 1 of the methodology).
2.  Compute the **per-token entropy gap**
    ``g(t) = H(s^k_t) - H(t^k_t)`` from the truncated top-k distributions
    that MultiLevelOT already produces (Eq. 8-9).
3.  Aggregate per span: either ``mean`` over teacher tokens in the span
    (Eq. 10) or ``sum`` (Eq. 11). The teacher side is canonical because the
    methodology indexes loss rows by teacher position (see Eq. 16).
4.  Sort spans by aggregated gap, designate the top ``r%`` as
    high-priority (weight ``1.0``); the rest are low-priority (weight
    ``delta``). This is Eq. 12-13.
5.  Build a per-position weight vector ``omega[t]`` of length ``T_max``:
       * ``mu(span)``         if teacher position ``t`` falls in any span
                              (matched region)
       * ``1.0``              otherwise (non-matched region)
    This is Eq. 14.

The returned weight tensor has shape ``(B, T_max)`` and is consumed by
``DistillationLoss`` to produce the span-weighted HAD / SL / SD components.
When ``top_r >= 1.0`` or ``low_delta == 1.0`` the weights collapse to all
ones and the loss reduces *exactly* to the original MultiLevelOT objective —
verified by inspection of Eq. 17 of the methodology.

Edge cases (all return all-ones weights, recovering MultiLevelOT):
  * empty / blank answer text;
  * tokeniser produces no usable offset map;
  * no parent spans found between the two tokenisations;
  * every span's teacher-side or student-side index list is empty.

The character-alignment step is exact — no whitespace heuristics — so the
only source of approximation is the sequence-level top-k truncation that
MultiLevelOT applies before this code is invoked.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from train.span_match import find_parent_token


def _per_token_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy along the last dim.

    ``probs`` has shape ``(..., k)`` and contains *probability values* (not
    logits) over the top-k support produced by MultiLevelOT's sequence-level
    ranking + truncation. Entries may be exactly zero (zero-padded rows
    beyond the answer length, or zeros introduced by the diff_size pad);
    ``clamp(min=eps)`` guards ``log(0)``.

    Returns a tensor of shape ``probs.shape[:-1]`` with the entropy of each
    distribution. The result is **never multiplied by anything** — callers
    interpret it directly as ``H(p)`` in nats.
    """
    safe = probs.clamp(min=eps)
    return -(probs * safe.log()).sum(dim=-1)


def _safe_offset_map(
    answer_text: str,
    tokenizer,
) -> List[Tuple[int, int]]:
    """Tokenize ``answer_text`` and return a list of non-degenerate char offsets.

    Special-token offsets ``(0, 0)`` are filtered out — they carry no
    character coverage and would otherwise confuse the parent-span builder.
    Returns ``[]`` when the tokenizer cannot produce offsets (slow tokenizer
    fallback) so the caller can short-circuit to all-ones weights.
    """
    if not answer_text or not answer_text.strip():
        return []
    try:
        enc = tokenizer(
            answer_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except (TypeError, ValueError):
        return []
    offsets = enc.get("offset_mapping")
    if not offsets:
        return []
    return [(int(s), int(e)) for s, e in offsets if s != e]


def compute_position_weights_one_sample(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_size: int,
    teacher_size: int,
    student_offsets: List[Tuple[int, int]],
    teacher_offsets: List[Tuple[int, int]],
    top_r: float,
    low_delta: float,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Build the position-weight vector for a single sample (Eq. 14).

    Parameters
    ----------
    student_probs / teacher_probs:
        Truncated top-k probability distributions, shape ``(T_max, k)``.
        Rows beyond the respective answer size are zero-padded.
    student_size / teacher_size:
        Number of valid (non-padded) rows on each side.
    student_offsets / teacher_offsets:
        Character (start, end) offsets produced by re-tokenising the
        ground-truth answer text. Index ``i`` corresponds to row ``i`` of
        the answer-truncated logits, so we clip both lists to
        ``min(student_size, len(student_offsets))`` etc. before alignment.
    top_r:
        Fraction of spans (sorted by entropy gap, descending) that receive
        weight ``1.0``. ``top_r >= 1.0`` collapses to MultiLevelOT.
    low_delta:
        Weight assigned to non-top-r spans, expected in ``(0, 0.1]`` per
        the methodology. ``low_delta == 1.0`` also collapses to MultiLevelOT.
    aggregation:
        ``"mean"`` (Eq. 10, length-normalised) or ``"sum"`` (Eq. 11,
        accumulated). Mean pooling makes spans of different sizes
        comparable; sum aggregation amplifies long spans.

    Returns
    -------
    Tensor of shape ``(T_max,)`` on the same device as ``teacher_probs``.
    All-ones in three cases: no answer text, no aligned spans found, or
    every span has empty student/teacher index list. In those cases the
    sample contributes to the loss under the original MultiLevelOT
    formulation, which is the documented fallback.
    """
    T_max = teacher_probs.size(0)
    device = teacher_probs.device
    weights = torch.ones(T_max, device=device)

    s_len = min(student_size, len(student_offsets))
    t_len = min(teacher_size, len(teacher_offsets))
    if s_len <= 0 or t_len <= 0:
        return weights

    s_dict, t_dict = find_parent_token(student_offsets[:s_len], teacher_offsets[:t_len])
    if not s_dict or not t_dict:
        return weights

    # Per-token entropy gap g(t) = H(s_t) - H(t_t), Eq. 9.
    H_s = _per_token_entropy(student_probs)  # (T_max,)
    H_t = _per_token_entropy(teacher_probs)  # (T_max,)
    g = H_s - H_t  # (T_max,)

    spans: List[Tuple[List[int], float]] = []
    for span_key, t_indices in t_dict.items():
        s_indices = s_dict.get(span_key, [])
        t_idx = [i for i in t_indices if i < t_len]
        s_idx = [i for i in s_indices if i < s_len]
        if not t_idx or not s_idx:
            # Span not "active" on both sides — skip and leave its positions
            # at unit weight (recovers MultiLevelOT on those positions).
            continue
        gap_vals = g[t_idx]  # entropy gap at teacher positions of this span
        if aggregation == "sum":
            agg_gap = float(gap_vals.sum().item())
        else:
            agg_gap = float(gap_vals.mean().item())
        spans.append((t_idx, agg_gap))

    if not spans:
        return weights

    # Sort descending by gap, top-r% receive weight 1.0; rest receive delta.
    spans.sort(key=lambda x: x[1], reverse=True)
    if top_r >= 1.0:
        num_high = len(spans)
    else:
        # ceil so any positive top_r selects at least one span when len>=1
        num_high = max(1, int(math.ceil(top_r * len(spans))))

    for span_rank, (t_idx, _) in enumerate(spans):
        mu = 1.0 if span_rank < num_high else float(low_delta)
        for pos in t_idx:
            if 0 <= pos < T_max:
                weights[pos] = mu
    return weights


def compute_batch_position_weights(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_sizes: List[int],
    teacher_sizes: List[int],
    student_labels: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    top_r: float,
    low_delta: float,
    aggregation: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Batched wrapper: build a (B, T_max) position-weight tensor.

    For each sample we:
      1. Decode the ground-truth answer text from the *student* labels —
         positions where ``label != ignore_index``. We use the student side
         because ``DistillationLoss`` already decodes student labels for
         CE; the teacher side often contains additional BOS/EOS artefacts
         that re-tokenise inconsistently.
      2. Re-tokenise the same string with both tokenisers to get character
         offsets.
      3. Run :func:`compute_position_weights_one_sample`.

    The decode→re-tokenise round-trip is the standard trick used elsewhere
    in this codebase (see ``train.span_match``) and is exact for our
    purposes: the SR-truncated logits at row ``t`` correspond to the
    answer token that ``offset_mapping[t]`` maps to. Any tokenizer
    re-encoding artefacts (off-by-one BOS) are absorbed by clipping
    ``min(answer_size, len(offsets))`` per sample.

    The returned weights live on the same device as ``teacher_probs`` so
    they multiply into the loss components without an extra ``.to()``.
    """
    B, T_max, _ = teacher_probs.shape
    device = teacher_probs.device
    weights = torch.ones((B, T_max), device=device)

    if student_tokenizer is None or teacher_tokenizer is None:
        # Should not happen when SpanOT is enabled; defensive — ensures
        # the all-ones fallback degrades gracefully to vanilla MultiLevelOT.
        return weights

    for b in range(B):
        raw_labels = student_labels[b]
        answer_token_ids = raw_labels[raw_labels != ignore_index].detach().cpu().tolist()
        if not answer_token_ids:
            continue
        answer_text = student_tokenizer.decode(answer_token_ids, skip_special_tokens=True)

        s_offsets = _safe_offset_map(answer_text, student_tokenizer)
        t_offsets = _safe_offset_map(answer_text, teacher_tokenizer)
        if not s_offsets or not t_offsets:
            continue

        weights[b] = compute_position_weights_one_sample(
            student_probs=student_probs[b],
            teacher_probs=teacher_probs[b],
            student_size=int(student_sizes[b]),
            teacher_size=int(teacher_sizes[b]),
            student_offsets=s_offsets,
            teacher_offsets=t_offsets,
            top_r=top_r,
            low_delta=low_delta,
            aggregation=aggregation,
        )
    return weights
