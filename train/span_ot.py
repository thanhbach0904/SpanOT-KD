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
2.  Compute **per-token entropies** on each side from the truncated top-k
    distributions that MultiLevelOT already produces:
    ``H_s[i] = H(s^k_i)`` for student row ``i`` and
    ``H_t[j] = H(t^k_j)`` for teacher row ``j``.
3.  Aggregate **on each side separately** within a span, then take the
    difference at the span level (Eq. 9-11 of the methodology, applied
    per-side rather than per-row):
        g(S_m) = agg_{i in Q(S_m)} H_s[i]  -  agg_{j in T(S_m)} H_t[j]
    where ``Q(S_m)`` are the student rows in the span and ``T(S_m)`` are
    the teacher rows. ``agg`` is either mean (length-normalised; default)
    or sum (accumulated; ablation only).

    Why per-side, not per-row: the student-side row ``i`` and the teacher-
    side row ``i`` index different tokenisations and therefore predict
    different character spans of the same answer string. Subtracting
    ``H_s[i] - H_t[i]`` at a shared row index is only meaningful when the
    two tokenisations coincide, which is precisely what the cross-tokenizer
    setting rules out. Per-side aggregation respects the span structure:
    both sides aggregate over **their own** rows of the span before any
    cross-side comparison.
4.  Sort spans by ``g(S_m)``, designate the top ``r%`` as high-priority
    (weight ``1.0``); the rest are low-priority (weight ``delta``). This
    is Eq. 12-13.
5.  Build a per-position weight vector ``omega[t]`` of length ``T_max``:
       * ``mu(span(t))``     if teacher position ``t`` falls in any aligned span
       * ``1.0``              otherwise (safe fallback only — see below)
    This is Eq. 14 of the methodology.

The returned weight tensor has shape ``(B, T_max)`` and is consumed by
``DistillationLoss`` to produce the span-weighted HAD / SL / SD components.
When ``top_r >= 1.0`` or ``low_delta == 1.0`` the weights collapse to all
ones and the loss reduces *exactly* to the original MultiLevelOT objective.

**A note on aggregation choice.** Under per-side aggregation, ``mean`` and
``sum`` are no longer just amplification-of-long-spans choices: ``sum``
introduces a systematic bias toward the side with more tokens whenever
|Q(S_m)| != |T(S_m)|, which is the normal case in cross-tokenizer settings.
``mean`` length-normalises each side independently and is recommended as
the default; ``sum`` is retained only so the methodology's two aggregation
variants remain ablate-able.

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

# Fires once per process to show how parent-span alignment works.
_SPAN_ALIGN_DEBUG_PRINTED: bool = False


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


def _aggregate(values: torch.Tensor, mode: str) -> float:
    """Aggregate a 1-D tensor of entropies on one side of a span.

    ``mode`` is ``"mean"`` (default in the methodology, recommended in the
    cross-tokenizer setting because it length-normalises each side
    independently) or ``"sum"`` (retained for ablation; introduces a
    systematic bias toward the side with more tokens in the span).

    The empty-tensor case is guarded upstream — callers only invoke this
    when both Q(S_m) and T(S_m) are non-empty. We still defend against an
    empty input to avoid NaN, returning 0.0 as a neutral element of the
    span-gap difference.
    """
    if values.numel() == 0:
        return 0.0
    if mode == "sum":
        return float(values.sum().item())
    # default: mean
    return float(values.mean().item())


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
        Rows beyond the respective answer size are zero-padded. The row
        index has different semantics on each side: student row ``i`` is
        the distribution at the ``i``-th token of the **student**
        tokenisation of the GT answer; teacher row ``j`` is the
        distribution at the ``j``-th token of the **teacher** tokenisation.
        These row spaces are not aligned in general — entropy comparisons
        between sides must aggregate per-side within a span.
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
        ``"mean"`` (Eq. 10, length-normalised; recommended) or ``"sum"``
        (Eq. 11, accumulated; retained for ablation). Aggregation is now
        applied **on each side separately** before the span-level
        difference is taken — see module docstring step 3 for why.

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

    # Per-token entropies indexed on each side's own row space.
    # CRITICAL: H_s[i] is the entropy of the student's distribution at
    # STUDENT-tokenisation row i; H_t[j] is the entropy at TEACHER-
    # tokenisation row j. The two row spaces predict different character
    # spans of the same answer string. We must therefore aggregate H_s
    # over Q(S_m) and H_t over T(S_m) **separately** before differencing
    # at the span level. Subtracting H_s[i] - H_t[i] at a shared row
    # index — the old implementation — mixes entropies of unrelated
    # predictions whenever the two tokenisations diverge within a span.
    H_s = _per_token_entropy(student_probs)  # (T_max,)
    H_t = _per_token_entropy(teacher_probs)  # (T_max,)

    spans: List[Tuple[List[int], float]] = []
    for span_key, t_indices in t_dict.items():
        s_indices = s_dict.get(span_key, [])
        t_idx = [i for i in t_indices if i < t_len]
        s_idx = [i for i in s_indices if i < s_len]
        if not t_idx or not s_idx:
            # Span not "active" on both sides — skip and leave its positions
            # at unit weight (recovers MultiLevelOT on those positions).
            continue
        # Pull entropies on each side's OWN rows of the span. Note that
        # |s_idx| and |t_idx| can (and usually will) differ across
        # tokenisers — this is precisely why we aggregate per-side.
        Hs_span = H_s[torch.as_tensor(s_idx, device=device, dtype=torch.long)]
        Ht_span = H_t[torch.as_tensor(t_idx, device=device, dtype=torch.long)]
        # Span-level gap: difference of per-side aggregated entropies.
        # Under aggregation="mean" each side is length-normalised, so the
        # difference is comparable regardless of |s_idx| vs |t_idx|.
        # Under aggregation="sum" the side with more tokens contributes
        # more terms — this is the documented ablation variant.
        agg_gap = _aggregate(Hs_span, aggregation) - _aggregate(Ht_span, aggregation)
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

    The decode -> re-tokenise round-trip is the standard trick used elsewhere
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

        global _SPAN_ALIGN_DEBUG_PRINTED
        if not _SPAN_ALIGN_DEBUG_PRINTED:
            _SPAN_ALIGN_DEBUG_PRINTED = True
            s_len_dbg = min(int(student_sizes[b]), len(s_offsets))
            t_len_dbg = min(int(teacher_sizes[b]), len(t_offsets))
            s_toks = [answer_text[s:e] for s, e in s_offsets[:s_len_dbg]]
            t_toks = [answer_text[s:e] for s, e in t_offsets[:t_len_dbg]]
            s_dict_dbg, t_dict_dbg = find_parent_token(s_offsets[:s_len_dbg], t_offsets[:t_len_dbg])
            print("\n[SpanOT-KD alignment debug] ---- first sample ----", flush=True)
            print(f"  answer_text      : {answer_text!r}", flush=True)
            print(f"  student tokens   : {s_toks}", flush=True)
            print(f"  teacher tokens   : {t_toks}", flush=True)
            print(f"  parent spans ({len(s_dict_dbg)}):", flush=True)
            for span_key, s_idxs in s_dict_dbg.items():
                t_idxs = t_dict_dbg.get(span_key, [])
                s_span_toks = [s_toks[i] for i in s_idxs if i < len(s_toks)]
                t_span_toks = [t_toks[i] for i in t_idxs if i < len(t_toks)]
                print(f"    chars {span_key}: student={s_span_toks}  teacher={t_span_toks}", flush=True)
            print("[SpanOT-KD alignment debug] ---- end ----\n", flush=True)

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