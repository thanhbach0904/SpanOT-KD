"""SpanOT-KD per-sample tracer — a *read-only* side analysis.

This module exists to find flaws in the SpanOT-KD methodology by dumping every
intermediate the per-position weight construction walks through, for a single
**deterministically chosen** sample. It changes nothing in the training math:
:func:`trace_sample` only reads detached tensors that the loss already holds and
writes a JSON + human-readable report to disk.

Activation is entirely env-gated, so when ``SPANOT_TRACE`` is unset the live
pipeline is byte-for-byte identical to before:

    SPANOT_TRACE=1            enable the tracer (rank 0 only)
    SPANOT_TRACE_STEP=0       optimiser step to trace (default 0)
    SPANOT_TRACE_B=0          batch index within that step (default 0)
    SPANOT_TRACE_N=1          trace this many consecutive steps from STEP
    SPANOT_TRACE_DIR=...      output dir (default ./spanot_trace)

Determinism: the BatchSampler shuffles with ``random.Random(seed)`` (see
data/sampler.py), so for a fixed ``--seed`` the batch at step S and the sample
at batch index B are fully determined. (step, B) is therefore a stable, seed-
reproducible anchor — no dependence on dataset internals.

The report is organised in seven stages, each targeting a specific suspected
flaw (see the module-level FLAW_NOTES). The three prioritised hypotheses are:

  * ENTROPY TRUNCATION BIAS — the entropies that drive span selection are
    computed on the unnormalised, global-top-50 column projection that
    MultiLevelOT produces, not on the true predictive distribution. Stage 3
    prints the truncated entropy *next to* the full-vocabulary entropy and the
    retained top-50 probability mass so the bias is directly measurable.
  * DECODE -> RE-TOKENIZE DRIFT — span offsets come from decoding the label
    ids and re-encoding the text; if re-encoding re-segments the answer, offset
    index i no longer corresponds to logits row i and every span index is
    silently shifted. Stage 2 lays the original per-token decode beside the
    re-encoded offset substrings so a mismatch is visible token by token.
  * ROW-SPACE MISMATCH — ``omega[t]`` is built from *teacher* parent-span
    membership but multiplies the L1/KL terms, which compare student row t
    against teacher row t positionally. Stage 1 prints the student- and
    teacher-predicted token at each position side by side so you can see
    whether row t even refers to the same word on both sides.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from train.span_match import find_parent_token
from train.span_ot import (
    _aggregate,
    _per_token_entropy,
    _safe_offset_map,
    compute_position_weights_one_sample,
)

FLAW_NOTES = {
    "entropy_truncation_bias": (
        "Stage 3: compare H_trunc (top-50 projection, unnormalised) vs H_full "
        "(full-vocab softmax). If they diverge, span selection is driven by a "
        "truncation artefact, not the real predictive uncertainty."
    ),
    "decode_retokenize_drift": (
        "Stage 2: original_token_text vs offset_substring per index. Any "
        "len/text mismatch means span indices are applied to the wrong logits "
        "rows."
    ),
    "row_space_mismatch": (
        "Stage 1 + Stage 6: student_pred vs teacher_pred at the same position. "
        "omega[t] is teacher-indexed but scales L1/KL terms that couple "
        "student[t] with teacher[t]; if the words differ, the weight is applied "
        "to a position whose two sides are unrelated."
    ),
    "downweight_only": (
        "Stage 6: max(omega) is 1.0 by construction. Unmatched positions AND "
        "top-priority spans both sit at 1.0, so enabling SpanOT can only ever "
        "*reduce* weights relative to MultiLevelOT, never emphasise."
    ),
}


def should_trace(step: Optional[int], rank: int) -> Optional[Tuple[int, int]]:
    """Return (trace_b, trace_n_remaining_ok) if this (step, rank) should trace.

    Returns the configured batch index to trace, or ``None`` when the tracer is
    disabled, this is not rank 0, or ``step`` is outside the traced window.
    Kept deliberately cheap so the common (disabled) path is a single env read.
    """
    if not os.environ.get("SPANOT_TRACE"):
        return None
    if rank != 0 or step is None:
        return None
    trace_step = int(os.environ.get("SPANOT_TRACE_STEP", "0"))
    trace_n = int(os.environ.get("SPANOT_TRACE_N", "1"))
    if not (trace_step <= step < trace_step + max(1, trace_n)):
        return None
    trace_b = int(os.environ.get("SPANOT_TRACE_B", "0"))
    return (trace_b, step)


def _decode_each(tokenizer, ids: List[int]) -> List[str]:
    """Decode each id individually so we can lay tokens beside char offsets."""
    return [tokenizer.decode([i], skip_special_tokens=False) for i in ids]


def _full_vocab_entropy(
    raw_logits_row: torch.Tensor,
    answer_index: int,
    answer_size: int,
    temperature: float,
    eps: float = 1e-12,
) -> List[float]:
    """Entropy of the *full* softmax distribution at each answer position.

    ``raw_logits_row`` is the untouched model output for one sample,
    shape ``(seq_len, V)``. This is the reference the truncated top-50 entropy
    in Stage 3 is compared against.
    """
    end = answer_index + answer_size
    sl = raw_logits_row[answer_index:end].float()
    probs = F.softmax(sl / max(temperature, 1e-6), dim=-1)
    safe = probs.clamp(min=eps)
    H = -(probs * safe.log()).sum(dim=-1)
    return [float(x) for x in H.tolist()]


def trace_sample(
    *,
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_raw_logits: torch.Tensor,
    teacher_raw_logits: torch.Tensor,
    student_size: int,
    teacher_size: int,
    student_answer_index: int,
    teacher_answer_index: int,
    student_label_row: torch.Tensor,
    teacher_label_row: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    student_temperature: float,
    teacher_temperature: float,
    top_r: float,
    low_delta: float,
    aggregation: str,
    ignore_index: int,
    meta: dict,
) -> dict:
    """Walk one sample through the full SpanOT weight construction and dump it.

    Every quantity is recomputed from detached tensors; nothing here feeds back
    into the loss. Returns the structured record (also written to disk) so a
    caller could assert on it in a unit test.
    """
    rec: dict = {"meta": dict(meta), "flaw_notes": FLAW_NOTES}
    device = teacher_probs.device

    # ----- Stage 0: sample identity -------------------------------------
    raw_ids = student_label_row[student_label_row != ignore_index].detach().cpu().tolist()
    answer_text = student_tokenizer.decode(raw_ids, skip_special_tokens=True)
    rec["stage0_identity"] = {
        "answer_token_ids": raw_ids,
        "answer_text": answer_text,
        "student_answer_index": int(student_answer_index),
        "teacher_answer_index": int(teacher_answer_index),
        "student_answer_size": int(student_size),
        "teacher_answer_size": int(teacher_size),
        "student_label_tokens": _decode_each(student_tokenizer, raw_ids),
    }

    # ----- Stage 1: forward output, position-by-position ----------------
    # The model's actual argmax prediction at each answer position, decoded over
    # the FULL vocab (the sorted top-50 probs cannot be decoded back to token
    # ids). Student row t and teacher row t are laid side by side to expose the
    # row-space mismatch.
    def _argmax_tokens(raw_logits_row, idx, size, tok):
        end = idx + size
        ids = torch.argmax(raw_logits_row[idx:end].float(), dim=-1).detach().cpu().tolist()
        return ids, _decode_each(tok, ids)

    s_pred_ids, s_pred_toks = _argmax_tokens(
        student_raw_logits, student_answer_index, student_size, student_tokenizer
    )
    t_pred_ids, t_pred_toks = _argmax_tokens(
        teacher_raw_logits, teacher_answer_index, teacher_size, teacher_tokenizer
    )
    n_show = max(student_size, teacher_size)
    pos_table = []
    for t in range(n_show):
        pos_table.append({
            "pos": t,
            "student_pred": s_pred_toks[t] if t < len(s_pred_toks) else None,
            "teacher_pred": t_pred_toks[t] if t < len(t_pred_toks) else None,
            "student_label": rec["stage0_identity"]["student_label_tokens"][t]
            if t < len(raw_ids) else None,
        })
    rec["stage1_forward"] = {
        "note": "L1/KL compare student[t] vs teacher[t]; size = min(sizes). "
                "Rows where student_pred != teacher_pred are positionally coupled "
                "but refer to different words.",
        "size_used_by_l1_kl": int(min(student_size, teacher_size)),
        "position_table": pos_table,
    }

    # ----- Stage 2: span alignment + decode->retokenize drift -----------
    s_offsets = _safe_offset_map(answer_text, student_tokenizer)
    t_offsets = _safe_offset_map(answer_text, teacher_tokenizer)
    s_len = min(int(student_size), len(s_offsets))
    t_len = min(int(teacher_size), len(t_offsets))

    s_sub = [answer_text[s:e] for s, e in s_offsets[:s_len]]
    t_sub = [answer_text[s:e] for s, e in t_offsets[:t_len]]

    # Drift check: the offsets come from re-encoding `answer_text`; the logits
    # rows came from `raw_ids`. Compare the original per-token decode against the
    # re-encoded offset substrings on the student side.
    orig_student_toks = rec["stage0_identity"]["student_label_tokens"]
    drift_rows = []
    for i in range(max(len(orig_student_toks), len(s_sub))):
        orig = orig_student_toks[i] if i < len(orig_student_toks) else None
        reenc = s_sub[i] if i < len(s_sub) else None
        drift_rows.append({
            "i": i,
            "original_label_token": orig,
            "reencoded_offset_substring": reenc,
            "match": (orig is not None and reenc is not None
                      and orig.strip() == reenc.strip()),
        })
    s_dict, t_dict = (find_parent_token(s_offsets[:s_len], t_offsets[:t_len])
                      if s_len > 0 and t_len > 0 else ({}, {}))
    span_membership = []
    for span_key, s_idxs in s_dict.items():
        t_idxs = t_dict.get(span_key, [])
        span_membership.append({
            "char_span": list(span_key),
            "student_idx": list(s_idxs),
            "teacher_idx": list(t_idxs),
            "student_tokens": [s_sub[i] for i in s_idxs if i < len(s_sub)],
            "teacher_tokens": [t_sub[i] for i in t_idxs if i < len(t_sub)],
        })
    rec["stage2_alignment"] = {
        "student_offsets": [list(x) for x in s_offsets[:s_len]],
        "teacher_offsets": [list(x) for x in t_offsets[:t_len]],
        "n_student_tokens_logits_vs_offsets": [int(student_size), len(s_offsets)],
        "n_teacher_tokens_logits_vs_offsets": [int(teacher_size), len(t_offsets)],
        "decode_retokenize_drift": drift_rows,
        "drift_detected": any(not r["match"] for r in drift_rows),
        "parent_spans": span_membership,
    }

    # ----- Stage 3: entropy, truncated vs full --------------------------
    H_s_trunc = _per_token_entropy(student_probs)  # (T_max,)
    H_t_trunc = _per_token_entropy(teacher_probs)
    s_mass = student_probs.sum(dim=-1)  # retained top-50 mass per row
    t_mass = teacher_probs.sum(dim=-1)
    H_s_full = _full_vocab_entropy(
        student_raw_logits, student_answer_index, student_size, student_temperature
    )
    H_t_full = _full_vocab_entropy(
        teacher_raw_logits, teacher_answer_index, teacher_size, teacher_temperature
    )

    def _entropy_rows(size, sub, H_trunc, H_full, mass):
        rows = []
        for i in range(size):
            rows.append({
                "row": i,
                "token": sub[i] if i < len(sub) else None,
                "H_trunc_top50": float(H_trunc[i].item()),
                "H_full_vocab": H_full[i] if i < len(H_full) else None,
                "retained_top50_mass": float(mass[i].item()),
            })
        return rows

    rec["stage3_entropy"] = {
        "note": "H_trunc is entropy over the unnormalised global-top-50 columns "
                "(improved_sort keeps the same 50 columns for every row). "
                "Compare against H_full_vocab; retained_top50_mass << 1 means "
                "the truncated entropy is on a small slice of the distribution.",
        "student_rows": _entropy_rows(s_len, s_sub, H_s_trunc, H_s_full, s_mass),
        "teacher_rows": _entropy_rows(t_len, t_sub, H_t_trunc, H_t_full, t_mass),
    }

    # ----- Stage 4: per-span entropy gap --------------------------------
    span_gaps = []
    for span_key, t_indices in t_dict.items():
        s_indices = s_dict.get(span_key, [])
        t_idx = [i for i in t_indices if i < t_len]
        s_idx = [i for i in s_indices if i < s_len]
        if not t_idx or not s_idx:
            span_gaps.append({
                "char_span": list(span_key),
                "skipped": True,
                "reason": "empty student or teacher index list after clipping",
            })
            continue
        Hs_span = H_s_trunc[torch.as_tensor(s_idx, device=device, dtype=torch.long)]
        Ht_span = H_t_trunc[torch.as_tensor(t_idx, device=device, dtype=torch.long)]
        agg_s = _aggregate(Hs_span, aggregation)
        agg_t = _aggregate(Ht_span, aggregation)
        span_gaps.append({
            "char_span": list(span_key),
            "skipped": False,
            "student_idx": s_idx,
            "teacher_idx": t_idx,
            "student_entropies": [float(x) for x in Hs_span.tolist()],
            "teacher_entropies": [float(x) for x in Ht_span.tolist()],
            "agg_mode": aggregation,
            "agg_student": agg_s,
            "agg_teacher": agg_t,
            "gap": agg_s - agg_t,
            "positive": (agg_s - agg_t) > 0,
        })
    rec["stage4_span_gap"] = {
        "note": "gap = agg(H_student) - agg(H_teacher), per-side aggregated. "
                "Only gap > 0 spans are eligible for high priority.",
        "spans": span_gaps,
    }

    # ----- Stage 5: selection -------------------------------------------
    active = [g for g in span_gaps if not g["skipped"]]
    pos_spans = sorted([g for g in active if g["gap"] > 0],
                       key=lambda x: x["gap"], reverse=True)
    neg_spans = [g for g in active if g["gap"] <= 0]
    import math
    if top_r >= 1.0:
        num_high = len(pos_spans)
    elif pos_spans:
        num_high = max(1, int(math.ceil(top_r * len(pos_spans))))
    else:
        num_high = 0
    selection = []
    for rank_i, g in enumerate(pos_spans):
        selection.append({
            "char_span": g["char_span"],
            "gap": g["gap"],
            "rank": rank_i,
            "priority": "high" if rank_i < num_high else "low",
            "weight": 1.0 if rank_i < num_high else float(low_delta),
            "teacher_idx": g["teacher_idx"],
        })
    rec["stage5_selection"] = {
        "n_positive_spans": len(pos_spans),
        "n_negative_spans": len(neg_spans),
        "num_high": num_high,
        "top_r": float(top_r),
        "low_delta": float(low_delta),
        "high_priority": selection,
        "negative_spans_weighted_delta": [
            {"char_span": g["char_span"], "gap": g["gap"], "teacher_idx": g["teacher_idx"]}
            for g in neg_spans
        ],
    }

    # ----- Stage 6: final position weights (fidelity-checked) -----------
    omega = compute_position_weights_one_sample(
        student_probs=student_probs,
        teacher_probs=teacher_probs,
        student_size=int(student_size),
        teacher_size=int(teacher_size),
        student_offsets=s_offsets,
        teacher_offsets=t_offsets,
        top_r=top_r,
        low_delta=low_delta,
        aggregation=aggregation,
    )
    T_max = omega.size(0)
    weight_table = []
    for t in range(min(T_max, max(t_len, int(teacher_size)))):
        weight_table.append({
            "pos": t,
            "teacher_token": t_sub[t] if t < len(t_sub) else None,
            "weight": float(omega[t].item()),
        })
    rec["stage6_position_weights"] = {
        "note": "omega is TEACHER-indexed. max(omega)=1.0 by construction: "
                "enabling SpanOT only ever downweights relative to MultiLevelOT.",
        "omega_max": float(omega.max().item()),
        "omega_min": float(omega.min().item()),
        "omega_mean": float(omega.mean().item()),
        "frac_below_one": float((omega < 1.0).float().mean().item()),
        "weight_table": weight_table,
    }

    # ----- Stage 7: loss impact, weighted vs unweighted -----------------
    size = min(int(student_size), int(teacher_size))
    if size > 0:
        w = omega[:size].to(student_probs.dtype)
        per_pos_l1 = (student_probs[:size] - teacher_probs[:size]).abs().sum(-1)
        l1_unw = float(per_pos_l1.mean().item())
        l1_w = float((per_pos_l1 * w).mean().item())
        # KL_wo single-sample: -sum(softmax(student) * log_softmax(teacher))
        p_s = F.log_softmax(teacher_probs[:size], dim=-1)
        p_t = F.softmax(student_probs[:size], dim=-1)
        per_pos_kl = -(p_t * p_s).sum(-1)
        kl_unw = float(per_pos_kl.mean().item())
        kl_w = float((per_pos_kl * w).mean().item())
        from models.distillation_model import Sinkhorn_seq
        sk = Sinkhorn_seq()
        ps = F.softmax(teacher_probs[:size] / sk.T, dim=-1)
        pt = F.softmax(student_probs[:size] / sk.T, dim=-1)
        sk_unw = float(0.001 * sk.sinkhorn_loss(x=ps, y=pt).item())
        sk_w = float(0.001 * sk.sinkhorn_loss(x=ps, y=pt, row_weights=w).item())
        rec["stage7_loss_impact"] = {
            "size_used": size,
            "l1": {"unweighted": l1_unw, "weighted": l1_w,
                   "ratio": (l1_w / l1_unw) if l1_unw else None},
            "kl": {"unweighted": kl_unw, "weighted": kl_w,
                   "ratio": (kl_w / kl_unw) if kl_unw else None},
            "sinkhorn": {"unweighted": sk_unw, "weighted": sk_w,
                         "ratio": (sk_w / sk_unw) if sk_unw else None},
        }
    else:
        rec["stage7_loss_impact"] = {"size_used": 0, "note": "no overlap"}

    _write_report(rec, meta)
    return rec


def _write_report(rec: dict, meta: dict) -> None:
    out_dir = os.environ.get("SPANOT_TRACE_DIR", "./spanot_trace")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"step{meta.get('step', 0)}_b{meta.get('b', 0)}"
    seed = meta.get("seed")
    if seed is not None:
        stem = f"seed{seed}_" + stem
    json_path = os.path.join(out_dir, stem + ".json")
    txt_path = os.path.join(out_dir, stem + ".txt")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(_render_text(rec))
    print(f"[SpanOT-trace] wrote {json_path} and {txt_path}", flush=True)


def _render_text(rec: dict) -> str:
    L: List[str] = []
    m = rec["meta"]
    L.append("=" * 72)
    L.append(f"SpanOT-KD TRACE  seed={m.get('seed')} step={m.get('step')} b={m.get('b')}")
    L.append("=" * 72)

    s0 = rec["stage0_identity"]
    L.append("\n[Stage 0] Sample identity")
    L.append(f"  answer_text: {s0['answer_text']!r}")
    L.append(f"  student size={s0['student_answer_size']} teacher size={s0['teacher_answer_size']}")

    L.append("\n[Stage 1] Forward output (row-space mismatch check)")
    L.append(f"  {rec['stage1_forward']['note']}")
    L.append(f"  {'pos':>3} | {'student_pred':<18} | {'teacher_pred':<18} | student_label")
    for r in rec["stage1_forward"]["position_table"]:
        flag = "  <-- differ" if (r["student_pred"] != r["teacher_pred"]) else ""
        L.append(f"  {r['pos']:>3} | {str(r['student_pred']):<18} | "
                 f"{str(r['teacher_pred']):<18} | {str(r['student_label'])}{flag}")

    s2 = rec["stage2_alignment"]
    L.append("\n[Stage 2] decode -> re-tokenize drift")
    L.append(f"  drift_detected = {s2['drift_detected']}")
    L.append(f"  student tokens: logits={s2['n_student_tokens_logits_vs_offsets'][0]} "
             f"offsets={s2['n_student_tokens_logits_vs_offsets'][1]}")
    L.append(f"  {'i':>3} | {'original_label':<18} | {'reencoded_offset':<18} | match")
    for r in s2["decode_retokenize_drift"]:
        L.append(f"  {r['i']:>3} | {str(r['original_label_token']):<18} | "
                 f"{str(r['reencoded_offset_substring']):<18} | {r['match']}")
    L.append(f"  parent spans: {len(s2['parent_spans'])}")
    for sp in s2["parent_spans"]:
        L.append(f"    chars {sp['char_span']}: student={sp['student_tokens']} "
                 f"teacher={sp['teacher_tokens']}")

    s3 = rec["stage3_entropy"]
    L.append("\n[Stage 3] Entropy: truncated top-50 vs full vocab")
    L.append(f"  {s3['note']}")
    for side in ("student_rows", "teacher_rows"):
        L.append(f"  -- {side} --")
        L.append(f"  {'row':>3} | {'token':<14} | {'H_trunc':>9} | {'H_full':>9} | top50_mass")
        for r in s3[side]:
            hf = f"{r['H_full_vocab']:.4f}" if r["H_full_vocab"] is not None else "  n/a"
            L.append(f"  {r['row']:>3} | {str(r['token']):<14} | {r['H_trunc_top50']:>9.4f} | "
                     f"{hf:>9} | {r['retained_top50_mass']:.4f}")

    L.append("\n[Stage 4] Per-span entropy gap")
    for g in rec["stage4_span_gap"]["spans"]:
        if g.get("skipped"):
            L.append(f"  span {g['char_span']}: SKIPPED ({g['reason']})")
        else:
            L.append(f"  span {g['char_span']}: gap={g['gap']:+.4f} "
                     f"(agg_s={g['agg_student']:.4f} agg_t={g['agg_teacher']:.4f}) "
                     f"positive={g['positive']}")

    s5 = rec["stage5_selection"]
    L.append("\n[Stage 5] Selection")
    L.append(f"  positive spans={s5['n_positive_spans']} negative={s5['n_negative_spans']} "
             f"num_high={s5['num_high']} (top_r={s5['top_r']}, delta={s5['low_delta']})")
    for sp in s5["high_priority"]:
        L.append(f"    [{sp['priority']:>4}] span {sp['char_span']} gap={sp['gap']:+.4f} "
                 f"-> weight {sp['weight']} (teacher_idx={sp['teacher_idx']})")

    s6 = rec["stage6_position_weights"]
    L.append("\n[Stage 6] Final position weights (teacher-indexed)")
    L.append(f"  {s6['note']}")
    L.append(f"  omega: min={s6['omega_min']} max={s6['omega_max']} mean={s6['omega_mean']:.4f} "
             f"frac_below_1={s6['frac_below_one']:.3f}")
    L.append(f"  {'pos':>3} | {'teacher_token':<18} | weight")
    for r in s6["weight_table"]:
        L.append(f"  {r['pos']:>3} | {str(r['teacher_token']):<18} | {r['weight']}")

    s7 = rec["stage7_loss_impact"]
    L.append("\n[Stage 7] Loss impact (this sample, weighted vs unweighted)")
    if s7.get("size_used", 0) > 0:
        for k in ("l1", "kl", "sinkhorn"):
            d = s7[k]
            ratio = f"{d['ratio']:.4f}" if d["ratio"] is not None else "n/a"
            L.append(f"  {k:<9}: unweighted={d['unweighted']:.6e} "
                     f"weighted={d['weighted']:.6e} ratio={ratio}")
    else:
        L.append(f"  {s7.get('note', 'no overlap')}")
    L.append("")
    return "\n".join(L)
