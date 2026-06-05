"""Span-alignment primitives for cross-tokenizer KD.

Provides find_parent_token() and compute_span_assignments(), which identify
*parent spans* — minimal character ranges whose boundaries are respected by
both the student and teacher tokenisations of the same text.

These two functions are the sole public API.  Everything else (match-rate
diagnostics, SpanMatchEvaluator) has been removed; the alignment code is the
only part that feeds into the live training pipeline (span_ot.py).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


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
