"""Generate tests/fixtures/span_weights_snapshot.json from the CURRENT
train.span_ot code. Run once BEFORE editing span_ot.py; the snapshot test
(test_span_matched_select.TestExistingArmsUnchanged) then asserts that the
entropy and random arms still produce exactly these weights.

Run: python tests/make_span_snapshot.py   (from Multi-Level-OT/)
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.span_ot import compute_position_weights_one_sample

# Two tokenisations of a 12-char string with different boundaries, so spans
# have unequal |s_idx| / |t_idx| (the realistic cross-tokenizer case).
S_OFFSETS = [(0, 2), (2, 5), (5, 6), (6, 9), (9, 12)]
T_OFFSETS = [(0, 1), (1, 2), (2, 5), (5, 8), (8, 9), (9, 10), (10, 12)]
UNIT_OFFSETS = [(i, i + 1) for i in range(6)]


def snapshot_cases():
    cases = []
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        T_max, k = 8, 5
        s = torch.softmax(torch.randn(T_max, k, generator=g) * 2, dim=-1)  # [8, 5]
        t = torch.softmax(torch.randn(T_max, k, generator=g) * 2, dim=-1)  # [8, 5]
        for name, so, to, ss, ts in (
            ("cross", S_OFFSETS, T_OFFSETS, 5, 7),
            ("unit", UNIT_OFFSETS, UNIT_OFFSETS, 6, 6),
        ):
            for top_r in (0.3, 0.5, 0.7, 1.0):
                cases.append(dict(seed=seed, name=name, top_r=top_r, s=s.tolist(), t=t.tolist(),
                                  s_off=so, t_off=to, s_size=ss, t_size=ts))
    return cases


def run_case(c, mode, rng_seed=0, pool="active"):
    return compute_position_weights_one_sample(
        student_probs=torch.tensor(c["s"]), teacher_probs=torch.tensor(c["t"]),
        student_size=c["s_size"], teacher_size=c["t_size"],
        student_offsets=[tuple(x) for x in c["s_off"]], teacher_offsets=[tuple(x) for x in c["t_off"]],
        top_r=c["top_r"], low_delta=0.1, select_mode=mode,
        rng=random.Random(rng_seed) if mode == "random" else None, random_pool=pool,
    ).tolist()


if __name__ == "__main__":
    out = []
    for c in snapshot_cases():
        c = dict(c)
        c["entropy"] = run_case(c, "entropy")
        c["random_active"] = run_case(c, "random", rng_seed=c["seed"], pool="active")
        c["random_pos"] = run_case(c, "random", rng_seed=c["seed"], pool="pos")
        out.append(c)
    path = Path(__file__).resolve().parent / "fixtures" / "span_weights_snapshot.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"wrote {len(out)} cases to {path}")
