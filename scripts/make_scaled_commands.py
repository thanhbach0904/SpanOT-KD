#!/usr/bin/env python3
"""WP8: print the `mlot_scaled` (global-scalar) commands from SpanOT-KD w_eff.

For each r in {0.3, 0.5, 0.7}: w_eff(r) = mean over the given seeds of
output_<ds>_<RUN_TAG(spanotkd, seed)>/r_<r>/w_eff.json["overall"], and the
command trains vanilla MLOT with distil_factor = base * w_eff(r).
Dev selection over the three r runs is done later by aggregate_controls.py
with the same rule as finetuning._run_sweep (dev generative metric, max).

VALIDITY (see change_logs.md, WP2): only meaningful because every component
is linear in omega. Approximations that remain and must be stated with any
mlot_scaled number:
  * w_eff varies during training; the scalar is its step-weighted overall mean.
  * w_eff_valid averages omega over the L1 support [:min(s, t)]. KL and Sinkhorn
    also cover rows past that (padding / longer side) at omega = 1 in SpanOT-KD,
    which the global scalar down-weights too.
  * mean_t(omega_t * l_t) != w_eff * mean_t(l_t) unless omega and the per-token
    loss are uncorrelated - which is exactly what SpanOT-KD tries to violate.
    The scalar matches total weight, not the weighted loss value.

Usage:
  python scripts/make_scaled_commands.py --repo $HOME/SpanOT-KD --dataset qed \
      --teacher /workspace/models/Llama-2-7b-chat-hf --student opt-350m --seeds 42 4 63
"""
import argparse
import json
import os
import statistics
import sys

R_VALUES = [0.3, 0.5, 0.7]


def run_tag(student, teacher, method, seed):
    return f"{student.split('-')[0]}_{os.path.basename(teacher.rstrip('/'))}_{method}_seed{seed}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", required=True, choices=["qed", "fairytaleqa"])
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--seeds", required=True, nargs="+", type=int)
    p.add_argument("--base_factor", type=float, default=0.15,
                   help="distil_factor actually passed by run_experiments_*.sh (0.15, not the config default 1.5)")
    args = p.parse_args()

    script = f"run_experiments_{args.dataset}.sh"
    missing = []
    for r in R_VALUES:
        vals = []
        for seed in args.seeds:
            path = os.path.join(args.repo, f"output_{args.dataset}_{run_tag(args.student, args.teacher, 'spanotkd', seed)}",
                                f"r_{r:.1f}", "w_eff.json")
            try:
                with open(path) as fh:
                    v = json.load(fh).get("overall")
            except FileNotFoundError:
                v = None
            if v is None:
                missing.append(path)
            else:
                vals.append(float(v))
        if not vals:
            print(f"# r={r}: no w_eff.json found for any seed - no command generated", file=sys.stderr)
            continue
        w = statistics.mean(vals)
        spread = f", std={statistics.stdev(vals):.4f}" if len(vals) > 1 else ""
        factor = args.base_factor * w
        print(f"# r={r}: w_eff = {w:.6f} (N={len(vals)} seeds{spread}) -> distil_factor = {factor:.6f}")
        for seed in args.seeds:
            print(f"DISTIL_FACTOR={factor:.6f} SCALED_R={r:.1f} bash {script} {args.teacher} {args.student} {seed} scaled")
    for m in missing:
        print(f"# MISSING: {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
