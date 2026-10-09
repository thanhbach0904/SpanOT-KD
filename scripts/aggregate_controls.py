#!/usr/bin/env python3
"""WP10: aggregate the control arms into tables + paired statistics.

Reads, for one (dataset, teacher, student):
  * test metric   eval_results/<ds>_<RUN_TAG>[_loss]/0shots.json  (QED: f1, FairytaleQA: rougeL;
                  the benchmark scripts already scale these by 100)
  * sweep info    output_<ds>_<RUN_TAG>/sweep_summary.json (selected r, dev metric per r)
  * w_eff         output_<ds>_<RUN_TAG>/r_<r>/w_eff.json   ("overall")
  * mlot_scaled   output_<ds>_<student>_<teacher>_mlotscaled_r<r>_seed<s>/run_results.json,
                  dev-selected per seed with the same rule as finetuning._run_sweep.

Writes <out_dir>/controls_table.md, controls_table.tex, controls_raw.csv.
Never fabricates: a missing file is a missing cell, listed at the top.

Usage:
  python scripts/aggregate_controls.py --repo $HOME/SpanOT-KD --dataset qed \
      --teacher /workspace/models/Llama-2-7b-chat-hf --student opt-350m --seeds 42 4 63
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics

import numpy as np

ARMS = ["spanotkd", "matchedweight", "randomspan", "vanilla", "mlotscaled"]
SWEEP_ARMS = ["spanotkd", "matchedweight", "randomspan"]
R_VALUES = [0.3, 0.5, 0.7]
PAIRS = [("spanotkd", "matchedweight"), ("spanotkd", "randomspan"),
         ("matchedweight", "randomspan"), ("spanotkd", "vanilla"), ("spanotkd", "mlotscaled")]
W_EFF_TOL = 0.05


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run_tag(student, teacher, method, seed):
    return f"{student.split('-')[0]}_{os.path.basename(teacher.rstrip('/'))}_{method}_seed{seed}"


def paired_stats(a, b, n_boot=10000, seed=0):
    """Paired differences d = a - b over seeds.

    Returns mean diff, percentile-bootstrap 95% CI of the mean (resampling seeds
    with replacement, fixed RNG seed), and Cohen's d_z = mean(d) / sd(d, ddof=1).
    With N <= 5 the percentile bootstrap is coarse and tends to under-cover.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else None, "ci_low": None, "ci_high": None, "d_z": None}
    if n >= 2:
        rng = np.random.default_rng(seed)
        boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)  # [n_boot]
        out["ci_low"], out["ci_high"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        sd = float(d.std(ddof=1))
        out["d_z"] = float(d.mean() / sd) if sd > 0 else None
    return out


def collect(args):
    ds = args.dataset
    test_key = "f1" if ds == "qed" else "rougeL"
    gen_key = "best_dev_f1" if ds == "qed" else "best_dev_rouge_l"
    gen_label = "dev_f1" if ds == "qed" else "dev_rouge_l"
    sfx = "" if args.ckpt == "gen" else "_loss"
    rows, missing = [], []

    for arm in ARMS:
        for seed in args.seeds:
            row = {"arm": arm, "seed": seed, "test": None, "selected_r": None, "w_eff_selected": None}
            if arm == "mlotscaled":
                cands = []
                for r in R_VALUES:
                    tag = run_tag(args.student, args.teacher, f"mlotscaled_r{r:.1f}", seed)
                    res = _load(os.path.join(args.repo, f"output_{ds}_{tag}", "run_results.json"))
                    key = gen_key if args.ckpt == "gen" else "best_dev_loss"
                    if res is None or res.get(key) is None:
                        missing.append(f"{arm} seed={seed} r={r}: run_results.json/{key}")
                        continue
                    row[f"dev_r{r:.1f}"] = res[key]
                    cands.append((r, float(res[key]), tag))
                if cands:
                    pick = (max if args.ckpt == "gen" else min)(cands, key=lambda x: x[1])
                    row["selected_r"] = pick[0]
                    tag = pick[2]
                else:
                    rows.append(row)
                    continue
            else:
                tag = run_tag(args.student, args.teacher, arm, seed)
            out_dir = os.path.join(args.repo, f"output_{ds}_{tag}")

            if arm in SWEEP_ARMS:
                summ = _load(os.path.join(out_dir, "sweep_summary.json"))
                if summ is None:
                    missing.append(f"{arm} seed={seed}: sweep_summary.json")
                else:
                    win = summ.get(f"winner_{gen_label}" if args.ckpt == "gen" else "winner_dev_loss", {})
                    row["selected_r"] = win.get("r")
                    for rk, pr in summ.get("per_run", {}).items():
                        row[f"dev_r{rk}"] = pr.get(gen_key if args.ckpt == "gen" else "best_dev_loss")
                for r in R_VALUES:
                    w = _load(os.path.join(out_dir, f"r_{r:.1f}", "w_eff.json"))
                    row[f"w_eff_r{r:.1f}"] = w.get("overall") if w else None
                    if w is None:
                        missing.append(f"{arm} seed={seed} r={r}: w_eff.json")
                if row["selected_r"] is not None:
                    row["w_eff_selected"] = row.get(f"w_eff_r{float(row['selected_r']):.1f}")

            ev = _load(os.path.join(args.repo, "eval_results", f"{ds}_{tag}{sfx}", "0shots.json"))
            if ev is None or ev.get(test_key) is None:
                missing.append(f"{arm} seed={seed}: eval_results/{ds}_{tag}{sfx}/0shots.json")
            else:
                row["test"] = float(ev[test_key])
            rows.append(row)
    return rows, missing, test_key


def fmt(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", required=True, choices=["qed", "fairytaleqa"])
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="default: every seed with a spanotkd output dir")
    p.add_argument("--ckpt", choices=["gen", "loss"], default="gen",
                   help="which canonical checkpoint's test eval to report (gen = best dev F1/ROUGE-L)")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    if args.seeds is None:
        pat = os.path.join(args.repo, f"output_{args.dataset}_{run_tag(args.student, args.teacher, 'spanotkd', 0)[:-1]}*")
        args.seeds = sorted({int(m.group(1)) for d in glob.glob(pat)
                             if (m := re.search(r"_seed(\d+)$", d))})
    rows, missing, test_key = collect(args)

    by_arm = {a: {r["seed"]: r for r in rows if r["arm"] == a} for a in ARMS}
    lines = []
    lines.append(f"# Controls: {args.dataset} / {os.path.basename(args.teacher.rstrip('/'))} -> {args.student} "
                 f"(test metric: {test_key} x100, checkpoint: {args.ckpt})\n")
    lines.append("## Caveats (read before using any number)\n")
    lines.append("1. `vanilla` has NO dev-selected r; the three span arms (and `mlotscaled`) are best-of-3 on dev. "
                 "Any span-arm vs vanilla gap includes this selection advantage.")
    lines.append("2. All arms must share seeds and the dev split (dev_split_seed = seed in the run scripts). "
                 f"Seeds requested: {args.seeds}.")
    lines.append("3. `w_eff` of `matchedweight` equals SpanOT-KD's rule applied to its OWN model's entropies, so the "
                 "two runs' w_eff can drift apart during training — check the w_eff table below.")
    lines.append("4. `mlotscaled` matches total weight with one global scalar; see make_scaled_commands.py docstring "
                 "for why this is an approximation.")
    lines.append(f"5. Missing cells: {len(missing)}")
    for m in missing:
        lines.append(f"   - {m}")
    lines.append("")

    lines.append("## Per-arm test metric (mean ± std over seeds)\n")
    lines.append("| arm | N | mean | std | per-seed |")
    lines.append("|---|---|---|---|---|")
    arm_vals = {}
    for a in ARMS:
        vals = [(s, by_arm[a][s]["test"]) for s in args.seeds if s in by_arm[a] and by_arm[a][s]["test"] is not None]
        arm_vals[a] = dict(vals)
        v = [x for _, x in vals]
        mean = statistics.mean(v) if v else None
        std = statistics.stdev(v) if len(v) > 1 else None
        lines.append(f"| {a} | {len(v)} | {fmt(mean)} | {fmt(std)} | "
                     + ", ".join(f"s{s}:{x:.2f}" for s, x in vals) + " |")
    lines.append("")

    lines.append("## Paired differences by seed (a − b)\n")
    lines.append("| a − b | N | mean diff | bootstrap 95% CI | Cohen's d_z |")
    lines.append("|---|---|---|---|---|")
    pair_rows = []
    for a, b in PAIRS:
        common = [s for s in args.seeds if s in arm_vals[a] and s in arm_vals[b]]
        st = paired_stats([arm_vals[a][s] for s in common], [arm_vals[b][s] for s in common])
        pair_rows.append((a, b, st))
        ci = "–" if st["ci_low"] is None else f"[{st['ci_low']:.2f}, {st['ci_high']:.2f}]"
        lines.append(f"| {a} − {b} | {st['n']} | {fmt(st['mean_diff'])} | {ci} | {fmt(st['d_z'])} |")
    min_n = min((st["n"] for _, _, st in pair_rows), default=0)
    if min_n < 3:
        lines.append("\n**N < 3 for at least one pair: no 'better' statement is supported; numbers only.**")
    lines.append("\nWith N ≤ 5 seeds the percentile bootstrap is coarse and under-covers; treat CIs as indicative.\n")

    lines.append("## Selected r per (arm, seed) and dev metric per r\n")
    lines.append("| arm | seed | selected r | dev r=0.3 | dev r=0.5 | dev r=0.7 |")
    lines.append("|---|---|---|---|---|---|")
    for a in SWEEP_ARMS + ["mlotscaled"]:
        for s in args.seeds:
            r = by_arm[a].get(s, {})
            lines.append(f"| {a} | {s} | {fmt(r.get('selected_r'))} | "
                         + " | ".join(fmt(r.get(f"dev_r{x:.1f}"), 4) for x in R_VALUES) + " |")
    lines.append("")

    lines.append(f"## w_eff_valid (overall) per (arm, r), mean over seeds; flag if |rel. diff| vs spanotkd > {W_EFF_TOL:.0%}\n")
    lines.append("| arm | r=0.3 | r=0.5 | r=0.7 |")
    lines.append("|---|---|---|---|")
    ref = {}
    for a in SWEEP_ARMS:
        cells = []
        for x in R_VALUES:
            v = [by_arm[a][s].get(f"w_eff_r{x:.1f}") for s in args.seeds if s in by_arm[a]]
            v = [y for y in v if y is not None]
            m = statistics.mean(v) if v else None
            if a == "spanotkd":
                ref[x] = m
            flag = ""
            if a != "spanotkd" and m is not None and ref.get(x):
                if abs(m - ref[x]) / ref[x] > W_EFF_TOL:
                    flag = " ⚠ NOT MATCHED"
            cells.append(fmt(m, 4) + flag)
        lines.append(f"| {a} | " + " | ".join(cells) + " |")

    os.makedirs(args.out_dir, exist_ok=True)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "controls_table.md"), "w", encoding="utf-8") as fh:
        fh.write(md)

    tex = ["\\begin{tabular}{lccc}", "\\toprule",
           f"Arm & $N$ & {test_key} (mean $\\pm$ std) & $\\Delta$ vs SpanOT-KD [95\\% CI] \\\\", "\\midrule"]
    for a in ARMS:
        v = list(arm_vals[a].values())
        ms = "--" if not v else (f"{statistics.mean(v):.2f}" + (f" $\\pm$ {statistics.stdev(v):.2f}" if len(v) > 1 else ""))
        delta = "--"
        for pa, pb, st in pair_rows:
            if pa == "spanotkd" and pb == a and st["ci_low"] is not None:
                delta = f"{-st['mean_diff']:.2f} [{-st['ci_high']:.2f}, {-st['ci_low']:.2f}]"
        tex.append(f"{a} & {len(v)} & {ms} & {delta} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(args.out_dir, "controls_table.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out_dir, "controls_raw.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(md)


if __name__ == "__main__":
    main()
