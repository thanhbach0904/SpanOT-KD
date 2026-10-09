"""WP10: aggregate_controls on a synthetic results tree with hand-computed stats.

Run: python -m unittest discover -s tests   (from Multi-Level-OT/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import aggregate_controls as agg

TEACHER, STUDENT, SEEDS = "/m/Llama-2-7b-chat-hf", "opt-350m", [1, 2, 3]
# QED test F1 (x100) per arm per seed. spanotkd - matched diffs = [1, 2, 3]
# -> mean 2, sd 1, d_z = 2.
F1 = {"spanotkd": [40.0, 42.0, 44.0], "matchedweight": [39.0, 40.0, 41.0],
      "randomspan": [38.0, 38.0, 38.0], "vanilla": [37.0, 37.5, 38.0]}


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def build_tree(root):
    for arm, vals in F1.items():
        for seed, v in zip(SEEDS, vals):
            tag = agg.run_tag(STUDENT, TEACHER, arm, seed)
            _w(os.path.join(root, "eval_results", f"qed_{tag}", "0shots.json"), {"f1": v})
            if arm != "vanilla":
                out = os.path.join(root, f"output_qed_{tag}")
                _w(os.path.join(out, "sweep_summary.json"), {
                    "per_run": {f"{r:.1f}": {"best_dev_f1": 0.3 + r / 10} for r in agg.R_VALUES},
                    "winner_dev_f1": {"r": 0.7}})
                for r in agg.R_VALUES:
                    # randomspan w_eff 20% off spanotkd -> must be flagged.
                    w = 0.5 if arm != "randomspan" else 0.6
                    _w(os.path.join(out, f"r_{r:.1f}", "w_eff.json"), {"overall": w})


class TestPairedStats(unittest.TestCase):
    def test_hand_computed(self):
        st = agg.paired_stats([40, 42, 44], [39, 40, 41])
        self.assertEqual(st["n"], 3)
        self.assertAlmostEqual(st["mean_diff"], 2.0)
        self.assertAlmostEqual(st["d_z"], 2.0)
        self.assertTrue(1.0 <= st["ci_low"] <= 2.0 <= st["ci_high"] <= 3.0)

    def test_constant_diff_has_no_d_z(self):
        st = agg.paired_stats([2, 3, 4], [1, 2, 3])
        self.assertIsNone(st["d_z"])
        self.assertAlmostEqual(st["ci_low"], 1.0)
        self.assertAlmostEqual(st["ci_high"], 1.0)

    def test_n1_gives_no_ci(self):
        st = agg.paired_stats([1.0], [0.0])
        self.assertIsNone(st["ci_low"])
        self.assertIsNone(st["d_z"])


class TestEndToEnd(unittest.TestCase):
    def test_tables(self):
        with tempfile.TemporaryDirectory() as root:
            build_tree(root)
            out = os.path.join(root, "results")
            sys_argv = sys.argv
            sys.argv = ["agg", "--repo", root, "--dataset", "qed", "--teacher", TEACHER,
                        "--student", STUDENT, "--out_dir", out]  # seeds auto-discovered
            try:
                agg.main()
            finally:
                sys.argv = sys_argv
            md = Path(out, "controls_table.md").read_text(encoding="utf-8")
            self.assertIn("| spanotkd − matchedweight | 3 | 2.00 |", md)
            self.assertIn("| spanotkd | 3 | 42.00 | 2.00 |", md)
            self.assertIn("NOT MATCHED", md.split("randomspan |")[-1])  # w_eff flag on randomspan row
            self.assertNotIn("NOT MATCHED", md.split("| matchedweight |")[-1].split("\n")[0])
            # mlotscaled was never run -> reported missing, not invented.
            self.assertIn("| mlotscaled | 0 | – |", md)
            self.assertTrue(Path(out, "controls_table.tex").exists())
            self.assertTrue(Path(out, "controls_raw.csv").exists())


if __name__ == "__main__":
    unittest.main()
