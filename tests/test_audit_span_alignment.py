"""WP6: audit_sample on stub tokenisers with hand-checkable span structure.

Run: python -m unittest discover -s tests   (from Multi-Level-OT/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import unittest

from _stubs import CharTokenizer, PairTokenizer
from audit_span_alignment import audit_sample, summarise


def labels(ids, prompt=2):
    return [-100] * prompt + list(ids)


class TestAuditSample(unittest.TestCase):
    def test_clean_alignment(self):
        # Student labels "abcdeX": size 6 - 1 (skip_student_eos) = 5 -> decoded
        # text "abcdeX" (6 chars), offsets clipped to 5 chars "abcde".
        # Teacher "ab|cd|eX": 3 tokens, all kept.
        s = labels([ord(c) for c in "abcdeX"])
        t = labels([1, 2, 3])
        rec = audit_sample(s, t, CharTokenizer(), PairTokenizer())
        self.assertIsNone(rec["fallback"])
        self.assertEqual((rec["s_size"], rec["t_size"]), (5, 3))
        # Student ends at 5, teacher at 6 -> end-offset mismatch is reported.
        self.assertTrue(rec["end_mismatch"])
        self.assertEqual(rec["mismatch_detail"]["student_end"], 5)
        self.assertEqual(rec["mismatch_detail"]["teacher_end"], 6)

    def test_empty_answer_is_no_offsets_fallback(self):
        rec = audit_sample(labels([]), labels([]), CharTokenizer(), PairTokenizer())
        self.assertEqual(rec["fallback"], "no_offsets")

    def test_summary_fractions(self):
        recs = [audit_sample(labels([ord(c) for c in "abcd!"]), labels([1, 2]), CharTokenizer(), PairTokenizer()),
                audit_sample(labels([]), labels([]), CharTokenizer(), PairTokenizer())]
        s = summarise(recs)
        self.assertEqual(s["n_samples"], 2)
        self.assertAlmostEqual(s["fallback_frac"]["no_offsets"], 0.5)
        self.assertIn("NOT COMPUTED", s["fallback_frac"]["no_positive_gap_spans"])
        # "abcd": 4 student chars, 2 teacher pairs, spans (0,2),(2,4): all aligned.
        self.assertAlmostEqual(s["student_token_unaligned_frac"], 0.0)
        self.assertAlmostEqual(s["teacher_token_unaligned_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
