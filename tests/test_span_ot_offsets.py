"""Unit tests for train.span_ot's slow-tokenizer offset reconstruction.

Run: python -m unittest tests.test_span_ot_offsets -v   (from Multi-Level-OT/)

Context: Qwen-7B-Chat's tokenizer (QWenTokenizer) has no fast (Rust) backend,
so HF's exact `return_offsets_mapping=True` raises NotImplementedError.
_reconstruct_offset_map_slow (train/span_ot.py) reconstructs offsets instead
by decoding strictly-growing id prefixes. These tests hand-verify that
reconstruction against (a) a fully controlled synthetic tokenizer where the
expected offsets are computed by hand, and (b) a real HF slow tokenizer
cross-checked against its own fast counterpart's ground-truth offsets.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.span_ot import _reconstruct_offset_map_slow, _safe_offset_map


class _FakeSlowTokenizer:
    """Deterministic word-level stand-in for a slow BPE tokenizer.

    Vocab is fixed so expected offsets can be computed by hand instead of
    trusting a real BPE merge table.
    """

    _ID_TO_TOKEN = {0: "the", 1: " cat", 2: " sat"}
    _TEXT_TO_IDS = {"the cat sat": [0, 1, 2]}
    is_fast = False
    name_or_path = "fake-slow-tokenizer"

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        return list(self._TEXT_TO_IDS[text])

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(self._ID_TO_TOKEN[i] for i in ids)


class _FakeCorruptDecodeTokenizer:
    """Simulates a tokenizer whose full decode does not round-trip."""

    is_fast = False
    name_or_path = "fake-corrupt-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return [0, 1]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "GARBLED"  # never equals the input text


class TestReconstructOffsetMapHandComputed(unittest.TestCase):
    def test_three_token_sentence(self):
        text = "the cat sat"
        # "the" -> [0, 3); " cat" -> [3, 7) ("the"+" cat" = "the cat", 7 chars);
        # " sat" -> [7, 11) ("the cat" + " sat" = "the cat sat", 11 chars).
        expected = [(0, 3), (3, 7), (7, 11)]
        got = _reconstruct_offset_map_slow(text, _FakeSlowTokenizer())
        self.assertEqual(got, expected)
        # Every reconstructed span must be a substring match against the
        # original text at the claimed character positions.
        for s, e in got:
            self.assertEqual(text[s:e], "".join(
                _FakeSlowTokenizer._ID_TO_TOKEN[i]
                for i in _FakeSlowTokenizer._TEXT_TO_IDS[text]
            )[s:e])

    def test_empty_text_returns_empty(self):
        self.assertEqual(_reconstruct_offset_map_slow("", _FakeSlowTokenizer()), [])

    def test_round_trip_mismatch_returns_empty(self):
        # Documented contract (span_ot.py:61-65): an unusable offset map
        # must degrade to all-ones weights for that sample, never a wrong span.
        got = _reconstruct_offset_map_slow("anything", _FakeCorruptDecodeTokenizer())
        self.assertEqual(got, [])


class TestReconstructOffsetMapAgainstRealTokenizer(unittest.TestCase):
    """Cross-validate against a real slow/fast tokenizer pair (GPT2), which
    uses the same byte-level BPE family as Qwen's tiktoken-based tokenizer.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import GPT2Tokenizer, GPT2TokenizerFast
        except ImportError as e:
            raise unittest.SkipTest(f"transformers not available: {e}")
        try:
            cls.slow_tok = GPT2Tokenizer.from_pretrained("gpt2")
            cls.fast_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception as e:
            raise unittest.SkipTest(f"gpt2 tokenizer files not available: {e}")

    def _ground_truth_offsets(self, text):
        enc = self.fast_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if s != e]

    def test_ascii_sentence_matches_fast_ground_truth(self):
        text = "The quick brown fox jumps over the lazy dog."
        got = _reconstruct_offset_map_slow(text, self.slow_tok)
        expected = self._ground_truth_offsets(text)
        self.assertEqual(got, expected)

    def test_sentence_with_punctuation_and_apostrophe(self):
        text = "It's a well-known fact, isn't it?"
        got = _reconstruct_offset_map_slow(text, self.slow_tok)
        expected = self._ground_truth_offsets(text)
        self.assertEqual(got, expected)

    def test_safe_offset_map_routes_slow_tokenizer_through_reconstruction(self):
        text = "SpanOT-KD reweights positions."
        got = _safe_offset_map(text, self.slow_tok)
        expected = self._ground_truth_offsets(text)
        self.assertEqual(got, expected)

    def test_safe_offset_map_routes_fast_tokenizer_through_exact_path(self):
        text = "SpanOT-KD reweights positions."
        got = _safe_offset_map(text, self.fast_tok)
        expected = self._ground_truth_offsets(text)
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
