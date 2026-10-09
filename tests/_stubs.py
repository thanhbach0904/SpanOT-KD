"""Model-free stubs so DistillationLoss.forward (incl. the real
compute_batch_position_weights path) can run on tiny CPU tensors.

CharTokenizer: one token per character.  PairTokenizer: one token per two
characters. Together they give a real cross-tokenizer span structure
(each parent span = 2 chars = 2 student tokens = 1 teacher token).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.distillation_model import DistillationLoss


class CharTokenizer:
    is_fast = True
    name_or_path = "stub-char"

    def decode(self, ids, skip_special_tokens=True, **_):
        return "".join(chr(int(i)) for i in ids)

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class PairTokenizer(CharTokenizer):
    name_or_path = "stub-pair"

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"offset_mapping": [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]}


def make_loss(select_mode="entropy", span=True, top_r=0.5, low_delta=0.1):
    loss = DistillationLoss(
        distillation_weight=0.15, span_kd_enabled=span, span_top_r=top_r,
        span_low_delta=low_delta, span_select_mode=select_mode,
    )
    # Constructor only loads tokenizers from a path; inject the stubs.
    loss.student_tokenizer = CharTokenizer()
    loss.teacher_tokenizer = PairTokenizer()
    return loss


def make_batch(seed=0, B=2, prompt=3, answer="abcdef", V=60, requires_grad=True):
    """Returns (student_out, teacher_out, s_labels, t_labels, W).

    Student answer = len(answer) char tokens; teacher answer = ceil(len/2)
    pair tokens. Labels: `prompt` -100s then answer ids (only decoded /
    counted, never used as vocabulary indices). Student logits = W * feats so
    gradients w.r.t. the leaf W can be checked.
    """
    g = torch.Generator().manual_seed(seed)
    L = prompt + len(answer)
    W = torch.randn(V, generator=g).mul(0.5).requires_grad_(requires_grad)  # [V]
    feats = torch.randn(B, L, V, generator=g)  # [B, L, V]
    s_logits = feats * W  # [B, L, V]
    t_logits = torch.randn(B, L, V, generator=g) * 2  # [B, L, V]

    s_ids = [ord(c) for c in answer]
    t_ids = s_ids[: (len(answer) + 1) // 2]
    s_labels = torch.full((B, L), -100, dtype=torch.long)
    t_labels = torch.full((B, L), -100, dtype=torch.long)
    s_labels[:, prompt:] = torch.tensor(s_ids)
    t_labels[:, prompt:prompt + len(t_ids)] = torch.tensor(t_ids)
    t_labels[:, prompt + len(t_ids):] = -100

    s_out = SimpleNamespace(logits=s_logits, loss=torch.tensor(0.0))
    t_out = SimpleNamespace(logits=t_logits, loss=torch.tensor(0.0))
    return s_out, t_out, s_labels, t_labels, W
