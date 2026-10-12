"""A preference batch must be trimmed by the longer of its two molecules."""
from __future__ import annotations

import torch

from dimol.data.collate import trim_collate


def _row(accepted_len: int, rejected_len: int, canvas: int = 64) -> dict:
    acc = torch.zeros(canvas, dtype=torch.long)
    rej = torch.zeros(canvas, dtype=torch.long)
    acc_mask = torch.zeros(canvas, dtype=torch.long)
    rej_mask = torch.zeros(canvas, dtype=torch.long)
    acc[:accepted_len] = 7
    rej[:rejected_len] = 9
    acc_mask[:accepted_len] = 1
    rej_mask[:rejected_len] = 1
    return {"token_ids": acc, "attention_mask": acc_mask,
            "rejected_token_ids": rej, "rejected_attention_mask": rej_mask}


def test_a_longer_rejected_molecule_is_not_truncated():
    batch = [_row(accepted_len=5, rejected_len=40)]
    out = trim_collate(batch)
    kept = int(out["rejected_attention_mask"].sum())
    assert kept == 40, f"the rejected molecule lost tokens: {kept} of 40"


def test_trimming_still_happens_when_both_are_short():
    batch = [_row(accepted_len=5, rejected_len=6)]
    out = trim_collate(batch)
    assert out["token_ids"].shape[1] < 64, "a short batch should still be trimmed"
    assert int(out["rejected_attention_mask"].sum()) == 6


def test_an_ordinary_batch_is_unaffected():
    canvas = 64
    mask = torch.zeros(canvas, dtype=torch.long)
    mask[:12] = 1
    out = trim_collate([{"token_ids": torch.ones(canvas, dtype=torch.long),
                         "attention_mask": mask}])
    assert out["token_ids"].shape[1] == 16  # 12 rounded up to a multiple of 8
