"""Collate functions.

``trim_collate`` shortens a batch to the longest real sequence it contains. It only
pays off together with length bucketing (``dimol/data/samplers.py``): with random
batches almost every batch contains one long molecule, so nothing gets trimmed.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data._utils.collate import default_collate


def trim_collate(
    batch: List[Dict[str, Any]], multiple_of: int = 8, min_len: int = 8
) -> Dict[str, torch.Tensor]:
    """Stack a batch and cut the time axis to its longest real sequence.

    The cut length is rounded up to ``multiple_of`` (tensor cores prefer multiples
    of 8) and never exceeds the stored length. Without an ``attention_mask`` in the
    batch there is nothing to trim by, so the batch is returned as collated.
    """
    out = default_collate(batch)
    mask = out.get("attention_mask")
    if mask is None:
        return out

    stored_len = int(mask.shape[1])
    real_len = int(mask.sum(dim=1).max())
    target = max(min_len, -(-real_len // multiple_of) * multiple_of)  # ceil to multiple
    target = min(target, stored_len)
    if target >= stored_len:
        return out
    return {k: (v[:, :target] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in out.items()}
