"""Length bucketing and batch trimming."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dimol.data.collate import trim_collate
from dimol.data.datasets import SmilesDataset
from dimol.data.loaders import build_dataloader
from dimol.data.samplers import BucketBatchSampler
from dimol.training.distributed import DistEnv

RNG = np.random.default_rng(0)


def _dataset(tmp_path: Path, n: int = 256, stored_len: int = 208) -> SmilesDataset:
    """A split whose lengths look like a molecule corpus: mostly short, a few long."""
    lengths = np.clip(RNG.normal(40, 15, size=n).astype(int), 5, stored_len)
    tokens = np.zeros((n, stored_len), dtype=np.uint16)
    mask = np.zeros((n, stored_len), dtype=np.uint8)
    for i, real in enumerate(lengths):
        tokens[i, :real] = RNG.integers(1, 500, size=real)
        mask[i, :real] = 1
    np.save(tmp_path / "train_tokens.npy", tokens)
    np.save(tmp_path / "train_attn_mask.npy", mask)
    return SmilesDataset(tmp_path, "train")


def test_lengths_come_from_the_mask(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, n=16)
    assert ds.lengths.shape == (16,)
    for i in range(16):
        assert int(ds.lengths[i]) == int(ds[i]["attention_mask"].sum())


def test_batches_are_length_homogeneous() -> None:
    lengths = RNG.integers(5, 200, size=1024)
    sampler = BucketBatchSampler(lengths, batch_size=32, pool_factor=8)
    spreads = [int(lengths[b].max() - lengths[b].min()) for b in sampler]
    assert len(list(sampler)) == len(sampler) == 1024 // 32
    assert max(spreads) < 60, spreads          # random batches would span ~195
    assert np.median(spreads) < 40


def test_every_index_used_once_per_epoch() -> None:
    lengths = RNG.integers(5, 200, size=100)
    sampler = BucketBatchSampler(lengths, batch_size=10, pool_factor=3)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(100))


def test_epoch_changes_the_order() -> None:
    lengths = RNG.integers(5, 200, size=256)
    sampler = BucketBatchSampler(lengths, batch_size=16, pool_factor=4)
    first = [b for batch in sampler for b in batch]
    sampler.set_epoch(1)
    second = [b for batch in sampler for b in batch]
    assert first != second and sorted(first) == sorted(second)


def test_ranks_get_disjoint_slices() -> None:
    lengths = RNG.integers(5, 200, size=128)
    seen = []
    for rank in range(4):
        sampler = BucketBatchSampler(lengths, batch_size=8, pool_factor=2, rank=rank, world_size=4)
        seen.append({i for batch in sampler for i in batch})
    for a in range(4):
        for b in range(a + 1, 4):
            assert not seen[a] & seen[b], (a, b)
    assert len(set().union(*seen)) == 128


def test_trim_collate_cuts_to_the_longest_real_sequence() -> None:
    batch = []
    for real in (10, 12, 7):
        ids = torch.zeros(64, dtype=torch.long)
        ids[:real] = 1
        batch.append({"token_ids": ids, "attention_mask": ids != 0})
    out = trim_collate(batch, multiple_of=8)
    assert out["token_ids"].shape == (3, 16)   # ceil(12 / 8) * 8
    assert out["attention_mask"].shape == (3, 16)
    assert int(out["attention_mask"].sum()) == 29


def test_trim_collate_never_grows_or_loses_tokens() -> None:
    ids = torch.ones(8, dtype=torch.long)       # a full-length sequence
    out = trim_collate([{"token_ids": ids, "attention_mask": ids != 0}], multiple_of=8)
    assert out["token_ids"].shape == (1, 8)


def test_loader_with_bucketing_covers_the_split(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, n=256)
    env = DistEnv(ddp=False, device="cpu", device_type="cpu")
    cfg = {
        "dataset": {"name": "smiles_npy", "data_dir": str(tmp_path), "split": "train"},
        "num_workers": 0,
        "pin_memory": False,
        "bucket_by_length": True,
        "bucket_pool_factor": 4,
        "trim_to_multiple_of": 8,
    }
    loader, sampler = build_dataloader(cfg, device_batch_size=32, env=env, is_train=True)
    widths, rows = [], 0
    for batch in loader:
        widths.append(batch["token_ids"].shape[1])
        rows += batch["token_ids"].shape[0]
    assert rows == 256 and len(loader) == 8
    assert max(widths) <= 208 and min(widths) < 208     # at least some batch got trimmed
    assert hasattr(sampler, "set_epoch")


def test_bucketing_needs_lengths() -> None:
    env = DistEnv(ddp=False, device="cpu", device_type="cpu")
    cfg = {
        "dataset": {"name": "random", "max_len": 16, "num_samples": 32},
        "num_workers": 0,
        "pin_memory": False,
        "bucket_by_length": True,
    }
    with pytest.raises(TypeError):
        build_dataloader(cfg, device_batch_size=8, env=env, is_train=True)
