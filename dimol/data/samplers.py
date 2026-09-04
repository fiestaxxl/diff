"""Length bucketing.

A denoiser step costs roughly linearly in the FFN and quadratically in attention
with respect to the sequence length, so batching molecules of similar length and
trimming the batch (see ``dimol/data/collate.py``) removes most of the padding
compute without dropping a single molecule.

Shape of one epoch:

    shuffle indices -> cut into pools of pool_factor * batch_size -> sort each pool
    by length -> cut the pool into batches -> shuffle the batch order

Sorting inside a pool rather than globally keeps randomness: batches are still
drawn from the whole corpus, they are just internally homogeneous in length.

IMPORTANT: with the current loss the padding mask is not applied (see
docs/architecture.md, section 3), so the loss averages over padded positions too.
Trimming therefore changes the ratio of real to padded positions inside a batch
and shifts the objective. Enable bucketing only together with a decision about
``loss.mask_padding``, and re-measure the baseline afterwards.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[List[int]]):
    """Yields lists of indices whose sequence lengths are close to each other.

    Also does what ``DistributedSampler`` does: with ``world_size > 1`` every rank
    receives a disjoint slice of the shuffled epoch, so a DataLoader can take this
    object as its ``batch_sampler``.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        batch_size: int,
        pool_factor: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        indices: Optional[Sequence[int]] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"bad rank/world_size: {rank}/{world_size}")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pool_size = max(self.batch_size, int(pool_factor) * self.batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(self.lengths), dtype=np.int64)
        )

    def set_epoch(self, epoch: int) -> None:
        """Same contract as DistributedSampler: reshuffles the epoch."""
        self.epoch = int(epoch)

    def _epoch_indices(self) -> np.ndarray:
        idx = self.indices
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            idx = rng.permutation(idx)
        if self.world_size > 1:
            usable = (len(idx) // self.world_size) * self.world_size
            idx = idx[:usable][self.rank :: self.world_size]
        return idx

    def _batches(self) -> List[np.ndarray]:
        idx = self._epoch_indices()
        batches: List[np.ndarray] = []
        for start in range(0, len(idx), self.pool_size):
            pool = idx[start : start + self.pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            for b in range(0, len(pool), self.batch_size):
                batch = pool[b : b + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch + 1).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches():
            yield [int(i) for i in batch]

    def __len__(self) -> int:
        per_rank = len(self.indices)
        if self.world_size > 1:
            per_rank = (per_rank // self.world_size)
        if self.drop_last:
            return per_rank // self.batch_size
        return -(-per_rank // self.batch_size)
