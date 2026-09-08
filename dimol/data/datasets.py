"""Datasets over tokenized SMILES stored as .npy arrays.

Moved from dimol/datasets/data.py with the commented-out code removed; the only
functional addition is support for a sharded layout (see SmilesDataset).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dimol import registry


class SimpleDataset(Dataset):
    """Random token ids. Used by the CPU smoke config and by tests."""

    def __init__(self, max_len: int, num_samples: int = 100):
        self.max_len = max_len
        self.tokens = torch.randint(0, 1000, (num_samples, max_len)).long()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        results = {'token_ids': self.tokens[idx]}
        return results


class SmilesDataset(Dataset):
    """A tokenized split stored as .npy arrays.

    Layout written by scripts/tokenize_dataset.py (one file pair per split):

        <data_dir>/<split>_tokens.npy       (N, T) uint16
        <data_dir>/<split>_attn_mask.npy    (N, T) uint8

    A sharded layout is also accepted, for corpora that do not fit into one file:

        <data_dir>/<split>_tokens_00000.npy, <split>_tokens_00001.npy, ...
        <data_dir>/<split>_attn_mask_00000.npy, <split>_attn_mask_00001.npy, ...

    Shards are sorted by file name and concatenated logically, so index i refers
    to the i-th molecule of the whole split and a single DataLoader iterates every
    shard. An epoch therefore ends only after the last shard, and the training
    loop starts the next epoch over the same (reshuffled) data.

    With ``mmap=True`` the arrays are memory-mapped instead of being read into
    RAM, so a corpus larger than memory still works; every returned row is copied
    into a fresh tensor, so nothing keeps a reference to the mapping.
    """

    def __init__(self, data_dir: str | Path, split: str, mmap: bool = True):
        data_dir = Path(data_dir)
        token_paths = self._find_shards(data_dir, split, "tokens")
        mask_paths = self._find_shards(data_dir, split, "attn_mask")

        if not token_paths:
            raise FileNotFoundError(
                f"No token arrays for split {split!r} in {data_dir}. Expected "
                f"{split}_tokens.npy or {split}_tokens_*.npy. "
                f"Present: {sorted(p.name for p in data_dir.glob('*.npy'))}"
            )
        if len(mask_paths) != len(token_paths):
            raise FileNotFoundError(
                f"Split {split!r}: {len(token_paths)} token shard(s) but "
                f"{len(mask_paths)} mask shard(s) in {data_dir}"
            )

        mmap_mode: Optional[str] = "r" if mmap else None
        self.tokens: List[np.ndarray] = [np.load(p, mmap_mode=mmap_mode) for p in token_paths]
        self.attn_mask: List[np.ndarray] = [np.load(p, mmap_mode=mmap_mode) for p in mask_paths]
        self.shard_paths = token_paths

        self.max_len = int(self.tokens[0].shape[1])
        sizes = []
        for path, tok, mask in zip(token_paths, self.tokens, self.attn_mask):
            if tok.shape != mask.shape:
                raise ValueError(f"{path.name}: token shape {tok.shape} != mask shape {mask.shape}")
            if int(tok.shape[1]) != self.max_len:
                raise ValueError(
                    f"{path.name}: sequence length {tok.shape[1]} differs from the first "
                    f"shard ({self.max_len}); shards must be tokenized with the same max_length"
                )
            sizes.append(int(tok.shape[0]))

        # start index of every shard, so a global index can be split into (shard, row)
        self.shard_sizes = sizes
        self._starts = np.cumsum([0] + sizes)
        self.num_samples = int(self._starts[-1])
        self._lengths: Optional[np.ndarray] = None

    @staticmethod
    def _find_shards(data_dir: Path, split: str, kind: str) -> List[Path]:
        single = data_dir / f"{split}_{kind}.npy"
        if single.exists():
            return [single]
        return sorted(data_dir.glob(f"{split}_{kind}_*.npy"))

    @property
    def num_shards(self) -> int:
        return len(self.tokens)

    @property
    def lengths(self) -> np.ndarray:
        """Real (unpadded) length of every molecule, from the attention masks.

        Computed once with a single pass over the mask arrays and cached in memory;
        length bucketing needs it. For a corpus where even that pass is expensive,
        cache it next to the data as a .npy file and load it here.
        """
        if self._lengths is None:
            self._lengths = np.concatenate(
                [np.asarray(mask).sum(axis=1, dtype=np.int32) for mask in self.attn_mask]
            )
        return self._lengths

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self.num_samples
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} molecules")
        shard = int(np.searchsorted(self._starts, idx, side="right")) - 1
        row = idx - int(self._starts[shard])
        return {
            "token_ids": torch.from_numpy(self.tokens[shard][row].astype(np.int64)),
            "attention_mask": torch.from_numpy(self.attn_mask[shard][row].astype(bool)),
        }


class PairedSmilesTextDataset(SmilesDataset):
    """A tokenized split plus the frozen caption encoding that goes with each molecule.

    Adds to the layout above, written by scripts/prepare_paired.py:

        <data_dir>/<split>_text.npy         (N, S, D) float16
        <data_dir>/<split>_text_mask.npy    (N, S)    uint8

    Row i of every array is the same pair, which is why the paired preparation refuses to
    deduplicate or reshuffle: a caption separated from its molecule is worthless, and a
    benchmark's splits are part of the benchmark.

    The caption states are memory-mapped like the molecules. They are large, 5 GB for
    ChEBI-20's training split at 128 tokens of SciBERT, and they never change, because the
    encoder is frozen and the encoding was done once.
    """

    def __init__(self, data_dir: str | Path, split: str, mmap: bool = True):
        super().__init__(data_dir, split, mmap=mmap)
        directory = Path(data_dir)
        text_path = directory / f"{split}_text.npy"
        mask_path = directory / f"{split}_text_mask.npy"
        for path in (text_path, mask_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing; run scripts/prepare_paired.py --stage text"
                )
        how = "r" if mmap else None
        self.text = np.load(text_path, mmap_mode=how)
        self.text_mask = np.load(mask_path, mmap_mode=how)
        if len(self.text) != self.num_samples:
            raise ValueError(
                f"split {split!r}: {self.num_samples} molecules but {len(self.text)} "
                "captions; the arrays are not aligned"
            )

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if idx < 0:
            idx += self.num_samples
        item["text"] = torch.from_numpy(np.asarray(self.text[idx], dtype=np.float32))
        item["text_mask"] = torch.from_numpy(np.asarray(self.text_mask[idx], dtype=bool))
        return item


registry.datasets.register("smiles_npy")(SmilesDataset)
registry.datasets.register("smiles_text_npy")(PairedSmilesTextDataset)
registry.datasets.register("random")(SimpleDataset)
