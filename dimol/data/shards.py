"""Writing a corpus as shards.

The training-time reader (``dimol.data.datasets.SmilesDataset``) already understands a
sharded layout and memory-maps it; these writers produce it without ever holding a
whole split in memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


class TextShardWriter:
    """Writes molecules into ``<out_dir>/shard_XXXXX.txt``, one per line."""

    def __init__(self, out_dir: str | Path, shard_size: Optional[int] = None, prefix: str = "shard"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix
        self.paths: List[Path] = []
        self.count = 0
        self._in_shard = 0
        self._handle = None

    def _open_next(self) -> None:
        path = self.out_dir / f"{self.prefix}_{len(self.paths):05d}.txt"
        self.paths.append(path)
        self._handle = path.open("w")
        self._in_shard = 0

    def write(self, line: str) -> None:
        if self._handle is None:
            self._open_next()
        assert self._handle is not None
        self._handle.write(line + "\n")
        self.count += 1
        self._in_shard += 1
        if self.shard_size and self._in_shard >= self.shard_size:
            self._handle.close()
            self._handle = None

    def close(self) -> List[Path]:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return self.paths

    def __enter__(self) -> "TextShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ArrayShardWriter:
    """Writes (tokens, mask) rows into ``<out_dir>/<split>_{tokens,attn_mask}_XXXXX.npy``.

    Rows are buffered per shard, so peak memory is ``shard_size * max_length * 3`` bytes
    (uint16 tokens plus uint8 mask), independent of the corpus size.

    ``shard_offset`` shifts the numbering, so a corpus tokenized in several passes can
    share one output directory. Overwriting is refused rather than silently allowed.
    """

    def __init__(
        self,
        out_dir: str | Path,
        split: str,
        max_length: int,
        shard_size: int = 250_000,
        token_dtype=np.uint16,
        shard_offset: int = 0,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_length = int(max_length)
        self.shard_size = int(shard_size)
        self.token_dtype = token_dtype
        # Where the shard numbering starts. A corpus that arrives in batches is
        # tokenized in several passes, and the passes have to land in one directory
        # without overwriting each other, because the dataset reads every
        # <split>_tokens_*.npy it finds.
        self.shard_offset = int(shard_offset)
        self.paths: List[Path] = []
        self.count = 0
        self._tokens = np.zeros((self.shard_size, self.max_length), dtype=token_dtype)
        self._mask = np.zeros((self.shard_size, self.max_length), dtype=np.uint8)
        self._filled = 0

    def write(self, tokens: Iterable[int], mask: Iterable[int]) -> None:
        self._tokens[self._filled] = np.asarray(tokens, dtype=self.token_dtype)
        self._mask[self._filled] = np.asarray(mask, dtype=np.uint8)
        self._filled += 1
        self.count += 1
        if self._filled >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self._filled == 0:
            return
        index = self.shard_offset + len(self.paths) // 2
        tokens_path = self.out_dir / f"{self.split}_tokens_{index:05d}.npy"
        mask_path = self.out_dir / f"{self.split}_attn_mask_{index:05d}.npy"
        for path in (tokens_path, mask_path):
            if path.exists():
                raise FileExistsError(
                    f"{path} is already there; a second tokenization pass into the same "
                    "directory needs shard_offset past the shards already written"
                )
        np.save(tokens_path, self._tokens[: self._filled])
        np.save(mask_path, self._mask[: self._filled])
        self.paths.extend([tokens_path, mask_path])
        self._filled = 0

    def close(self) -> List[Path]:
        self.flush()
        return self.paths

    def __enter__(self) -> "ArrayShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
