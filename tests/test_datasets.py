"""SmilesDataset: single-file and sharded .npy layouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dimol.data.datasets import SmilesDataset


def _write(path: Path, name: str, rows: int, length: int, fill: int) -> None:
    np.save(path / f"{name}_tokens.npy", np.full((rows, length), fill, dtype=np.uint16))
    np.save(path / f"{name}_attn_mask.npy", np.ones((rows, length), dtype=np.uint8))


def test_single_file_layout(tmp_path: Path) -> None:
    _write(tmp_path, "val", rows=5, length=8, fill=3)
    ds = SmilesDataset(tmp_path, "val")
    assert len(ds) == 5 and ds.num_shards == 1 and ds.max_len == 8
    item = ds[4]
    assert item["token_ids"].tolist() == [3] * 8
    assert item["token_ids"].dtype.is_floating_point is False
    assert item["attention_mask"].dtype is not None and bool(item["attention_mask"].all())


def test_sharded_layout_is_concatenated(tmp_path: Path) -> None:
    """Every shard must be visible through one dataset, otherwise a DataLoader
    built once would only ever see the first shard."""
    for i in range(3):
        np.save(tmp_path / f"train_tokens_{i:05d}.npy", np.full((2, 4), i, dtype=np.uint16))
        np.save(tmp_path / f"train_attn_mask_{i:05d}.npy", np.ones((2, 4), dtype=np.uint8))

    ds = SmilesDataset(tmp_path, "train")
    assert len(ds) == 6 and ds.num_shards == 3 and ds.shard_sizes == [2, 2, 2]
    # global index -> (shard, row): shard id is stored as the token value
    assert [ds[i]["token_ids"][0].item() for i in range(6)] == [0, 0, 1, 1, 2, 2]
    assert ds[-1]["token_ids"][0].item() == 2


def test_single_file_wins_over_shards(tmp_path: Path) -> None:
    _write(tmp_path, "train", rows=4, length=4, fill=9)
    np.save(tmp_path / "train_tokens_00000.npy", np.zeros((7, 4), dtype=np.uint16))
    np.save(tmp_path / "train_attn_mask_00000.npy", np.ones((7, 4), dtype=np.uint8))
    ds = SmilesDataset(tmp_path, "train")
    assert len(ds) == 4 and ds.num_shards == 1


def test_mmap_off(tmp_path: Path) -> None:
    _write(tmp_path, "test", rows=3, length=4, fill=1)
    ds = SmilesDataset(tmp_path, "test", mmap=False)
    assert len(ds) == 3 and not isinstance(ds.tokens[0], np.memmap)


def test_missing_split_lists_what_is_there(tmp_path: Path) -> None:
    _write(tmp_path, "train", rows=2, length=4, fill=1)
    with pytest.raises(FileNotFoundError) as exc:
        SmilesDataset(tmp_path, "validation")
    assert "train_tokens.npy" in str(exc.value)


def test_shape_mismatch(tmp_path: Path) -> None:
    np.save(tmp_path / "train_tokens.npy", np.zeros((3, 4), dtype=np.uint16))
    np.save(tmp_path / "train_attn_mask.npy", np.ones((3, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        SmilesDataset(tmp_path, "train")


def test_inconsistent_shard_length(tmp_path: Path) -> None:
    np.save(tmp_path / "train_tokens_00000.npy", np.zeros((2, 4), dtype=np.uint16))
    np.save(tmp_path / "train_attn_mask_00000.npy", np.ones((2, 4), dtype=np.uint8))
    np.save(tmp_path / "train_tokens_00001.npy", np.zeros((2, 6), dtype=np.uint16))
    np.save(tmp_path / "train_attn_mask_00001.npy", np.ones((2, 6), dtype=np.uint8))
    with pytest.raises(ValueError) as exc:
        SmilesDataset(tmp_path, "train")
    assert "same max_length" in str(exc.value)


def test_index_out_of_range(tmp_path: Path) -> None:
    _write(tmp_path, "train", rows=2, length=4, fill=1)
    ds = SmilesDataset(tmp_path, "train")
    with pytest.raises(IndexError):
        ds[2]


def test_dataloader_covers_all_shards(tmp_path: Path) -> None:
    """One epoch of a DataLoader built once must yield every shard."""
    from torch.utils.data import DataLoader

    for i in range(4):
        np.save(tmp_path / f"train_tokens_{i:05d}.npy", np.full((3, 4), i, dtype=np.uint16))
        np.save(tmp_path / f"train_attn_mask_{i:05d}.npy", np.ones((3, 4), dtype=np.uint8))

    ds = SmilesDataset(tmp_path, "train")
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    seen = [int(v) for batch in loader for v in batch["token_ids"][:, 0]]
    assert sorted(seen) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    # the second epoch starts over the same data
    seen_again = [int(v) for batch in loader for v in batch["token_ids"][:, 0]]
    assert sorted(seen_again) == sorted(seen)


def test_a_second_tokenization_pass_can_share_the_directory(tmp_path):
    """A corpus arriving in batches is tokenized in passes into one directory."""
    from dimol.data.shards import ArrayShardWriter

    first = ArrayShardWriter(tmp_path, "train", max_length=4, shard_size=2)
    for _ in range(4):
        first.write([1, 2, 3, 0], [1, 1, 1, 0])
    written = first.close()
    assert [p.name for p in written if "tokens" in p.name] == [
        "train_tokens_00000.npy", "train_tokens_00001.npy"]

    second = ArrayShardWriter(tmp_path, "train", max_length=4, shard_size=2,
                              shard_offset=2)
    for _ in range(2):
        second.write([4, 5, 0, 0], [1, 1, 0, 0])
    more = second.close()
    assert [p.name for p in more if "tokens" in p.name] == ["train_tokens_00002.npy"]
    assert len(sorted(tmp_path.glob("train_tokens_*.npy"))) == 3


def test_a_second_pass_without_an_offset_is_refused(tmp_path):
    from dimol.data.shards import ArrayShardWriter

    first = ArrayShardWriter(tmp_path, "train", max_length=4, shard_size=2)
    first.write([1, 2, 3, 0], [1, 1, 1, 0])
    first.close()

    second = ArrayShardWriter(tmp_path, "train", max_length=4, shard_size=2)
    second.write([4, 5, 0, 0], [1, 1, 0, 0])
    with pytest.raises(FileExistsError, match="shard_offset"):
        second.close()
