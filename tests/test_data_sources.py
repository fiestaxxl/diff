"""SMILES sources: reading files, selecting a split, clear errors."""
from __future__ import annotations

from pathlib import Path

import pytest

from dimol.data.sources import iter_smiles


def _source(tmp_path: Path) -> dict:
    (tmp_path / "train.txt").write_text("CCO\nc1ccccc1\n\nCC(=O)O\n")
    (tmp_path / "val.csv").write_text("id,SMILES\n1,CCN\n2,CCC\n")
    return {
        "column": "SMILES",
        "files": {"train": str(tmp_path / "train.txt"), "val": str(tmp_path / "val.csv")},
    }


def test_txt_split(tmp_path: Path) -> None:
    out = list(iter_smiles(_source(tmp_path), "train", canonicalize=False))
    assert out == ["CCO", "c1ccccc1", "CC(=O)O"]  # blank lines are dropped


def test_csv_split(tmp_path: Path) -> None:
    out = list(iter_smiles(_source(tmp_path), "val", canonicalize=False))
    assert out == ["CCN", "CCC"]


def test_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        list(iter_smiles(_source(tmp_path), "test", canonicalize=False))


def test_empty_source() -> None:
    with pytest.raises(KeyError):
        list(iter_smiles({}, "train", canonicalize=False))


def test_unknown_kwarg(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        list(iter_smiles(_source(tmp_path), "train", canonicalise=False))


def test_a_parquet_file_is_read_in_batches_not_loaded(tmp_path):
    """A ZINC-20 shard is nine million rows; read_table would hold the column in memory."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"SMILES": ["CCO", "c1ccccc1", "CC(=O)O"]}), path)
    out = list(iter_smiles({"files": {"train": str(path)}}, "train", canonicalize=False))
    assert out == ["CCO", "c1ccccc1", "CC(=O)O"]


def test_a_directory_of_parquet_shards_is_read_in_order(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    shards = tmp_path / "data"
    shards.mkdir()
    pq.write_table(pa.table({"SMILES": ["CCO"]}), shards / "train-00000.parquet")
    pq.write_table(pa.table({"SMILES": ["CCN"]}), shards / "train-00001.parquet")
    out = list(iter_smiles({"files": {"train": str(shards)}}, "train", canonicalize=False))
    assert out == ["CCO", "CCN"]


def test_text_shards_still_win_over_parquet_in_a_mixed_directory(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "shard_00000.txt").write_text("CCO\n")
    pq.write_table(pa.table({"SMILES": ["CCN"]}), mixed / "extra.parquet")
    out = list(iter_smiles({"files": {"train": str(mixed)}}, "train", canonicalize=False))
    assert out == ["CCO"]
