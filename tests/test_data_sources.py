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
