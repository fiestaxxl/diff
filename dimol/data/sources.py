"""Raw SMILES sources: a HuggingFace dataset or a local file.

The dataset name (``liupf/ChEBI-20-MM``) and the column name used to be hardcoded
in the scripts; they now live in the ``prepare.source`` config section.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def canonicalize(smi: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if the molecule does not parse."""
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None


def iter_smiles(
    source: Dict[str, Any],
    split: str,
    canonicalize_smiles: bool = True,
    **kwargs: Any,
) -> Iterator[str]:
    """Iterate over the SMILES of a single split.

    ``source`` accepts two shapes:
      * ``{hf_repo: liupf/ChEBI-20-MM, column: SMILES}`` for a HuggingFace dataset;
      * ``{files: {train: path/train.txt, ...}}`` for a text file, one molecule per
        line (or a csv/parquet file with a ``column`` column).
    """
    canonicalize_smiles = bool(kwargs.pop("canonicalize", canonicalize_smiles))
    if kwargs:
        raise TypeError(f"iter_smiles: unknown arguments {sorted(kwargs)}")

    column = source.get("column", "SMILES")
    files = source.get("files") or {}

    if files:
        path = files.get(split)
        if path is None:
            raise KeyError(f"prepare.source.files has no {split!r} split")
        raw_iter = _iter_path(Path(path), column)
    elif source.get("hf_repo"):
        raw_iter = _iter_hf(source["hf_repo"], split, column, source.get("hf_config"))
    else:
        raise KeyError("prepare.source: either hf_repo or files must be set")

    for smi in raw_iter:
        if not smi:
            continue
        out = canonicalize(smi) if canonicalize_smiles else smi
        if out:
            yield out


def _iter_hf(repo: str, split: str, column: str, config_name: Optional[str] = None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(repo, config_name) if config_name else load_dataset(repo)
    if split not in ds:
        raise KeyError(f"dataset {repo} has no {split!r} split; available: {list(ds)}")
    for ex in ds[split]:
        yield ex.get(column)


def _iter_path(path: Path, column: str) -> Iterator[str]:
    """A single file, or a directory of shards written by TextShardWriter."""
    if path.is_dir():
        shards = sorted(path.glob("shard_*.txt")) or sorted(path.glob("*.txt"))
        if not shards:
            raise FileNotFoundError(f"no shards in {path}")
        for shard in shards:
            yield from _iter_file(shard, column)
        return
    yield from _iter_file(path, column)


def _iter_file(path: Path, column: str) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".smi"):
        with path.open() as f:
            for line in f:
                yield line.strip()
    elif suffix == ".csv":
        import csv

        with path.open() as f:
            for row in csv.DictReader(f):
                yield row.get(column)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=[column])
        for value in table.column(column).to_pylist():
            yield value
    else:
        raise ValueError(f"do not know how to read {path} (supported: .txt/.smi/.csv/.parquet)")
