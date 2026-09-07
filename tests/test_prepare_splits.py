"""Carving val and test out of a single-split corpus, and the length histogram."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

MOLECULES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOCC", "CCCl", "CCBr",
             "c1ccncc1", "CC(C)C", "CCS", "CC#N", "CC=O", "CCCC", "COC"]


def test_percentiles_from_a_histogram_match_the_values():
    from prepare_data import _percentiles

    values = np.array([3, 3, 5, 8, 8, 8, 12, 40])
    histogram = np.bincount(values, minlength=64)
    out = _percentiles(histogram)
    assert out["count"] == len(values)
    assert out["max"] == 40
    assert abs(out["mean"] - values.mean()) < 0.01
    assert out["p50"] == 8


def test_an_empty_histogram_gives_nothing():
    from prepare_data import _percentiles

    assert _percentiles(np.zeros(10, dtype=np.int64)) == {}


def _run(tmp_path: Path, extra: list[str]) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "raw.txt"
    raw.write_text("\n".join(MOLECULES) + "\n")
    config = tmp_path / "prep.yaml"
    config.write_text(f"""
run_name: t
prepare:
  source:
    files: {{train: {raw}}}
    column: SMILES
    splits: {{train: train}}
  canonicalize: false
  deduplicate: true
  max_smiles_len: 120
  nprocs: 1
  out_dir: {tmp_path / 'corpus'}
  shard_size: 100
""")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
                    str(config), *extra], check=True, capture_output=True, text=True)
    return json.loads((tmp_path / "corpus" / "stats.json").read_text())


def test_without_carving_everything_lands_in_train(tmp_path):
    report = _run(tmp_path, [])
    counts = report["length_chars"]
    assert set(counts) == {"train"}
    assert counts["train"]["count"] == len(MOLECULES)


def test_carving_splits_the_corpus_and_keeps_it_disjoint(tmp_path):
    report = _run(tmp_path, ["prepare.carve.from=train", "prepare.carve.val=0.2",
                             "prepare.carve.test=0.2"])
    counts = {k: v["count"] for k, v in report["length_chars"].items()}
    assert sum(counts.values()) == len(MOLECULES)
    assert set(counts) <= {"train", "val", "test"}
    corpus = tmp_path / "corpus"
    seen = set()
    for split in counts:
        for shard in (corpus / split).glob("*.txt"):
            for line in shard.read_text().splitlines():
                assert line not in seen, f"{line} is in two splits"
                seen.add(line)


def test_carving_is_reproducible(tmp_path):
    a = _run(tmp_path / "a", ["prepare.carve.from=train", "prepare.carve.val=0.25"])
    b = _run(tmp_path / "b", ["prepare.carve.from=train", "prepare.carve.val=0.25"])
    assert ({k: v["count"] for k, v in a["length_chars"].items()}
            == {k: v["count"] for k, v in b["length_chars"].items()})


def test_carving_everything_away_is_refused(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _run(tmp_path, ["prepare.carve.from=train", "prepare.carve.val=0.6",
                        "prepare.carve.test=0.5"])
