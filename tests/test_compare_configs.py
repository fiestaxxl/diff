"""The seed-aware comparison: grouping, averaging and the verdict rule."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metrics(tmp_path: Path, runs: dict) -> Path:
    payload = {
        name: {"validity": v / 100.0, "uniqueness": 1.0, "n": 10000}
        for name, v in runs.items()
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))
    return path


def _run(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_configs.py"), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def test_seeds_of_one_config_are_grouped_and_averaged(tmp_path):
    path = _metrics(tmp_path, {"cfg_s42": 10.0, "cfg_s43": 20.0, "ref_s42": 5.0})
    out = _run(str(path))
    line = next(l for l in out.splitlines() if l.startswith("cfg"))
    assert " 2" in line and "15.00%" in line and "10.00-20.00" in line


def test_a_single_seed_is_reported_as_unresolved(tmp_path):
    path = _metrics(tmp_path, {"cfg_s42": 30.0, "ref_s42": 5.0, "ref_s43": 5.5})
    out = _run(str(path), "--reference", "ref")
    assert "one seed, unresolved" in out


def test_an_effect_is_called_only_when_the_ranges_do_not_touch(tmp_path):
    path = _metrics(tmp_path, {
        "good_s42": 30.0, "good_s43": 28.0,
        "noisy_s42": 30.0, "noisy_s43": 4.0,
        "ref_s42": 5.0, "ref_s43": 6.0,
    })
    out = _run(str(path), "--reference", "ref")
    good = next(l for l in out.splitlines() if l.startswith("good"))
    noisy = next(l for l in out.splitlines() if l.startswith("noisy"))
    assert "clears the reference" in good
    assert "inside the spread" in noisy


def test_a_missing_reference_is_reported(tmp_path):
    path = _metrics(tmp_path, {"cfg_s42": 10.0})
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_configs.py"),
         str(path), "--reference", "nope"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "not in these files" in proc.stderr


def test_a_run_listed_twice_is_counted_once(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    payload = {"cfg_s42": {"validity": 0.10, "uniqueness": 1.0}}
    (first / "metrics.json").write_text(json.dumps(payload))
    (second / "metrics.json").write_text(
        json.dumps({"cfg_s42": {"validity": 0.30, "uniqueness": 1.0}})
    )
    out = _run(str(first / "metrics.json"), str(second / "metrics.json"))
    line = next(l for l in out.splitlines() if l.startswith("cfg"))
    assert " 1" in line and "10.00%" in line
    assert "counted once" in out
