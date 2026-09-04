"""Smoke training run: config -> build -> a few steps on CPU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_smoke_train(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            str(ROOT / "configs/diffusion_smoke_cpu.yaml"),
            f"save_folder={tmp_path}",
            "autoresume=false",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    out = result.stdout
    # every step must report loss, accuracy and tokens-per-sec
    assert "[train ba 1" in out, out[-2000:]
    assert "loss " in out and "acc " in out and "tok/s " in out, out[-2000:]
    assert "training finished" in out


def test_cpu_smoke_resume(tmp_path: Path) -> None:
    """A second launch picks up the checkpoint instead of starting over."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/train.py"),
        str(ROOT / "configs/diffusion_smoke_cpu.yaml"),
        f"save_folder={tmp_path}",
        "autoresume=true",
        "max_duration=2ba",
        "save_interval=2ba",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "resuming from" in second.stdout
