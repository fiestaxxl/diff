#!/usr/bin/env python3
"""Cumulative benchmark ladder over the speed-up checklist.

Each rung adds one optimization on top of the previous one and is measured in a
fresh process (so compile caches and global flags do not leak), then the table
reports the per-rung delta and the cumulative speed-up:

    python scripts/bench_ladder.py configs/diffusion_chebi.yaml
    python scripts/bench_ladder.py configs/diffusion_chebi.yaml device_train_microbatch_size=64

Rungs that need CUDA are skipped with an n/a marker when running on CPU.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEP_RE = re.compile(r"median ([0-9.]+) ms")

# (label, extra overrides on top of the previous rung, needs_cuda)
LADDER = [
    ("0. baseline: fp32, no compile", ["precision=fp32", "tf32=false", "compile=false",
                                       "loss.autocast_scope=forward"], False),
    ("1. + tf32", ["tf32=true"], True),
    ("2. + bf16 autocast (forward only)", ["precision=amp_bf16"], True),
    ("2b. + autocast over readout and loss", ["loss.autocast_scope=loss"], True),
    ("3. + torch.compile (inductor)", ["compile=true", "compile_backend=inductor"], False),
    ("6. + fused AdamW", ["optimizer.fused=true"], True),
]


def run(config: str, overrides: list[str]) -> float | None:
    cmd = [sys.executable, str(HERE / "bench_step.py"), config, *overrides]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(proc.stdout[-1500:] + proc.stderr[-1500:])
        return None
    match = STEP_RE.search(proc.stdout)
    return float(match.group(1)) if match else None


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/bench_ladder.py <config.yaml> [key=value ...]")
    config, base_overrides = sys.argv[1], sys.argv[2:]
    has_cuda = torch.cuda.is_available()
    print(f"cuda available: {has_cuda}\n")

    active: list[str] = list(base_overrides)
    rows: list[tuple[str, float | None]] = []
    baseline: float | None = None
    previous: float | None = None

    for label, overrides, needs_cuda in LADDER:
        active = active + overrides
        if needs_cuda and not has_cuda:
            rows.append((label, None))
            continue
        median = run(config, active)
        rows.append((label, median))
        if median is None:
            continue
        if baseline is None:
            baseline = median
        step_delta = "" if previous is None else f"{(previous / median - 1) * 100:+.1f}%"
        cumulative = f"{baseline / median:.2f}x"
        print(f"{label:42s} {median:8.1f} ms  step {step_delta:>7s}  cumulative {cumulative:>6s}")
        previous = median

    print()
    for label, median in rows:
        if median is None:
            print(f"{label:42s}      n/a  (needs CUDA, rerun this script on the node)")


if __name__ == "__main__":
    main()
