"""Checkpoints: weights (as before, config.json + safetensors) plus trainer state.

The weight format is unchanged, so ``DiffusionTransformer.from_pretrained`` and
``SmilesAR.from_pretrained`` still read checkpoints the way they used to.
A ``trainer_state.pt`` file is stored next to them (optimizer, scaler, step,
epoch, RNG state); without it a run cannot be resumed correctly.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from dimol.training.distributed import unwrap_model

_CKPT_RE = re.compile(r"^ep(\d+)-ba(\d+)$")
STATE_NAME = "trainer_state.pt"


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    best_metric: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def checkpoint_dir(save_folder: str | Path, run_name: str) -> Path:
    return Path(save_folder) / run_name


def _ckpt_name(epoch: int, step: int) -> str:
    return f"ep{epoch}-ba{step}"


def list_checkpoints(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    items = [p for p in root.iterdir() if p.is_dir() and _CKPT_RE.match(p.name)]

    def key(p: Path) -> tuple[int, int]:
        m = _CKPT_RE.match(p.name)
        assert m
        return int(m.group(2)), int(m.group(1))

    return sorted(items, key=key)


def latest_checkpoint(root: str | Path) -> Optional[Path]:
    ckpts = list_checkpoints(root)
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    root: str | Path,
    model: torch.nn.Module,
    state: TrainerState,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    weights_only: bool = False,
    keep_last: int = 3,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    root = Path(root)
    out = root / _ckpt_name(state.epoch, state.step)
    out.mkdir(parents=True, exist_ok=True)

    raw = unwrap_model(model)
    raw.save_model(out)  # config.json + model.safetensors (unchanged format)

    if not weights_only:
        payload: Dict[str, Any] = {
            "state": state.to_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, out / STATE_NAME)

    (out / "meta.json").write_text(
        json.dumps({"step": state.step, "epoch": state.epoch}, indent=2)
    )
    _prune(root, keep_last)
    return out


def _prune(root: Path, keep_last: int) -> None:
    if keep_last is None or keep_last <= 0:
        return
    ckpts = list_checkpoints(root)
    for old in ckpts[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    weights_only: bool = False,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> TrainerState:
    """Load weights (and the trainer state, if present) from a checkpoint directory."""
    path = Path(path)
    raw = unwrap_model(model)

    # pick the weight format by file extension rather than by name
    candidates = [path / getattr(raw, "WEIGHTS_NAME", "model.safetensors"), path / "model.pt"]
    weights = next((c for c in candidates if c.exists()), None)
    if weights is None:
        raise FileNotFoundError(f"{path} contains neither {candidates[0].name} nor model.pt")

    if weights.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # safetensors checkpoint but the package is missing
            raise ImportError(
                f"{weights.name} requires safetensors: pip install safetensors"
            ) from exc

        sd = load_file(str(weights), device="cpu")
    else:
        sd = torch.load(weights, map_location="cpu")

    missing, unexpected = raw.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[ckpt] missing={list(missing)} unexpected={list(unexpected)}")

    state = TrainerState()
    state_path = path / STATE_NAME
    if not weights_only and state_path.exists():
        payload = torch.load(state_path, map_location=map_location, weights_only=False)
        st = payload.get("state", {})
        state = TrainerState(
            step=int(st.get("step", 0)),
            epoch=int(st.get("epoch", 0)),
            best_metric=st.get("best_metric"),
        )
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        rng = payload.get("rng") or {}
        if rng.get("cpu") is not None:
            torch.set_rng_state(rng["cpu"].cpu() if hasattr(rng["cpu"], "cpu") else rng["cpu"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:  # the number of GPUs may have changed
                print(f"[ckpt] could not restore the cuda RNG state: {exc}")
        print(f"[ckpt] resuming from ep{state.epoch}-ba{state.step} ({path})")
    else:
        print(f"[ckpt] loaded weights only from {path}")
    return state
