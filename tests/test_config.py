"""Configs load, the schema catches typos, batch sizes are derived correctly."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from dimol.config import (
    Duration,
    duration_hit,
    duration_reached,
    load_config,
    update_batch_size_info,
)

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.run_name
    assert cfg.task in ("diffusion", "ar")
    assert str(cfg.precision) in ("amp_bf16", "amp_fp16", "fp32")


def test_cli_override(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS[0], ["seed=123", "optimizer.lr=0.5"])
    assert cfg.seed == 123
    assert cfg.optimizer.lr == 0.5


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("run_name: x\nnot_a_real_key: 1\n")
    with pytest.raises(Exception):
        load_config(bad)


def test_duration_parsing() -> None:
    assert Duration.parse("100ba").in_batches() == 100
    assert Duration.parse("2ep").in_batches(steps_per_epoch=50) == 100
    assert Duration.parse(7).in_batches() == 7
    with pytest.raises(ValueError):
        Duration.parse("10 epochs")


def test_duration_events() -> None:
    assert duration_hit("10ba", step=20, steps_per_epoch=None)
    assert not duration_hit("10ba", step=21, steps_per_epoch=None)
    assert not duration_hit("10ba", step=0, steps_per_epoch=None)  # step zero never counts
    assert duration_reached("5ba", step=5, epoch=0)
    assert duration_reached("2ep", step=1, epoch=2)
    assert not duration_reached("2ep", step=1000, epoch=1)


def test_batch_size_derivation() -> None:
    cfg = OmegaConf.create(
        {
            "global_train_batch_size": 512,
            "device_train_microbatch_size": 64,
            "device_eval_batch_size": None,
            "n_gpus": None,
            "device_train_batch_size": None,
            "device_train_grad_accum": None,
        }
    )
    update_batch_size_info(cfg, world_size=2)
    assert cfg.device_train_batch_size == 256
    assert cfg.device_train_grad_accum == 4
    assert cfg.device_eval_batch_size == 64


def test_batch_size_indivisible() -> None:
    cfg = OmegaConf.create(
        {
            "global_train_batch_size": 100,
            "device_train_microbatch_size": 64,
            "device_eval_batch_size": None,
            "n_gpus": None,
            "device_train_batch_size": None,
            "device_train_grad_accum": None,
        }
    )
    with pytest.raises(ValueError):
        update_batch_size_info(cfg, world_size=3)


def test_the_analysis_node_accepts_arbitrary_keys() -> None:
    """The analysis scripts put a checkpoint and a few knobs there; struct mode must allow it."""
    cfg = load_config(CONFIGS[0], ["analyze.checkpoint=runs/x/ep1-ba1", "analyze.points=7"])
    assert cfg.analyze["checkpoint"] == "runs/x/ep1-ba1"
    assert int(cfg.analyze["points"]) == 7
