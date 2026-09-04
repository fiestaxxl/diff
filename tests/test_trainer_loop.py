"""Training loop: step counting, grad accum, intervals, checkpoints, resume.

The tests run on a toy model and task: they check loop control, not the diffusion
math (that is covered by test_tasks.py / test_diffusion_math.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dimol.training.distributed import DistEnv
from dimol.training.logging import Logger, MultiLogger
from dimol.training.optim import WarmupCosine
from dimol.training.trainer import Trainer


class ToyModel(nn.Module):
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "model.pt"

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.saves = 0

    def save_model(self, out) -> None:
        self.saves += 1
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / self.WEIGHTS_NAME)
        (out / self.CONFIG_NAME).write_text("{}")


class ToyTask:
    seq_len = 4

    def __init__(self) -> None:
        self.calls = 0
        self.steps_seen: List[int] = []

    def compute_loss(self, model, batch, step: int = 0) -> Dict[str, torch.Tensor]:
        self.calls += 1
        self.steps_seen.append(step)
        raw = model.module if hasattr(model, "module") else model
        loss = raw.w.sum() * batch["token_ids"].float().mean()
        return {"loss": loss, "token_acc": torch.tensor(0.5)}


class Capture(Logger):
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.metrics: List[tuple] = []

    def log_line(self, line: str) -> None:
        self.lines.append(line)

    def log_metrics(self, metrics, step: int) -> None:
        self.metrics.append((step, metrics))


def make_cfg(**over) -> DictConfig:
    base = {
        "run_name": "toy", "task": "diffusion", "seed": 0,
        "max_duration": "4ba", "global_train_batch_size": 4,
        "device_train_microbatch_size": 2, "device_train_grad_accum": 2,
        "device_eval_batch_size": 2, "eval_interval": "2ba",
        "eval_subset_num_batches": -1, "eval_before_train": "skip",
        "precision": "fp32", "sync_each_step": False,
        "log_to_console": True, "console_log_interval": "1ba",
        "save_folder": None, "save_interval": "100ep", "save_last": False,
        "save_num_checkpoints_to_keep": 2, "save_weights_only": False,
        "autoresume": False, "load_path": None, "load_weights_only": False,
        "algorithms": {"gradient_clipping": {"clipping_type": "norm", "clipping_threshold": 1.0}},
        "sampling": {"enabled": False, "interval": "2ba", "log_examples": 2},
    }
    base.update(over)
    return OmegaConf.create(base)


def build(cfg: DictConfig, n_micro: int = 4, n_eval: int = 2, model: Optional[ToyModel] = None):
    model = model or ToyModel()
    task = ToyTask()
    cap = Capture()
    batch = {"token_ids": torch.ones(2, 4, dtype=torch.long)}
    trainer = Trainer(
        model=model, task=task,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=WarmupCosine(max_lr=1e-3, min_lr=1e-4, t_warmup="2ba", t_max="8ba"),
        cfg=cfg, dist_env=DistEnv(ddp=False, device="cpu", device_type="cpu"),
        logger=MultiLogger([cap], is_master=True),
        train_loader=[batch] * n_micro,
        eval_loader=[batch] * n_eval if n_eval else [],
    )
    return trainer, cap, task, model


def test_step_count_and_grad_accum() -> None:
    trainer, cap, task, _ = build(make_cfg())
    trainer.fit()
    train_lines = [line for line in cap.lines if line.startswith("[train")]
    assert trainer.state.step == 4
    assert trainer.steps_per_epoch == 2
    assert len(train_lines) == 4
    # every step: loss, accuracy and tokens-per-sec
    assert "loss " in train_lines[0] and "acc " in train_lines[0] and "tok/s " in train_lines[0]


def test_eval_fires_on_interval() -> None:
    trainer, cap, _, _ = build(make_cfg())
    trainer.fit()
    assert len([line for line in cap.lines if line.startswith("[eval")]) == 2


def test_epoch_duration() -> None:
    trainer, _, _, _ = build(make_cfg(max_duration="3ep", eval_interval="100ba"))
    trainer.fit()
    assert (trainer.state.step, trainer.state.epoch) == (6, 3)


def test_partial_accumulation_is_discarded_at_epoch_end() -> None:
    """len(loader) is not divisible by grad accum: the remainder must not leak on."""
    trainer, cap, task, _ = build(make_cfg(max_duration="1ep", eval_interval="100ba"), n_micro=5)
    trainer.fit()
    assert trainer.state.step == 2       # 5 microbatches -> 2 full steps
    assert task.calls == 5               # the fifth microbatch is computed and dropped


def test_sampling_hook_and_examples(tmp_path: Path) -> None:
    seen = []

    def sample_fn(step: int):
        seen.append(step)
        return {"metrics": {"validity": 0.5}, "examples": ["CCO", "c1ccccc1"]}

    cfg = make_cfg(eval_interval="100ba",
                   sampling={"enabled": True, "interval": "2ba", "log_examples": 2})
    trainer, cap, _, _ = build(cfg)
    trainer.sample_fn = sample_fn
    trainer.fit()
    assert seen == [2, 4]
    assert any(line.startswith("[sample") and "validity" in line for line in cap.lines)


def test_checkpoint_not_saved_twice(tmp_path: Path) -> None:
    model = ToyModel()
    cfg = make_cfg(save_folder=str(tmp_path), save_interval="2ba", save_last=True,
                   eval_interval="100ba")
    trainer, _, _, _ = build(cfg, model=model)
    trainer.fit()
    assert model.saves == 2, f"expected steps 2 and 4, got {model.saves} saves"
    assert {p.name for p in (tmp_path / "toy").iterdir() if p.is_dir()} == {"ep0-ba2", "ep1-ba4"}


def test_resume_finished_run_is_noop() -> None:
    trainer, cap, task, _ = build(make_cfg(eval_interval="100ba"))
    trainer.state.step, trainer.state.epoch = 4, 2
    trainer.fit()
    assert task.calls == 0 and trainer.state.step == 4


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    cfg = make_cfg(save_folder=str(tmp_path), save_interval="2ba", save_last=True,
                   eval_interval="100ba", autoresume=True, max_duration="2ba")
    first, _, _, _ = build(cfg, model=ToyModel())
    first.fit()
    assert first.state.step == 2

    second, cap, task, _ = build(cfg, model=ToyModel())
    second.maybe_resume()
    assert second.state.step == 2
    second.fit()                      # max_duration already reached
    assert task.calls == 0


def test_empty_eval_loader_is_skipped() -> None:
    trainer, cap, _, _ = build(make_cfg(), n_eval=0)
    trainer.fit()
    assert any("skipped" in line for line in cap.lines)


def test_empty_train_loader_raises() -> None:
    with pytest.raises(ValueError):
        build(make_cfg(), n_micro=0)


def test_bad_clipping_type_raises() -> None:
    with pytest.raises(ValueError):
        build(make_cfg(algorithms={"gradient_clipping":
                                   {"clipping_type": "value", "clipping_threshold": 1.0}}))
