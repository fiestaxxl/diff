"""Loggers: console (every step), comet, jsonl.

The metric key naming follows llm-foundry/Composer so that the charts look
familiar: ``loss/train/total``, ``metrics/train/token_acc``,
``throughput/tokens_per_sec``, ``lr-AdamW/group0``, ``l2_norm/grad/global``,
``time_seconds/batch/batch_total``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dimol import registry


def _fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}K"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"


class Logger:
    """Base interface."""

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        pass

    def log_line(self, line: str) -> None:
        pass

    def log_text(self, name: str, text: str, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class MultiLogger(Logger):
    """Fan-out to several loggers. Only the master process writes."""

    def __init__(self, loggers: Iterable[Logger], is_master: bool = True):
        self.loggers: List[Logger] = [lg for lg in loggers if lg is not None]
        self.is_master = is_master

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        if not self.is_master:
            return
        for lg in self.loggers:
            lg.log_hyperparameters(params)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if not self.is_master:
            return
        for lg in self.loggers:
            lg.log_metrics(metrics, step)

    def log_line(self, line: str) -> None:
        if not self.is_master:
            return
        for lg in self.loggers:
            lg.log_line(line)

    def log_text(self, name: str, text: str, step: int) -> None:
        if not self.is_master:
            return
        for lg in self.loggers:
            lg.log_text(name, text, step)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()


@registry.loggers.register("console")
class ConsoleLogger(Logger):
    def __init__(self, stream: str = "stdout", **kwargs: Any):
        self.stream = sys.stderr if stream == "stderr" else sys.stdout

    def log_line(self, line: str) -> None:
        print(line, file=self.stream, flush=True)

    def log_text(self, name: str, text: str, step: int) -> None:
        print(f"[{name} @ ba {step}]\n{text}", file=self.stream, flush=True)


@registry.loggers.register("jsonl")
class JsonlLogger(Logger):
    """Machine-readable metric log: one json line per event."""

    def __init__(self, path: str = "metrics.jsonl", **kwargs: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        self._write({"event": "hparams", "params": params})

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        row: Dict[str, Any] = {"event": "metrics", "step": step, "wall_clock": time.time()}
        row.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        self._write(row)

    def log_text(self, name: str, text: str, step: int) -> None:
        self._write({"event": "text", "name": name, "step": step, "text": text})

    def _write(self, row: Dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


@registry.loggers.register("comet")
class CometLogger(Logger):
    def __init__(
        self,
        project_name: str = "diffusion_smiles_model",
        workspace: Optional[str] = None,
        experiment_name: Optional[str] = None,
        mode: str = "create",
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        import comet_ml

        self.exp = comet_ml.start(project_name=project_name, workspace=workspace, mode=mode)
        if experiment_name:
            self.exp.set_name(experiment_name)
        if tags:
            self.exp.add_tags(list(tags))

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        self.exp.log_parameters(params)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        payload = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self.exp.log_metrics(payload, step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.exp.log_text(text, step=step, metadata={"name": name})

    def close(self) -> None:
        try:
            self.exp.end()
        except Exception:
            pass


@registry.loggers.register("tensorboard")
class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str = "tb", **kwargs: Any):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        scalars = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        self.writer.add_text("config", json.dumps(scalars, indent=2, default=str))

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(name, text, global_step=step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


# ----------------------------------------------------------------------
# Step line formatting: loss / tokens-per-sec / accuracy on every step
# ----------------------------------------------------------------------
_SHORT = {
    "mse_loss": "mse",
    "ce_loss": "ce",
    "mse_loss_t0": "mse_t0",
    "grammar_loss": "gram",
    "token_acc": "acc",
    "ppl": "ppl",
}
_HIDE_PREFIXES = ("token_acc_", "emb_", "eps_theta_")


def format_step_line(
    *,
    split: str,
    step: int,
    max_steps: Optional[int],
    epoch: int,
    metrics: Dict[str, float],
    lr: Optional[float] = None,
    grad_norm: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    samples_per_sec: Optional[float] = None,
    dt_ms: Optional[float] = None,
    mem_gb: Optional[float] = None,
) -> str:
    head = f"[{split} ba {step}"
    if max_steps:
        head += f"/{max_steps}"
    head += f" ep {epoch}]"

    parts: list[str] = []
    if "loss" in metrics:
        parts.append(f"loss {metrics['loss']:.4f}")
    if "token_acc" in metrics:
        parts.append(f"acc {metrics['token_acc']:.4f}")
    for k, v in metrics.items():
        if k in ("loss", "token_acc") or k.startswith(_HIDE_PREFIXES):
            continue
        parts.append(f"{_SHORT.get(k, k)} {_fmt_num(float(v))}")
    if lr is not None:
        parts.append(f"lr {lr:.3e}")
    if grad_norm is not None:
        parts.append(f"gnorm {grad_norm:.3f}")
    if tokens_per_sec is not None:
        parts.append(f"tok/s {_fmt_num(tokens_per_sec)}")
    if samples_per_sec is not None:
        parts.append(f"mol/s {_fmt_num(samples_per_sec)}")
    if dt_ms is not None:
        parts.append(f"dt {dt_ms:.0f}ms")
    if mem_gb is not None:
        parts.append(f"mem {mem_gb:.1f}G")
    return head + " " + " | ".join(parts)


def namespaced_metrics(
    metrics: Dict[str, float],
    split: str,
    *,
    lr: Optional[float] = None,
    optimizer_name: str = "AdamW",
    grad_norm: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    samples_per_sec: Optional[float] = None,
    dt_s: Optional[float] = None,
    mem_gb: Optional[float] = None,
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """A flat Composer-style metric dict for comet/tensorboard/jsonl."""
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        v = float(v)
        if k == "loss":
            out[f"loss/{split}/total"] = v
        elif k.endswith("_loss") or k.startswith("loss"):
            out[f"loss/{split}/{k}"] = v
        else:
            out[f"metrics/{split}/{k}"] = v
    if lr is not None:
        out[f"lr-{optimizer_name}/group0"] = float(lr)
    if grad_norm is not None:
        out["l2_norm/grad/global"] = float(grad_norm)
    if tokens_per_sec is not None:
        out["throughput/tokens_per_sec"] = float(tokens_per_sec)
    if samples_per_sec is not None:
        out["throughput/samples_per_sec"] = float(samples_per_sec)
    if dt_s is not None:
        out["time_seconds/batch/batch_total"] = float(dt_s)
    if mem_gb is not None:
        out["memory/peak_allocated_gb"] = float(mem_gb)
    if epoch is not None:
        out["trainer/epoch"] = float(epoch)
    return out
