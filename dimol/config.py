"""Loading and validation of the experiment yaml config.

The conventions follow llm-foundry (as vendored in
/Users/ivangurev/code/sber/llm-foundry):

    python scripts/train.py configs/diffusion_chebi.yaml optimizer.lr=1e-4 seed=1

* one yaml per run, its path is the first argument;
* CLI overrides are dotted OmegaConf paths and win over the file;
* durations are expressed in batches/epochs: ``20000ba``, ``500ep``;
* batching: you write ``global_train_batch_size`` and
  ``device_train_microbatch_size``, while grad accumulation is DERIVED
  (``device_train_grad_accum``), as in ``update_batch_size_info``.

One deliberate difference from llm-foundry: the config is validated against a
schema (the dataclasses below) with struct mode enabled, so a typo in a key name
fails immediately instead of being silently ignored.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

# ----------------------------------------------------------------------
# Durations: "20000ba" (batches = optimizer steps), "500ep" (epochs)
# ----------------------------------------------------------------------
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ba|ep)\s*$")


@dataclass(frozen=True)
class Duration:
    value: float
    unit: str  # 'ba' | 'ep'

    @classmethod
    def parse(cls, spec: Any) -> "Duration":
        if isinstance(spec, Duration):
            return spec
        if isinstance(spec, bool):
            raise ValueError(f"Expected a duration, got a bool: {spec!r}")
        if isinstance(spec, (int, float)):
            return cls(float(spec), "ba")
        m = _DURATION_RE.match(str(spec))
        if not m:
            raise ValueError(
                f"Cannot parse duration {spec!r}. "
                "Expected '1000ba', '50ep' or a plain number (treated as ba)."
            )
        return cls(float(m.group(1)), m.group(2))

    def in_batches(self, steps_per_epoch: Optional[int] = None) -> int:
        if self.unit == "ba":
            return int(self.value)
        if not steps_per_epoch:
            raise ValueError("steps_per_epoch is required to convert epochs into steps")
        return int(self.value * steps_per_epoch)

    def __str__(self) -> str:
        v = int(self.value) if float(self.value).is_integer() else self.value
        return f"{v}{self.unit}"


def duration_reached(spec: Optional[Any], step: int, epoch: int) -> bool:
    """Whether the end of training has been reached (``max_duration``)."""
    if spec is None:
        return False
    d = Duration.parse(spec)
    return step >= int(d.value) if d.unit == "ba" else epoch >= int(d.value)


def duration_hit(spec: Optional[Any], step: int, steps_per_epoch: Optional[int] = None) -> bool:
    """Whether a periodic event should fire at this step."""
    if spec is None:
        return False
    d = Duration.parse(spec)
    n = d.in_batches(steps_per_epoch)
    return n > 0 and step > 0 and step % n == 0


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4  # initial value; overwritten by the schedule at every step
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.01
    fused: Optional[bool] = None  # None -> auto: cuda and supported by torch


@dataclass
class SchedulerConfig:
    name: str = "warmup_cosine"
    max_lr: float = 3e-3
    min_lr: float = 3e-5
    t_warmup: Any = "500ba"
    t_max: Any = "2500ba"


@dataclass
class GradientClipping:
    clipping_type: str = "norm"
    clipping_threshold: Optional[float] = 1.0


@dataclass
class Algorithms:
    gradient_clipping: GradientClipping = field(default_factory=GradientClipping)


@dataclass
class DDPConfig:
    backend: str = "nccl"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


@dataclass
class EmaConfig:
    """Averaged weights for sampling; written next to the checkpoint as <name>_ema."""

    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0
    update_every: int = 1


@dataclass
class SamplingConfig:
    """Periodic generation during training (diffusion only)."""

    enabled: bool = True
    interval: Any = "500ba"
    num_samples: int = 64
    num_timesteps: int = 300
    variance: float = 1.0
    t_start: float = 1e-4
    t_end: float = 0.999
    seed: Optional[int] = None  # None -> process rank (as in the old code)
    log_examples: int = 10
    batch_size: Optional[int] = None  # None -> all num_samples in one go
    progress: bool = False  # tqdm over SDE steps


@dataclass
class DimolConfig:
    # ---- run identity ----
    run_name: str = "run"
    task: str = "diffusion"  # diffusion | ar
    seed: int = 42
    variables: Dict[str, Any] = field(default_factory=dict)  # only for ${variables.x}

    # ---- components (built through dimol.registry by their name field) ----
    tokenizer: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    diffusion: Dict[str, Any] = field(default_factory=dict)
    loss: Dict[str, Any] = field(default_factory=dict)
    # free-form nodes for the analysis scripts: they take a training config and only
    # need somewhere to put a checkpoint path and a few knobs
    analyze: Dict[str, Any] = field(default_factory=dict)
    train_loader: Dict[str, Any] = field(default_factory=dict)
    eval_loader: Optional[Dict[str, Any]] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    algorithms: Algorithms = field(default_factory=Algorithms)

    # ---- training parameters ----
    max_duration: Any = "500ep"
    global_train_batch_size: Optional[int] = None  # None -> microbatch * world_size
    device_train_microbatch_size: int = 512
    device_eval_batch_size: Optional[int] = None  # None -> same as the train microbatch
    eval_interval: Any = "25ba"
    eval_subset_num_batches: int = -1
    eval_before_train: str = "skip"  # skip | regular
    decoder_pretrain_steps: int = 0  # steps where lambda_mse = 0 (kept from the old config)

    # ---- compute ----
    precision: str = "amp_fp16"  # amp_bf16 | amp_fp16 | fp32
    tf32: bool = False
    compile: bool = False
    compile_backend: str = "inductor"
    device: Optional[str] = None  # None -> cuda when available
    deterministic: bool = False
    sync_each_step: bool = True  # cuda.synchronize() before measuring step time
    dist_timeout: int = 1800
    ddp_config: DDPConfig = field(default_factory=DDPConfig)

    # ---- logging ----
    log_to_console: bool = True
    console_log_interval: Any = "1ba"  # every step: loss / tokens-per-sec / accuracy
    loggers: Dict[str, Any] = field(default_factory=dict)

    # ---- checkpointing ----
    save_folder: Optional[str] = "checkpoints"
    save_interval: Optional[Any] = "100ep"
    save_num_checkpoints_to_keep: int = 3
    save_last: bool = True
    save_weights_only: bool = False
    autoresume: bool = True
    load_path: Optional[str] = None
    load_weights_only: bool = False

    # ---- generation during training and in scripts/generate.py ----
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    generate: Dict[str, Any] = field(default_factory=dict)

    # ---- data preparation: scripts/train_tokenizer.py, scripts/tokenize_dataset.py ----
    prepare: Dict[str, Any] = field(default_factory=dict)

    # ---- fields filled in at runtime (do not set them in yaml) ----
    n_gpus: Optional[int] = None
    device_train_batch_size: Optional[int] = None
    device_train_grad_accum: Optional[int] = None
    n_params: Optional[int] = None


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_config(path: str | Path, cli_args: Optional[List[str]] = None) -> DictConfig:
    """yaml + CLI overrides -> a validated DictConfig (struct mode)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    file_cfg = OmegaConf.load(path)
    cli_cfg = OmegaConf.from_cli(list(cli_args or []))
    schema = OmegaConf.structured(DimolConfig)

    cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg  # type: ignore[return-value]


def parse_args(argv: Optional[List[str]] = None) -> Tuple[Path, List[str]]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Usage: python <script>.py <config.yaml> [key.subkey=value ...]")
    return Path(argv[0]), argv[1:]


def load_from_argv(argv: Optional[List[str]] = None) -> DictConfig:
    cfg_path, overrides = parse_args(argv)
    return load_config(cfg_path, overrides)


# ----------------------------------------------------------------------
# Derived batch sizes (counterpart of llm-foundry's update_batch_size_info)
# ----------------------------------------------------------------------
def update_batch_size_info(cfg: DictConfig, world_size: int) -> DictConfig:
    """Fill in device_train_batch_size / device_train_grad_accum / n_gpus."""
    micro = int(cfg.device_train_microbatch_size)
    if micro <= 0:
        raise ValueError("device_train_microbatch_size must be > 0")

    if cfg.global_train_batch_size is None:
        cfg.global_train_batch_size = micro * world_size

    global_bs = int(cfg.global_train_batch_size)
    if global_bs % world_size != 0:
        raise ValueError(
            f"global_train_batch_size={global_bs} is not divisible by world_size={world_size}"
        )
    device_bs = global_bs // world_size
    if device_bs % micro != 0:
        raise ValueError(
            f"device batch {device_bs} (= {global_bs}/{world_size}) is not divisible by "
            f"device_train_microbatch_size={micro}"
        )

    cfg.n_gpus = world_size
    cfg.device_train_batch_size = device_bs
    cfg.device_train_grad_accum = max(1, math.ceil(device_bs / micro))
    if cfg.device_eval_batch_size is None:
        cfg.device_eval_batch_size = micro
    return cfg


# ----------------------------------------------------------------------
# Run state on disk
# ----------------------------------------------------------------------
def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def flatten_config(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    """A flat dict for the loggers (comet/wandb accept scalars only)."""
    out: Dict[str, Any] = {}
    if isinstance(cfg, DictConfig):
        cfg = config_to_dict(cfg)
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(flatten_config(v, f"{prefix}{k}."))
    elif isinstance(cfg, (list, tuple)):
        out[prefix.rstrip(".")] = json.dumps(list(cfg))
    else:
        out[prefix.rstrip(".")] = cfg
    return out


def _git_state() -> Dict[str, Any]:
    def run(*cmd: str) -> Optional[str]:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except Exception:
            return None

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def save_run_state(cfg: DictConfig, run_dir: str | Path) -> Path:
    """Store the resolved config, git state and environment next to the run."""
    import torch

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.resolved.yaml")

    env: Dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "git": _git_state(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    return run_dir


def log_config(cfg: DictConfig) -> None:
    print("=" * 78)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 78, flush=True)
