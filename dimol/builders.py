"""Build objects from config: model, probability path, task, optimizer, lr
schedule, loggers, sampler.

Each function takes a config node and returns a ready object. Every swappable
component is registered in dimol.registry, so the yaml only needs a ``name``.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from dimol import registry
from dimol.config import flatten_config
from dimol.diffusion.conditionals import (
    CosineAlpha,
    CosineBeta,
    LinearAlpha,
    SquareRootBeta,
)
from dimol.diffusion.diff_eqs import LearnedScoreSDE
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.diffusion.simulators import EulerMaruyamaSimulator, EulerSimulator
from dimol.models.denoiser import DenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.gpt import SmilesAR, SmilesARConfig
from dimol.training import logging as dimol_logging  # noqa: F401  (registers the loggers)
from dimol.training import tasks as dimol_tasks  # registers the tasks
from dimol.training.distributed import DistEnv
from dimol.training.logging import MultiLogger
from dimol.training.optim import param_groups

if TYPE_CHECKING:
    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

# ----------------------------------------------------------------------
# Component registration
# ----------------------------------------------------------------------
registry.alphas.register("cosine")(CosineAlpha)
# LinearAlpha / SquareRootBeta take no device argument (unlike the cosine ones),
# so they are registered through an adapter that swallows it
registry.alphas.register("linear")(lambda device=None, **_: LinearAlpha())
registry.betas.register("cosine")(CosineBeta)
registry.betas.register("sqrt")(lambda device=None, **_: SquareRootBeta())
# LinearBeta from dimol/diffusion/conditionals.py is deliberately NOT registered:
# its __call__ applies clamp(1-t, min=1e-5), which makes the base-class check
# beta(1)=0 fail on construction. The class itself is left in the code as is.
registry.sdes.register("learned_score")(LearnedScoreSDE)
registry.simulators.register("euler_maruyama")(EulerMaruyamaSimulator)
registry.simulators.register("euler")(EulerSimulator)


@registry.models.register("diffusion_transformer")
def build_diffusion_transformer(**kwargs: Any) -> DiffusionTransformer:
    return DiffusionTransformer(TransformerConfig(**kwargs))


@registry.models.register("smiles_ar")
def build_smiles_ar(**kwargs: Any) -> SmilesAR:
    return SmilesAR(SmilesARConfig(**kwargs))


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
def build_tokenizer(cfg: DictConfig) -> "SmilesTokenizer":
    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

    node = _as_dict(cfg.tokenizer)
    path = node.get("path")
    if not path:
        raise KeyError("tokenizer.path is not set")
    return SmilesTokenizer.load(path)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(cfg: DictConfig, tokenizer: Optional["SmilesTokenizer"] = None) -> torch.nn.Module:
    node = _as_dict(cfg.model)
    if tokenizer is not None and "vocab_size" in node:
        if int(node["vocab_size"]) < tokenizer.vocab_size:
            raise ValueError(
                f"model.vocab_size={node['vocab_size']} is smaller than the tokenizer vocab "
                f"({tokenizer.vocab_size})"
            )
    return registry.models.build(node)


# ----------------------------------------------------------------------
# Probability path (alpha/beta + p_simple)
# ----------------------------------------------------------------------
def build_path(
    cfg: DictConfig,
    *,
    seq_len: int,
    emb_dim: int,
    device: str | torch.device,
) -> GaussianConditionalProbabilityPath:
    node = _as_dict(cfg.diffusion)
    alpha_cfg = node.get("alpha", {"name": "cosine"})
    beta_cfg = node.get("beta", {"name": "cosine"})
    alpha = registry.alphas.build(alpha_cfg, device=device)
    beta = registry.betas.build(beta_cfg, device=device)
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[seq_len, emb_dim],
        alpha=alpha,
        beta=beta,
        p_simple_std=float(node.get("p_simple_std", 1.0)),
    )
    return path.to(device)


# ----------------------------------------------------------------------
# Task (loss)
# ----------------------------------------------------------------------
def build_task(
    cfg: DictConfig,
    *,
    path: Optional[GaussianConditionalProbabilityPath],
    tokenizer: Optional["SmilesTokenizer"],
    seq_len: int,
    device: str | torch.device,
    autocast: Callable[[], Any] = nullcontext,
) -> dimol_tasks.Task:
    loss_cfg = _as_dict(cfg.loss)
    diff_cfg = _as_dict(cfg.diffusion)
    task_name = str(cfg.task)

    pad_idx = int(_as_dict(cfg.model).get("pad_idx", 0))
    kwargs: Dict[str, Any] = {
        "pad_idx": pad_idx,
        "label_smoothing": float(loss_cfg.get("label_smoothing", 0.0)),
        "autocast": autocast,
        "autocast_scope": str(loss_cfg.get("autocast_scope", "loss")),
        "seq_len": seq_len,
    }

    if task_name == "ar":
        return registry.tasks.build({"name": "ar", **kwargs})

    class_weights = None
    cw_path = loss_cfg.get("class_weights_path")
    if cw_path:
        class_weights = torch.load(cw_path, weights_only=True).to(device)
        print(f"Using class weights: {cw_path}")

    vocab_size = int(_as_dict(cfg.model)["vocab_size"])
    lambda_grammar = float(loss_cfg.get("lambda_grammar", 0.0))
    grammar_enabled = bool(loss_cfg.get("grammar_enabled", True)) and lambda_grammar != 0.0
    paren_delta = ring_count = None
    if grammar_enabled:
        if tokenizer is None:
            raise ValueError("the grammar loss needs a tokenizer")
        paren_delta, ring_count = dimol_tasks.build_grammar_tables(tokenizer, vocab_size, device)

    kwargs.update(
        {
            "name": "diffusion",
            "path": path,
            "regime": str(diff_cfg.get("regime", "epsilon")),
            "x0_noise_std": float(diff_cfg.get("x0_noise_std", 0.25)),
            "x0_noise_kind": str(diff_cfg.get("x0_noise_kind", "gaussian")),
            "embedding_norm": str(diff_cfg.get("embedding_norm", "none")),
            "time_sampler": str(diff_cfg.get("time_sampler", "uniform")),
            "time_logit_mean": float(diff_cfg.get("time_logit_mean", 0.0)),
            "time_logit_std": float(diff_cfg.get("time_logit_std", 1.0)),
            "t_eps": float(diff_cfg.get("t_eps", 5e-5)),
            "lambda_mse": float(loss_cfg.get("lambda_mse", 1.0)),
            "lambda_ce": float(loss_cfg.get("lambda_ce", 1.0)),
            "lambda_grammar": lambda_grammar,
            "alpha_eps": float(loss_cfg.get("alpha_eps", 1e-3)),
            "ce_alpha_threshold": float(loss_cfg.get("ce_alpha_threshold", 0.80)),
            "mse_t0_alpha_threshold": float(loss_cfg.get("mse_t0_alpha_threshold", 0.80)),
            "grammar_alpha_threshold": float(loss_cfg.get("grammar_alpha_threshold", 0.5)),
            "mask_padding": loss_cfg.get("mask_padding", "none"),
            "pad_weight": float(loss_cfg.get("pad_weight", 1.0)),
            "self_cond_prob": float(loss_cfg.get("self_cond_prob", 0.5)),
            "caption_dropout": float(loss_cfg.get("caption_dropout", 0.1)),
            "gate_mode": str(loss_cfg.get("gate_mode", "threshold")),
            "gate_fraction": float(loss_cfg.get("gate_fraction", 0.41)),
            "ce_include_pad": bool(loss_cfg.get("ce_include_pad", False)),
            "min_snr_gamma": (
                None
                if loss_cfg.get("min_snr_gamma") is None
                else float(loss_cfg["min_snr_gamma"])
            ),
            "ce_input": str(loss_cfg.get("ce_input", "x0")),
            "decoder_pretrain_steps": int(cfg.decoder_pretrain_steps),
            "grammar_enabled": grammar_enabled,
            "class_weights": class_weights,
            "paren_delta": paren_delta,
            "ring_count": ring_count,
        }
    )
    return registry.tasks.build(kwargs)


# ----------------------------------------------------------------------
# Optimizer and lr schedule
# ----------------------------------------------------------------------
def build_optimizer(
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    device_type: str,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    node = _as_dict(cfg.optimizer)
    name = node.pop("name")
    no_decay = node.pop("no_decay_patterns", None)
    if no_decay is None:
        # old behaviour: the denoiser kept token_embedding out of weight decay,
        # while the AR model had no special rules
        no_decay = ["token_embedding"] if str(cfg.task) == "diffusion" else []
    params = param_groups(
        model,
        weight_decay=float(node.pop("weight_decay", 0.0)),
        no_decay_patterns=no_decay,
        verbose=verbose,
    )
    return registry.optimizers.build(
        {"name": name, **node}, params=params, device_type=device_type, verbose=verbose
    )


def build_scheduler(cfg: DictConfig, steps_per_epoch: Optional[int] = None):
    node = _as_dict(cfg.scheduler)
    return registry.schedulers.build(node, steps_per_epoch=steps_per_epoch)


# ----------------------------------------------------------------------
# Loggers
# ----------------------------------------------------------------------
def build_loggers(cfg: DictConfig, env: DistEnv, run_dir: Optional[Path] = None) -> MultiLogger:
    """``loggers: {console: {}, comet: {...}}`` -> MultiLogger (rank 0 writes only)."""
    node = _as_dict(cfg.loggers)
    built: List[Any] = []
    for name, kwargs in node.items():
        kwargs = dict(kwargs or {})
        if not kwargs.pop("enabled", True):
            continue
        if not env.is_master:
            continue
        if name == "jsonl" and "path" not in kwargs and run_dir is not None:
            kwargs["path"] = str(Path(run_dir) / "metrics.jsonl")
        if name == "comet" and "experiment_name" not in kwargs:
            kwargs["experiment_name"] = str(cfg.run_name)
        built.append(registry.loggers.build({"name": name, **kwargs}))
    if bool(cfg.log_to_console) and "console" not in node and env.is_master:
        built.append(registry.loggers.build({"name": "console"}))
    return MultiLogger(built, is_master=env.is_master)


def log_hyperparameters(logger: MultiLogger, cfg: DictConfig) -> None:
    logger.log_hyperparameters(flatten_config(cfg))


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def build_sampler_pipeline(
    cfg: DictConfig,
    model: torch.nn.Module,
    path: GaussianConditionalProbabilityPath,
    *,
    variance: float,
    regime: str,
) -> Tuple[DenoiserModel, Any]:
    """DenoiserModel plus the SDE simulator (same order and parameters as before)."""
    score_model = DenoiserModel(model, path, regime=regime)
    sde = LearnedScoreSDE(path, score_model, variance)
    simulator = EulerMaruyamaSimulator(sde)
    return score_model, simulator


# ----------------------------------------------------------------------
def _as_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]
    return dict(node)


__all__ = [
    "build_tokenizer",
    "build_model",
    "build_path",
    "build_task",
    "build_optimizer",
    "build_scheduler",
    "build_loggers",
    "build_sampler_pipeline",
    "log_hyperparameters",
]
