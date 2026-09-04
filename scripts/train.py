#!/usr/bin/env python3
"""The single training entrypoint: a yaml path -> a built experiment -> training.

    python scripts/train.py configs/diffusion_chebi.yaml
    python scripts/train.py configs/diffusion_chebi.yaml loss.lambda_grammar=0.01 run_name=lg_0.01
    torchrun --nproc_per_node=8 scripts/train.py configs/diffusion_chebi.yaml

Everything that defines the experiment (architecture, hyperparameters, optimizer,
precision, grad accumulation, eval/checkpoint intervals, losses, loggers) is set
in the yaml.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import (  # noqa: E402
    build_loggers,
    build_model,
    build_optimizer,
    build_path,
    build_scheduler,
    build_task,
    build_tokenizer,
    log_hyperparameters,
)
from dimol.config import (  # noqa: E402
    load_from_argv,
    log_config,
    save_run_state,
    update_batch_size_info,
)
from dimol.data.loaders import build_dataloader  # noqa: E402
from dimol.eval.sampling import SamplingParams, sample_smiles, validity  # noqa: E402
from dimol.training.distributed import (  # noqa: E402
    destroy_distributed,
    init_distributed,
    seed_all,
)
from dimol.training.trainer import Trainer  # noqa: E402


def _infer_seq_len(loader: torch.utils.data.DataLoader, cfg: DictConfig) -> int:
    ds_len = getattr(loader.dataset, "max_len", None)
    cfg_len = cfg.variables.get("seq_len", None) if cfg.variables else None
    if ds_len is not None and cfg_len is not None and int(ds_len) != int(cfg_len):
        raise ValueError(
            f"variables.seq_len={cfg_len} does not match the dataset length ({ds_len}). "
            "Fix the config or re-tokenize the data."
        )
    seq_len = ds_len if ds_len is not None else cfg_len
    if seq_len is None:
        raise ValueError("Could not infer seq_len: set variables.seq_len")
    return int(seq_len)


def _autocast_factory(cfg: DictConfig, device_type: str):
    precision = str(cfg.precision)
    if precision == "fp32" or device_type != "cuda":
        from contextlib import nullcontext

        return nullcontext
    dtype = torch.bfloat16 if precision == "amp_bf16" else torch.float16
    return partial(torch.autocast, device_type=device_type, dtype=dtype)


def _build_scaler(cfg: DictConfig, device_type: str):
    """GradScaler is only needed for fp16 (bf16 has the same range as fp32)."""
    if str(cfg.precision) != "amp_fp16" or device_type != "cuda":
        return None
    from torch.amp import GradScaler

    return GradScaler(device=device_type)


def _sample_fn(
    step: int,
    *,
    model: torch.nn.Module,
    path: Any,
    tokenizer: Any,
    cfg: DictConfig,
    device: str,
    rank: int,
    run_dir: Optional[Path],
) -> Dict[str, Any]:
    sc = cfg.sampling
    params = SamplingParams(
        num_samples=int(sc.num_samples),
        num_timesteps=int(sc.num_timesteps),
        variance=float(sc.variance),
        t_start=float(sc.t_start),
        t_end=float(sc.t_end),
        seed=int(sc.seed) if sc.seed is not None else rank,
        batch_size=int(sc.batch_size) if sc.batch_size else None,
        regime=str(cfg.diffusion.get("regime", "epsilon")),
        progress=bool(sc.progress),
    )
    smiles = sample_smiles(model, path, tokenizer, params, device)
    metrics = {"validity": validity(smiles)}
    if run_dir is not None and rank == 0:
        out = Path(run_dir) / "samples"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"ba{step}.txt").write_text("\n".join(smiles))
    return {"metrics": metrics, "examples": smiles}


def main(cfg: DictConfig) -> None:
    # ---- environment ----
    env = init_distributed(
        backend=str(cfg.ddp_config.backend),
        timeout_min=int(cfg.dist_timeout) // 60 or 1,
        device=cfg.device,
    )
    cfg = update_batch_size_info(cfg, env.world_size)
    seed_all(int(cfg.seed), rank=env.rank, deterministic=bool(cfg.deterministic), tf32=bool(cfg.tf32))

    run_dir = Path(cfg.save_folder) / str(cfg.run_name) if cfg.save_folder else None
    if env.is_master:
        log_config(cfg)

    # ---- tokenizer -> model -> data (same order as llm-foundry) ----
    tokenizer = build_tokenizer(cfg) if cfg.tokenizer.get("path", None) else None

    model = build_model(cfg, tokenizer)
    model.to(env.device)
    n_params = sum(p.numel() for p in model.parameters())
    cfg.n_params = n_params
    if env.is_master:
        print(f"[model] {type(model).__name__}: {n_params:,} parameters")
        if run_dir is not None:
            # save the config after the model is built so that n_params and the
            # derived batch sizes end up in it
            save_run_state(cfg, run_dir)

    if bool(cfg.compile):
        model = torch.compile(model, backend=str(cfg.compile_backend))
    if env.ddp:
        model = DDP(
            model,
            device_ids=[env.local_rank] if env.device_type == "cuda" else None,
            find_unused_parameters=bool(cfg.ddp_config.find_unused_parameters),
            gradient_as_bucket_view=bool(cfg.ddp_config.gradient_as_bucket_view),
        )

    train_loader, train_sampler = build_dataloader(
        cfg.train_loader, int(cfg.device_train_microbatch_size), env, is_train=True
    )
    eval_loader = eval_sampler = None
    if cfg.eval_loader is not None:
        eval_loader, eval_sampler = build_dataloader(
            cfg.eval_loader, int(cfg.device_eval_batch_size), env, is_train=False
        )
    seq_len = _infer_seq_len(train_loader, cfg)

    # ---- optimizer, lr schedule, loggers ----
    optimizer = build_optimizer(cfg, model, device_type=env.device_type, verbose=env.is_master)
    grad_accum = int(cfg.device_train_grad_accum or 1)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    scheduler = build_scheduler(cfg, steps_per_epoch=steps_per_epoch)
    logger = build_loggers(cfg, env, run_dir=run_dir)
    log_hyperparameters(logger, cfg)

    # ---- probability path and task ----
    autocast = _autocast_factory(cfg, env.device_type)
    path = None
    if str(cfg.task) == "diffusion":
        emb_dim = int(cfg.model["emb_dim"])
        path = build_path(cfg, seq_len=seq_len, emb_dim=emb_dim, device=env.device)
    task = build_task(
        cfg,
        path=path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        device=env.device,
        autocast=autocast,
    )

    sample_fn = None
    if str(cfg.task) == "diffusion" and bool(cfg.sampling.enabled) and tokenizer is not None:
        try:  # probe rdkit now rather than sampling.interval steps later
            import rdkit  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sampling.enabled=true requires rdkit (pip install 'dimol[chem]'); "
                "otherwise set sampling.enabled=false"
            ) from exc
        sample_fn = partial(
            _sample_fn,
            model=model,
            path=path,
            tokenizer=tokenizer,
            cfg=cfg,
            device=env.device,
            rank=env.rank,
            run_dir=run_dir,
        )

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        dist_env=env,
        logger=logger,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_sampler=train_sampler,
        eval_sampler=eval_sampler,
        sample_fn=sample_fn,
        scaler=_build_scaler(cfg, env.device_type),
    )
    trainer.maybe_resume()
    try:
        trainer.fit()
    finally:
        logger.close()
        destroy_distributed()


if __name__ == "__main__":
    main(load_from_argv())
