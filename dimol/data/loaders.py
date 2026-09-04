"""Build datasets and DataLoaders from config."""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dimol import registry
from dimol.data import datasets as _datasets  # noqa: F401  (registers datasets in the registry)
from dimol.data.collate import trim_collate
from dimol.data.samplers import BucketBatchSampler
from dimol.training.distributed import DistEnv


def build_dataset(cfg: Any) -> Dataset:
    return registry.datasets.build(cfg)


def build_dataloader(
    cfg: Any,
    device_batch_size: int,
    env: DistEnv,
    is_train: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """A config node like ``{dataset: {...}, num_workers: 4, ...}`` -> (loader, sampler)."""
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg or {})

    dataset_cfg = cfg.pop("dataset", None)
    if dataset_cfg is None:
        raise KeyError("loader config has no 'dataset' section")
    dataset = build_dataset(dataset_cfg)

    shuffle = bool(cfg.pop("shuffle", is_train))
    drop_last = bool(cfg.pop("drop_last", is_train))
    num_workers = int(cfg.pop("num_workers", 4))
    bucket_by_length = bool(cfg.pop("bucket_by_length", False))
    bucket_pool_factor = int(cfg.pop("bucket_pool_factor", 64))
    bucket_seed = int(cfg.pop("bucket_seed", 0))
    trim_to_multiple_of = int(cfg.pop("trim_to_multiple_of", 8))

    loader_kwargs: Dict[str, Any] = {
        "batch_size": device_batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.pop("pin_memory", True)),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.pop("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.pop("prefetch_factor", 2))
    else:
        cfg.pop("persistent_workers", None)
        cfg.pop("prefetch_factor", None)

    if bucket_by_length:
        # batches of similar length, then trimmed by trim_collate; see
        # dimol/data/samplers.py for why this shifts the loss and what to decide first
        lengths = getattr(dataset, "lengths", None)
        if lengths is None:
            raise TypeError(
                f"bucket_by_length=true needs a dataset exposing .lengths, "
                f"got {type(dataset).__name__}"
            )
        batch_sampler = BucketBatchSampler(
            lengths,
            batch_size=device_batch_size,
            pool_factor=bucket_pool_factor,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=bucket_seed,
            rank=env.rank,
            world_size=env.world_size,
        )
        loader_kwargs.pop("batch_size")
        loader_kwargs.pop("drop_last")
        loader_kwargs["batch_sampler"] = batch_sampler
        loader_kwargs["collate_fn"] = partial(trim_collate, multiple_of=trim_to_multiple_of)
        if cfg:
            raise TypeError(f"unknown keys in loader config: {sorted(cfg)}")
        return DataLoader(dataset, **loader_kwargs), batch_sampler

    sampler: Optional[DistributedSampler] = None
    if env.ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    if cfg:
        raise TypeError(f"unknown keys in loader config: {sorted(cfg)}")

    return DataLoader(dataset, **loader_kwargs), sampler
