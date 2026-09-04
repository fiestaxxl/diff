"""A single training loop shared by all tasks (diffusion / ar).

It replaces the three trainers of the old ``dimol/diffusion/trainers.py`` (two of
which were broken). The step logic is preserved: grad accumulation with the loss
divided by the number of microbatches, DDP gradient sync disabled on intermediate
microbatches, unscale -> clip -> step for fp16, and the lr written into the
optimizer by hand before every step.

What is new compared to the old loop (infrastructure, not math):
* durations and intervals come from the config in ba/ep instead of "every Nth epoch";
* checkpoints carry optimizer/scaler/step state, so autoresume works;
* every step logs loss, accuracy and tokens-per-sec.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig

from dimol.config import Duration, duration_hit, duration_reached
from dimol.training import checkpoint as ckpt_utils
from dimol.training.checkpoint import TrainerState
from dimol.training.distributed import DistEnv, all_reduce_min, reduce_metrics
from dimol.training.logging import (
    MultiLogger,
    format_step_line,
    namespaced_metrics,
)
from dimol.training.optim import LRScheduler
from dimol.training.tasks import Task


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        cfg: DictConfig,
        dist_env: DistEnv,
        logger: MultiLogger,
        train_loader: torch.utils.data.DataLoader,
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
        train_sampler: Optional[Any] = None,
        eval_sampler: Optional[Any] = None,
        sample_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        scaler: Optional[Any] = None,
    ):
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.env = dist_env
        self.logger = logger
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.sample_fn = sample_fn
        self.scaler = scaler

        self.state = TrainerState()
        self._last_saved: Optional[tuple[int, int]] = None
        self.grad_accum_steps = int(cfg.device_train_grad_accum or 1)
        self.micro_per_epoch = self._loader_len(train_loader)
        if self.micro_per_epoch == 0:
            raise ValueError(
                "train_loader is empty: the batch is larger than the dataset with "
                "drop_last=true. Lower device_train_microbatch_size or set "
                "train_loader.drop_last=false"
            )
        self.steps_per_epoch = (
            max(1, self.micro_per_epoch // self.grad_accum_steps) if self.micro_per_epoch else None
        )
        self.clip_cfg = cfg.algorithms.gradient_clipping
        if (
            self.clip_cfg.clipping_threshold is not None
            and str(self.clip_cfg.clipping_type) != "norm"
        ):
            raise ValueError(
                f"algorithms.gradient_clipping.clipping_type={self.clip_cfg.clipping_type} "
                "is not supported (available: norm)"
            )
        self.optimizer_name = type(optimizer).__name__

        self.ckpt_root = (
            ckpt_utils.checkpoint_dir(cfg.save_folder, cfg.run_name) if cfg.save_folder else None
        )
        self.max_steps = self._max_steps()
        self.tokens_per_step = int(cfg.global_train_batch_size or 0) * int(getattr(task, "seq_len", 0))

    # ------------------------------------------------------------------
    @staticmethod
    def _loader_len(loader: Optional[torch.utils.data.DataLoader]) -> Optional[int]:
        try:
            return len(loader)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _max_steps(self) -> Optional[int]:
        d = Duration.parse(self.cfg.max_duration)
        if d.unit == "ba":
            return int(d.value)
        return d.in_batches(self.steps_per_epoch) if self.steps_per_epoch else None

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.env.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def maybe_resume(self) -> None:
        cfg = self.cfg
        path = None
        if cfg.load_path:
            path = cfg.load_path
        elif cfg.autoresume and self.ckpt_root is not None:
            latest = ckpt_utils.latest_checkpoint(self.ckpt_root)
            path = str(latest) if latest else None

        if path is None:
            return

        self.state = ckpt_utils.load_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            weights_only=bool(cfg.load_weights_only),
            map_location=self.env.device,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        model, optimizer = self.model, self.optimizer

        if self.env.is_master:
            print(
                f"[trainer] max_duration={cfg.max_duration} "
                f"(max_steps={self.max_steps}) | global_bs={cfg.global_train_batch_size} "
                f"| micro_bs={cfg.device_train_microbatch_size} | accum={self.grad_accum_steps} "
                f"| world={self.env.world_size} | precision={cfg.precision}",
                flush=True,
            )

        if str(cfg.eval_before_train) == "regular" and self.eval_loader is not None:
            self._run_eval()

        if duration_reached(cfg.max_duration, self.state.step, self.state.epoch):
            if self.env.is_master:
                print(
                    f"[trainer] max_duration={cfg.max_duration} already reached at "
                    f"ep{self.state.epoch}-ba{self.state.step}: nothing to train"
                )
            return

        optimizer.zero_grad(set_to_none=True)
        model.train()

        micro_step = 0
        accum_metrics: Dict[str, torch.Tensor] = {}
        t0 = time.time()
        epoch = self.state.epoch
        stop = False

        while not stop:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            micro_in_epoch = 0
            t0 = time.time()  # do not charge the loader restart to the first step

            for batch in self.train_loader:
                micro_in_epoch += 1
                batch = self._to_device(batch)

                if self.env.ddp:
                    model.require_backward_grad_sync = micro_step == self.grad_accum_steps - 1

                metrics = self.task.compute_loss(model, batch, step=self.state.step)
                loss = metrics["loss"] / self.grad_accum_steps  # scale down for averaging

                for k, v in metrics.items():
                    if not torch.is_tensor(v):
                        v = torch.as_tensor(float(v), device=loss.device)
                    accum_metrics[k] = (
                        accum_metrics.get(k, 0.0) + v.detach() / self.grad_accum_steps
                    )

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1
                if micro_step % self.grad_accum_steps != 0:
                    continue  # accumulate more

                # ---- optimizer step boundary ----
                grad_norm = self._clip_gradients()
                lr = self.scheduler(self.state.step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

                if self.env.device_type == "cuda" and bool(cfg.sync_each_step):
                    torch.cuda.synchronize()  # wait for the GPU before timing

                dt = time.time() - t0
                reduced = reduce_metrics(accum_metrics, ddp=self.env.ddp)
                accum_metrics = {}

                self.state.step += 1
                self.state.epoch = epoch
                self._log_train_step(reduced, lr=lr, grad_norm=grad_norm, dt=dt)

                # ---- periodic events ----
                if self.eval_loader is not None and duration_hit(
                    cfg.eval_interval, self.state.step, self.steps_per_epoch
                ):
                    self._run_eval()
                    model.train()

                if (
                    self.sample_fn is not None
                    and cfg.sampling.enabled
                    and duration_hit(cfg.sampling.interval, self.state.step, self.steps_per_epoch)
                ):
                    self._run_sampling()
                    model.train()

                if duration_hit(cfg.save_interval, self.state.step, self.steps_per_epoch):
                    self._save()

                if duration_reached(cfg.max_duration, self.state.step, epoch):
                    stop = True
                    break

                t0 = time.time()

            if micro_in_epoch == 0:
                raise RuntimeError(
                    "train_loader yielded no batches for a whole epoch: training cannot advance"
                )

            if micro_step != 0:
                # the epoch ended mid-accumulation: the remainder is dropped, otherwise
                # one step would mix two epochs and, under DDP, part of the gradients
                # would stay unsynchronized (require_backward_grad_sync=False)
                if self.env.is_master:
                    print(
                        f"[trainer] dropped {micro_step} microbatches at the boundary of "
                        f"epoch {epoch}: len(train_loader) is not divisible by grad accum "
                        f"({self.micro_per_epoch} % {self.grad_accum_steps} != 0)"
                    )
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                accum_metrics = {}

            if not stop:
                epoch += 1
                self.state.epoch = epoch
                if duration_reached(cfg.max_duration, self.state.step, epoch):
                    stop = True

        if cfg.save_last:
            self._save()
        if self.env.is_master:
            print(f"[trainer] training finished at ep{self.state.epoch}-ba{self.state.step}")

    # ------------------------------------------------------------------
    def _clip_gradients(self) -> Optional[torch.Tensor]:
        threshold = self.clip_cfg.clipping_threshold
        if threshold is None:
            return None
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)  # clipping must see unscaled gradients
        # kept as a tensor: it is turned into a float together with the metrics,
        # so the step costs one device-to-host copy instead of one per value
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), threshold)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_floats(
        metrics: Dict[str, torch.Tensor], extra: Optional[torch.Tensor] = None
    ) -> tuple[Dict[str, float], Optional[float]]:
        """Move every scalar of the step to the host with a single copy.

        On CUDA each float(tensor) is a blocking synchronization, and a diffusion
        step produces about fifteen metrics plus the gradient norm.
        """
        keys = list(metrics)
        tensors = [metrics[k].detach().reshape(1).float() for k in keys]
        if extra is not None:
            tensors.append(extra.detach().reshape(1).float())
        if not tensors:
            return {}, None
        values = torch.cat(tensors).cpu().tolist()
        out = {k: values[i] for i, k in enumerate(keys)}
        return out, (values[-1] if extra is not None else None)

    # ------------------------------------------------------------------
    def _log_train_step(
        self,
        metrics: Dict[str, torch.Tensor],
        *,
        lr: float,
        grad_norm: Optional[torch.Tensor],
        dt: float,
    ) -> None:
        if not self.env.is_master:
            return
        if not duration_hit(self.cfg.console_log_interval, self.state.step, self.steps_per_epoch):
            log_console = False
        else:
            log_console = bool(self.cfg.log_to_console)

        scalars, grad_norm_value = self._to_floats(metrics, grad_norm)
        tokens_per_sec = self.tokens_per_step / dt if dt > 0 and self.tokens_per_step else None
        samples_per_sec = (
            float(self.cfg.global_train_batch_size) / dt
            if dt > 0 and self.cfg.global_train_batch_size
            else None
        )
        mem_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if self.env.device_type == "cuda" else None
        )

        if log_console:
            self.logger.log_line(
                format_step_line(
                    split="train",
                    step=self.state.step,
                    max_steps=self.max_steps,
                    epoch=self.state.epoch,
                    metrics=scalars,
                    lr=lr,
                    grad_norm=grad_norm_value,
                    tokens_per_sec=tokens_per_sec,
                    samples_per_sec=samples_per_sec,
                    dt_ms=dt * 1000,
                    mem_gb=mem_gb,
                )
            )

        self.logger.log_metrics(
            namespaced_metrics(
                scalars,
                "train",
                lr=lr,
                optimizer_name=self.optimizer_name,
                grad_norm=grad_norm_value,
                tokens_per_sec=tokens_per_sec,
                samples_per_sec=samples_per_sec,
                dt_s=dt,
                mem_gb=mem_gb,
                epoch=self.state.epoch,
            ),
            step=self.state.step,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_eval(self) -> Dict[str, float]:
        assert self.eval_loader is not None
        cfg = self.cfg
        self.model.eval()
        if self.eval_sampler is not None:
            self.eval_sampler.set_epoch(self.state.epoch)

        limit = int(cfg.eval_subset_num_batches)
        sums: Dict[str, torch.Tensor] = {}
        n_batches = 0
        for batch in self.eval_loader:
            if 0 < limit <= n_batches:
                break
            batch = self._to_device(batch)
            metrics = self.task.compute_loss(self.model, batch, step=self.state.step)
            for k, v in metrics.items():
                if not torch.is_tensor(v):
                    v = torch.as_tensor(float(v), device=self.env.device)
                sums[k] = sums.get(k, torch.zeros((), device=self.env.device)) + v.detach()
            n_batches += 1

        # if any rank saw no batches, skip the reduction: otherwise one rank would
        # enter all_reduce and another would not (a hang until the dist timeout)
        min_batches = all_reduce_min(n_batches, ddp=self.env.ddp, device=self.env.device)
        if min_batches == 0:
            if self.env.is_master:
                self.logger.log_line(
                    f"[eval ba {self.state.step}] skipped: some ranks have no batches "
                    f"(eval_loader too small for world_size={self.env.world_size})"
                )
            self.model.train()
            return {}

        local_mean = {k: v / max(n_batches, 1) for k, v in sums.items()}
        reduced = reduce_metrics(local_mean, ddp=self.env.ddp)
        scalars, _ = self._to_floats(reduced)

        if self.env.is_master:
            self.logger.log_line(
                format_step_line(
                    split="eval",
                    step=self.state.step,
                    max_steps=self.max_steps,
                    epoch=self.state.epoch,
                    metrics=scalars,
                )
            )
            self.logger.log_metrics(namespaced_metrics(scalars, "eval"), step=self.state.step)

        self.model.train()
        return scalars

    # ------------------------------------------------------------------
    def _run_sampling(self) -> None:
        assert self.sample_fn is not None
        self.model.eval()
        try:
            result = self.sample_fn(self.state.step)
        finally:
            self.model.train()

        metrics = {k: float(v) for k, v in (result.get("metrics") or {}).items()}
        if metrics:
            tensor_metrics = {
                k: torch.as_tensor(v, device=self.env.device, dtype=torch.float32)
                for k, v in metrics.items()
            }
            reduced = reduce_metrics(tensor_metrics, ddp=self.env.ddp)
            metrics, _ = self._to_floats(reduced)

        if self.env.is_master:
            if metrics:
                self.logger.log_line(
                    format_step_line(
                        split="sample",
                        step=self.state.step,
                        max_steps=self.max_steps,
                        epoch=self.state.epoch,
                        metrics=metrics,
                    )
                )
                self.logger.log_metrics(
                    namespaced_metrics(metrics, "sample"), step=self.state.step
                )
            examples = result.get("examples") or []
            if examples:
                text = "\n".join(str(s) for s in examples[: int(self.cfg.sampling.log_examples)])
                self.logger.log_text("samples", text, step=self.state.step)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self.ckpt_root is None:
            return
        key = (self.state.epoch, self.state.step)
        if self._last_saved == key:
            return  # already saved (e.g. save_interval coincided with the end of training)
        self._last_saved = key
        if self.env.ddp:
            dist.barrier()
        if self.env.is_master:
            path = ckpt_utils.save_checkpoint(
                self.ckpt_root,
                self.model,
                self.state,
                optimizer=self.optimizer,
                scaler=self.scaler,
                weights_only=bool(self.cfg.save_weights_only),
                keep_last=int(self.cfg.save_num_checkpoints_to_keep),
            )
            print(f"[ckpt] saved: {path}", flush=True)
        if self.env.ddp:
            dist.barrier()
