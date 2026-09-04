#!/usr/bin/env python3
"""Measure the time of one training step for a given config.

    python scripts/bench_step.py configs/diffusion_chebi.yaml
    python scripts/bench_step.py configs/diffusion_chebi.yaml precision=amp_bf16 compile=true

The batch is synthetic (random token ids of the configured length), so no data
directory is needed and runs are comparable across machines. What is measured is
forward + backward + optimizer step, i.e. exactly what the trainer does between
two step boundaries, with a CUDA sync around it.
"""
from __future__ import annotations

import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_model, build_optimizer, build_path, build_task  # noqa: E402
from dimol.config import load_from_argv, update_batch_size_info  # noqa: E402
from dimol.training.distributed import DistEnv, seed_all  # noqa: E402


class _StubTokenizer:
    """Synthetic vocab so the grammar loss can be timed without a tokenizer file.

    Only the two members build_grammar_tables touches are provided; the table
    values do not affect timing, only the shapes do.
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]

    def __init__(self, vocab_size: int):
        alphabet = "CNOSFPclnos()[]=#@+-123456789"
        vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        i = len(vocab)
        while i < vocab_size:
            vocab[f"{alphabet[i % len(alphabet)]}{i}"] = i
            i += 1
        self._vocab = vocab

    def get_vocab(self) -> dict:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


def _autocast_factory(cfg: DictConfig, device_type: str):
    precision = str(cfg.precision)
    if precision == "fp32" or device_type != "cuda":
        from contextlib import nullcontext

        return nullcontext
    dtype = torch.bfloat16 if precision == "amp_bf16" else torch.float16
    return partial(torch.autocast, device_type=device_type, dtype=dtype)


def main(cfg: DictConfig, n_warmup: int = 3, n_steps: int = 20) -> None:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    env = DistEnv(ddp=False, device=str(device), device_type=device_type)
    cfg = update_batch_size_info(cfg, 1)
    seed_all(int(cfg.seed), tf32=bool(cfg.tf32))

    seq_len = int(cfg.variables.get("seq_len", None) or 208)
    micro_bs = int(cfg.device_train_microbatch_size)
    vocab = int(cfg.model["vocab_size"])

    model = build_model(cfg).to(device)
    if bool(cfg.compile):
        model = torch.compile(model, backend=str(cfg.compile_backend))
    optimizer = build_optimizer(cfg, model, device_type=device_type, verbose=False)

    path = None
    if str(cfg.task) == "diffusion":
        path = build_path(cfg, seq_len=seq_len, emb_dim=int(cfg.model["emb_dim"]), device=device)
    task = build_task(
        cfg,
        path=path,
        tokenizer=_StubTokenizer(vocab),
        seq_len=seq_len,
        device=device,
        autocast=_autocast_factory(cfg, device_type),
    )

    scaler = None
    if str(cfg.precision) == "amp_fp16" and device_type == "cuda":
        from torch.amp import GradScaler

        scaler = GradScaler(device=device_type)

    ids = torch.randint(1, vocab, (micro_bs, seq_len), device=device)
    ids[:, int(seq_len * 0.2):] = 0  # pad tail, as in the real data
    batch = {"token_ids": ids, "attention_mask": ids != 0}

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = task.compute_loss(model, batch)["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    for _ in range(n_warmup):
        one_step()
    if device_type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        one_step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    median = statistics.median(times)
    tokens = micro_bs * seq_len
    print(
        f"device={device} precision={cfg.precision} tf32={cfg.tf32} compile={cfg.compile} "
        f"micro_bs={micro_bs} seq_len={seq_len} params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"step: median {median:.1f} ms | min {min(times):.1f} ms | max {max(times):.1f} ms "
        f"| {tokens / median * 1000:,.0f} tok/s"
    )
    if device_type == "cuda":
        print(f"peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main(load_from_argv())
