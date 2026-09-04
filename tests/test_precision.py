"""Mixed precision: what autocast covers and what stays in fp32."""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionTask

SEQ_LEN, VOCAB = 16, 64


def _setup():
    torch.manual_seed(0)
    cfg = TransformerConfig(
        model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=1,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ_LEN + 4,
    )
    model = DiffusionTransformer(cfg)
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, cfg.emb_dim], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    ids = torch.randint(1, VOCAB, (4, SEQ_LEN))
    ids[:, -4:] = 0
    return model, path, {"token_ids": ids, "attention_mask": ids != 0}


def _task(path, autocast):
    return DiffusionTask(
        path, pad_idx=0, lambda_grammar=0.0, grammar_enabled=False,
        seq_len=SEQ_LEN, autocast=autocast,
    )


def test_readout_runs_in_low_precision_under_autocast() -> None:
    """The (B, L, V) logits are the largest tensor in the step; autocast must cover
    the readout matmul, not only the denoiser forward."""
    model, path, batch = _setup()
    seen = {}
    model.out_proj.register_forward_hook(lambda m, i, o: seen.__setitem__("logits", o))

    _task(path, nullcontext).compute_loss(model, batch)
    assert seen["logits"].dtype is torch.float32

    autocast = partial(torch.autocast, device_type="cpu", dtype=torch.bfloat16)
    _task(path, autocast).compute_loss(model, batch)
    assert seen["logits"].dtype is torch.bfloat16


def test_reductions_stay_fp32_under_autocast() -> None:
    """autocast keeps mse_loss / cross_entropy in fp32 by its own op policy, so the
    loss terms are still accumulated in full precision."""
    model, path, batch = _setup()
    autocast = partial(torch.autocast, device_type="cpu", dtype=torch.bfloat16)
    metrics = _task(path, autocast).compute_loss(model, batch)
    for key in ("loss", "mse_loss", "ce_loss", "mse_loss_t0"):
        assert metrics[key].dtype is torch.float32, key
        assert torch.isfinite(metrics[key]), key


def test_fp32_path_is_untouched() -> None:
    model, path, batch = _setup()
    metrics = _task(path, nullcontext).compute_loss(model, batch)
    assert all(torch.isfinite(v) for v in metrics.values())
