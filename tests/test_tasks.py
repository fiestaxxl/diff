"""Losses: output shape, a stable metric key set, finite values."""
from __future__ import annotations

import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.gpt import SmilesAR, SmilesARConfig
from dimol.training.tasks import ARTask, DiffusionTask

SEQ_LEN = 12
VOCAB = 64


def _tiny_denoiser() -> DiffusionTransformer:
    return DiffusionTransformer(
        TransformerConfig(
            model_dim=32,
            emb_dim=8,
            time_dim=32,
            num_heads=4,
            num_text_blocks=1,
            vocab_size=VOCAB,
            pad_idx=0,
            max_pos=SEQ_LEN + 4,
        )
    )


def _batch(batch_size: int = 4) -> dict:
    ids = torch.randint(1, VOCAB, (batch_size, SEQ_LEN))
    ids[:, -2:] = 0  # padding
    return {"token_ids": ids, "attention_mask": ids != 0}


def _diffusion_task(**kwargs) -> DiffusionTask:
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    params = dict(pad_idx=0, lambda_grammar=0.0, grammar_enabled=False, seq_len=SEQ_LEN)
    params.update(kwargs)
    return DiffusionTask(path, **params)


def test_diffusion_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(0)
    model, task = _tiny_denoiser(), _diffusion_task()
    metrics = task.compute_loss(model, _batch())
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_diffusion_metric_keys_are_stable() -> None:
    """The key set must not depend on the data: otherwise all_reduce would receive
    different metric sets on different ranks and DDP training would hang."""
    torch.manual_seed(0)
    model, task = _tiny_denoiser(), _diffusion_task()
    keys = [set(task.compute_loss(model, _batch(bs)).keys()) for bs in (2, 8, 16)]
    assert keys[0] == keys[1] == keys[2]
    assert {"loss", "mse_loss", "ce_loss", "mse_loss_t0", "grammar_loss", "token_acc"} <= keys[0]


def test_x_regime_predicts_x0() -> None:
    torch.manual_seed(0)
    model, task = _tiny_denoiser(), _diffusion_task(regime="x")
    metrics = task.compute_loss(model, _batch())
    assert torch.isfinite(metrics["loss"])


def test_decoder_pretrain_zeroes_mse_weight() -> None:
    """decoder_pretrain_steps: on the first steps mse contributes zero to the total."""
    torch.manual_seed(0)
    model = _tiny_denoiser()
    task = _diffusion_task(decoder_pretrain_steps=10, lambda_ce=1.0, lambda_mse=1.0)
    m = task.compute_loss(model, _batch(), step=0)
    assert torch.allclose(m["loss"], m["ce_loss"], atol=1e-5)


def test_ar_loss() -> None:
    torch.manual_seed(0)
    model = SmilesAR(
        SmilesARConfig(vocab_size=VOCAB, model_dim=32, n_heads=4, n_layers=1, max_pos=SEQ_LEN + 4)
    )
    task = ARTask(pad_idx=0, seq_len=SEQ_LEN)
    metrics = task.compute_loss(model, _batch())
    assert set(metrics) == {"loss", "token_acc", "ppl"}
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
