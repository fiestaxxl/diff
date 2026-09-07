"""loss.mask_padding makes the objective independent of how much padding a batch has.

That property is what length bucketing (and any trimming) needs: with the default
mask_padding=False the loss averages over pad positions, so a trimmed batch gives a
different number for the same molecules.
"""
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionTask

STORED, TRIM, VOCAB = 64, 16, 32


@pytest.fixture()
def model_and_path():
    torch.manual_seed(0)
    cfg = TransformerConfig(
        model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=1,
        vocab_size=VOCAB, pad_idx=0, max_pos=STORED + 4,
    )
    model = DiffusionTransformer(cfg).eval()
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[STORED, cfg.emb_dim], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    return model, path


@pytest.fixture()
def batches():
    torch.manual_seed(1)
    ids = torch.zeros(4, STORED, dtype=torch.long)
    for i, real in enumerate((5, 8, 11, 12)):
        ids[i, :real] = torch.randint(1, VOCAB, (real,))
    full = {"token_ids": ids, "attention_mask": ids != 0}
    trimmed = {"token_ids": ids[:, :TRIM], "attention_mask": (ids != 0)[:, :TRIM]}
    return full, trimmed


@pytest.fixture()
def deterministic(monkeypatch):
    """Remove the Monte-Carlo part of the loss: zero noise, fixed t."""
    monkeypatch.setattr(torch, "randn_like", lambda x, **kw: torch.zeros_like(x))
    monkeypatch.setattr(torch, "rand", lambda *shape, **kw: torch.full(shape, 0.5, **kw))


def _loss(model, path, batch, mask_padding: bool) -> float:
    task = DiffusionTask(
        path, pad_idx=0, lambda_grammar=0.0, grammar_enabled=False,
        seq_len=STORED, autocast=nullcontext, mask_padding=mask_padding,
    )
    with torch.no_grad():
        return float(task.compute_loss(model, batch)["loss"])


def test_masked_loss_is_invariant_to_trimming(model_and_path, batches, deterministic) -> None:
    model, path = model_and_path
    full, trimmed = batches
    a = _loss(model, path, full, mask_padding=True)
    b = _loss(model, path, trimmed, mask_padding=True)
    assert a == pytest.approx(b, rel=1e-4), (a, b)


def test_unmasked_loss_depends_on_padding(model_and_path, batches, deterministic) -> None:
    model, path = model_and_path
    full, trimmed = batches
    a = _loss(model, path, full, mask_padding=False)
    b = _loss(model, path, trimmed, mask_padding=False)
    assert abs(a - b) > 0.1 * abs(a), (a, b)


def test_mask_padding_requires_the_mask(model_and_path, batches) -> None:
    model, path = model_and_path
    full, _ = batches
    task = DiffusionTask(
        path, pad_idx=0, lambda_grammar=0.0, grammar_enabled=False,
        seq_len=STORED, autocast=nullcontext, mask_padding=True,
    )
    with pytest.raises(ValueError):
        task.compute_loss(model, {"token_ids": full["token_ids"]})


def test_ce_input_x0_hat_changes_the_gradient_path(model_and_path, batches, deterministic) -> None:
    """With ce_input=x0_hat the cross-entropy is computed on the denoiser output, so its
    gradient reaches the denoiser weights; with x0 it only touches the readout and the
    embedding table."""
    model, path = model_and_path
    full, _ = batches
    for ce_input, expect_denoiser_grad in (("x0", False), ("x0_hat", True)):
        model.zero_grad(set_to_none=True)
        task = DiffusionTask(
            path, pad_idx=0, lambda_grammar=0.0, grammar_enabled=False, lambda_mse=0.0,
            ce_alpha_threshold=0.0,  # every sample reaches the cross-entropy
            seq_len=STORED, autocast=nullcontext, ce_input=ce_input,
        )
        metrics = task.compute_loss(model, full)
        metrics["loss"].backward()
        # up_proj belongs to the denoiser only: it is not touched by the readout or the
        # embedding table. Parameters inside the blocks are useless for this check,
        # because adaLN-Zero starts with a closed residual gate and they see no gradient
        # on the first step either way.
        denoiser = model.up_proj.weight
        has_grad = denoiser.grad is not None and bool(denoiser.grad.abs().sum() > 0)
        assert has_grad is expect_denoiser_grad, (ce_input, has_grad)
