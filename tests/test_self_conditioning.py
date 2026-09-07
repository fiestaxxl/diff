"""Self-conditioning: the second input exists, starts inert, and carries at sampling."""
from __future__ import annotations

import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.models.denoiser import DenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionTask

SEQ_LEN = 12
VOCAB = 64


def _model(self_conditioning: bool) -> DiffusionTransformer:
    return DiffusionTransformer(
        TransformerConfig(model_dim=32, emb_dim=8, time_dim=32, num_heads=4,
                          num_text_blocks=1, vocab_size=VOCAB, pad_idx=0,
                          max_pos=SEQ_LEN + 4, self_conditioning=self_conditioning)
    )


def _batch(batch_size: int = 4) -> dict:
    ids = torch.randint(1, VOCAB, (batch_size, SEQ_LEN))
    ids[:, -2:] = 0
    return {"token_ids": ids, "attention_mask": ids != 0}


def _task(**kwargs) -> DiffusionTask:
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    params = dict(pad_idx=0, lambda_grammar=0.0, grammar_enabled=False, seq_len=SEQ_LEN)
    params.update(kwargs)
    return DiffusionTask(path, **params)


def test_the_input_projection_doubles_and_the_extra_half_starts_at_zero():
    plain, conditioned = _model(False), _model(True)
    assert plain.up_proj.in_features == 8
    assert conditioned.up_proj.in_features == 16
    assert torch.count_nonzero(conditioned.up_proj.weight[:, 8:]) == 0


def test_at_initialisation_the_second_input_changes_nothing():
    model = _model(True)
    x = torch.randn(2, SEQ_LEN, 8)
    t = torch.full((2, 1), 0.5)
    without = model(input_embeddings=x, time=t)
    with_estimate = model(input_embeddings=x, time=t, x0_self=torch.randn_like(x))
    assert torch.allclose(without, with_estimate, atol=1e-6)


def test_after_training_the_second_input_matters():
    """Once up_proj has moved, the estimate is actually read."""
    model = _model(True)
    with torch.no_grad():
        model.up_proj.weight[:, 8:].normal_(std=0.1)
    x = torch.randn(2, SEQ_LEN, 8)
    t = torch.full((2, 1), 0.5)
    without = model(input_embeddings=x, time=t)
    with_estimate = model(input_embeddings=x, time=t, x0_self=torch.randn_like(x))
    assert not torch.allclose(without, with_estimate, atol=1e-4)


def test_a_plain_model_refuses_the_second_input():
    model = _model(False)
    try:
        model(input_embeddings=torch.randn(1, SEQ_LEN, 8), time=torch.zeros(1, 1),
              x0_self=torch.randn(1, SEQ_LEN, 8))
    except ValueError as err:
        assert "self_conditioning" in str(err)
    else:
        raise AssertionError("a model without the input must say so")


def test_the_loss_runs_with_and_without_the_extra_pass():
    torch.manual_seed(0)
    model, batch = _model(True), _batch()
    always = _task(self_cond_prob=1.0).compute_loss(model, batch)
    never = _task(self_cond_prob=0.0).compute_loss(model, batch)
    for out in (always, never):
        assert torch.isfinite(out["loss"])
    always["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_the_extra_pass_does_not_leak_gradient():
    """The first estimate is detached: its graph must not be part of the backward."""
    torch.manual_seed(0)
    model, batch = _model(True), _batch()
    out = _task(self_cond_prob=1.0).compute_loss(model, batch)
    out["loss"].backward()
    grad = model.up_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_the_denoiser_carries_the_estimate_between_solver_steps():
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    wrapper = DenoiserModel(_model(True), path)
    assert wrapper._x0_self is None
    x = torch.randn(2, SEQ_LEN, 8)
    wrapper(x, torch.full((2, 1, 1), 0.3))
    assert wrapper._x0_self is not None and wrapper._x0_self.shape == x.shape
    wrapper.reset()
    assert wrapper._x0_self is None


def test_a_plain_model_in_the_denoiser_gets_no_extra_argument():
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    wrapper = DenoiserModel(_model(False), path)
    out = wrapper(torch.randn(2, SEQ_LEN, 8), torch.full((2, 1, 1), 0.3))
    assert torch.isfinite(out).all()
