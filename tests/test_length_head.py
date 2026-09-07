"""Length conditioning: the model reads it, training supplies it, sampling requires it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.models.denoiser import DenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionTask

SEQ_LEN = 12
VOCAB = 64


def _model(**flags) -> DiffusionTransformer:
    return DiffusionTransformer(
        TransformerConfig(model_dim=32, emb_dim=8, time_dim=32, num_heads=4,
                          num_text_blocks=1, vocab_size=VOCAB, pad_idx=0,
                          max_pos=SEQ_LEN + 4, **flags)
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


def test_the_length_embedding_starts_inert():
    model = _model(length_conditioning=True)
    assert torch.count_nonzero(model.length_embedding.weight) == 0
    x, t = torch.randn(2, SEQ_LEN, 8), torch.full((2, 1), 0.5)
    short = model(input_embeddings=x, time=t, length=torch.tensor([3, 3]))
    long = model(input_embeddings=x, time=t, length=torch.tensor([11, 11]))
    assert torch.allclose(short, long, atol=1e-6)


def test_once_trained_the_length_changes_the_prediction():
    """Both paths have to move: adaLN-Zero starts every modulation at zero, so the time
    and length embeddings are inert until the modulation weights are trained."""
    from dimol.models.layers import Modulation

    model = _model(length_conditioning=True)
    with torch.no_grad():
        model.length_embedding.weight.normal_(std=0.5)
        for module in model.modules():
            if isinstance(module, Modulation):
                module.out_layer.weight.normal_(std=0.1)
    x, t = torch.randn(2, SEQ_LEN, 8), torch.full((2, 1), 0.5)
    short = model(input_embeddings=x, time=t, length=torch.tensor([3, 3]))
    long = model(input_embeddings=x, time=t, length=torch.tensor([11, 11]))
    assert not torch.allclose(short, long, atol=1e-4)


def test_a_conditioned_model_refuses_to_run_without_a_length():
    model = _model(length_conditioning=True)
    with pytest.raises(ValueError, match="no length was passed"):
        model(input_embeddings=torch.randn(1, SEQ_LEN, 8), time=torch.zeros(1, 1))


def test_a_plain_model_refuses_a_length():
    model = _model()
    with pytest.raises(ValueError, match="length_conditioning is off"):
        model(input_embeddings=torch.randn(1, SEQ_LEN, 8), time=torch.zeros(1, 1),
              length=torch.tensor([4]))


def test_training_takes_the_length_from_the_attention_mask():
    torch.manual_seed(0)
    model, batch = _model(length_conditioning=True), _batch()
    out = _task().compute_loss(model, batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.length_embedding.weight.grad is not None


def test_training_without_a_mask_is_reported():
    model = _model(length_conditioning=True)
    batch = {"token_ids": _batch()["token_ids"]}
    with pytest.raises(ValueError, match="attention_mask"):
        _task().compute_loss(model, batch)


def test_length_and_self_conditioning_compose():
    torch.manual_seed(0)
    model = _model(length_conditioning=True, self_conditioning=True)
    out = _task(self_cond_prob=1.0).compute_loss(model, _batch())
    assert torch.isfinite(out["loss"])


def test_the_denoiser_forwards_the_length_it_was_given():
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    wrapper = DenoiserModel(_model(length_conditioning=True), path)
    wrapper.length = torch.tensor([5, 5])
    out = wrapper(torch.randn(2, SEQ_LEN, 8), torch.full((2, 1, 1), 0.3))
    assert torch.isfinite(out).all()


def test_sampling_a_conditioned_model_without_a_prior_is_refused():
    from dimol.eval.sampling import SamplingParams, sample_smiles

    class StubTokenizer:
        SPECIAL_TOKENS = ()
        pad_id = bos_id = eos_id = unk_id = 0

        def get_vocab(self):
            return {"C": 1}

    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    with pytest.raises(ValueError, match="length_prior"):
        sample_smiles(_model(length_conditioning=True), path, StubTokenizer(),
                      SamplingParams(num_samples=2), device="cpu")
