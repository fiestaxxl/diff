"""Classifier-free guidance: two forwards, extrapolated, and scale 0 changes nothing."""
from __future__ import annotations

import pytest
import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.eval.sampling import SamplingParams, sample_smiles
from dimol.models.denoiser import DenoiserModel, GuidedDenoiserModel
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.layers import MultiheadCrossAttention

SEQ, VOCAB, TEXT_DIM, TEXT_LEN = 10, 32, 16, 5


class StubTokenizer:
    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "C": 4, "N": 5, "O": 6}

    def decode_batch(self, ids, special_decode=True):
        return ["C" * len(row) for row in ids]


def _model(trained: bool = True) -> DiffusionTransformer:
    model = DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4, text_dim=TEXT_DIM))
    if trained:
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, MultiheadCrossAttention):
                    module.out_layer.weight.normal_(std=0.3)
    model.eval()
    return model


def _path():
    return GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu"))


def _caption(batch: int = 4):
    text = torch.randn(batch, TEXT_LEN, TEXT_DIM)
    mask = torch.ones(batch, TEXT_LEN, dtype=torch.bool)
    return text, mask


def test_scale_zero_is_the_plain_conditional_model():
    torch.manual_seed(0)
    model, path = _model(), _path()
    text, mask = _caption(2)
    x, t = torch.randn(2, SEQ, 8), torch.full((2, 1, 1), 0.4)

    guided = GuidedDenoiserModel(model, path, scale=0.0)
    guided.text, guided.text_mask = text, mask
    plain = DenoiserModel(model, path)
    assert torch.allclose(guided(x, t), plain(x, t, text=text, text_mask=mask), atol=1e-5)


def test_guidance_moves_the_score_away_from_the_unconditional_one():
    torch.manual_seed(0)
    model, path = _model(), _path()
    text, mask = _caption(2)
    x, t = torch.randn(2, SEQ, 8), torch.full((2, 1, 1), 0.4)

    scores = {}
    for scale in (0.0, 1.0, 3.0):
        guided = GuidedDenoiserModel(model, path, scale=scale)
        guided.text, guided.text_mask = text, mask
        scores[scale] = guided(x, t)
    # farther from the conditional prediction as the scale grows
    d1 = (scores[1.0] - scores[0.0]).norm().item()
    d3 = (scores[3.0] - scores[0.0]).norm().item()
    assert d3 > d1 > 0


def test_the_two_forwards_differ_only_in_the_caption_mask():
    """The unconditional branch must be the same weights, not a second model."""
    torch.manual_seed(0)
    model, path = _model(), _path()
    seen = []
    original = model.forward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("text_mask").any().item())
        return original(*args, **kwargs)

    model.forward = spy
    text, mask = _caption(2)
    guided = GuidedDenoiserModel(model, path, scale=1.0)
    guided.text, guided.text_mask = text, mask
    guided(torch.randn(2, SEQ, 8), torch.full((2, 1, 1), 0.4))
    assert seen == [True, False]  # conditional first, then with the caption masked off


def test_sampling_takes_one_caption_per_sample():
    torch.manual_seed(0)
    text, mask = _caption(4)
    out = sample_smiles(_model(), _path(), StubTokenizer(),
                        SamplingParams(num_samples=4, num_timesteps=4, text=text,
                                       text_mask=mask, guidance=1.0, seed=0),
                        device="cpu")
    assert len(out) == 4


def test_a_caption_count_mismatch_is_refused():
    text, mask = _caption(3)
    with pytest.raises(ValueError, match="one caption per sample"):
        sample_smiles(_model(), _path(), StubTokenizer(),
                      SamplingParams(num_samples=5, num_timesteps=4, text=text,
                                     text_mask=mask), device="cpu")


def test_captions_are_sliced_to_match_their_batch():
    """With batch_size below num_samples the caption rows must follow the molecules."""
    torch.manual_seed(0)
    text, mask = _caption(4)
    out = sample_smiles(_model(), _path(), StubTokenizer(),
                        SamplingParams(num_samples=4, batch_size=2, num_timesteps=4,
                                       text=text, text_mask=mask, guidance=0.5, seed=0),
                        device="cpu")
    assert len(out) == 4
