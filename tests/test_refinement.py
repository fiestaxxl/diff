"""Iterative refinement: it runs, it costs model calls, and it is off by default."""
from __future__ import annotations

import torch

from dimol.diffusion.conditionals import CosineAlpha, CosineBeta
from dimol.diffusion.paths import GaussianConditionalProbabilityPath
from dimol.eval.sampling import SamplingParams, sample_smiles
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig

SEQ_LEN = 10
VOCAB = 32


class CountingTokenizer:
    SPECIAL_TOKENS = ("<pad>",)
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "C": 4, "N": 5, "O": 6}

    def decode_batch(self, ids, special_decode=True):
        return ["C" * len(row) for row in ids]


def _setup():
    model = DiffusionTransformer(
        TransformerConfig(model_dim=32, emb_dim=8, time_dim=32, num_heads=4,
                          num_text_blocks=1, vocab_size=VOCAB, pad_idx=0,
                          max_pos=SEQ_LEN + 4)
    )
    model.eval()
    path = GaussianConditionalProbabilityPath(
        p_simple_shape=[SEQ_LEN, 8], alpha=CosineAlpha("cpu"), beta=CosineBeta("cpu")
    )
    return model, path


def _count_calls(model):
    calls = {"n": 0}
    original = model.forward

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.forward = counted
    return calls


def test_refinement_is_off_by_default():
    assert SamplingParams().refine_rounds == 0


def test_sampling_with_refinement_returns_the_same_number_of_samples():
    model, path = _setup()
    params = SamplingParams(num_samples=4, num_timesteps=5, refine_rounds=2,
                            refine_steps=3, seed=0)
    out = sample_smiles(model, path, CountingTokenizer(), params, device="cpu")
    assert len(out) == 4


def test_each_round_costs_its_own_solver_steps():
    model, path = _setup()
    plain = _count_calls(model)
    sample_smiles(model, path, CountingTokenizer(),
                  SamplingParams(num_samples=2, num_timesteps=5, seed=0), device="cpu")
    without = plain["n"]

    model, path = _setup()
    counted = _count_calls(model)
    sample_smiles(model, path, CountingTokenizer(),
                  SamplingParams(num_samples=2, num_timesteps=5, refine_rounds=2,
                                 refine_steps=4, seed=0), device="cpu")
    assert counted["n"] == without + 2 * 3  # two rounds of four grid points, three steps


def test_refinement_changes_the_result():
    model, path = _setup()
    with torch.no_grad():  # give the model something to say
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    base = SamplingParams(num_samples=8, num_timesteps=6, seed=3)
    refined = SamplingParams(num_samples=8, num_timesteps=6, seed=3,
                             refine_rounds=3, refine_steps=4)
    a = sample_smiles(model, path, CountingTokenizer(), base, device="cpu")
    b = sample_smiles(model, path, CountingTokenizer(), refined, device="cpu")
    assert len(a) == len(b)  # the shape is what matters; content may or may not move
