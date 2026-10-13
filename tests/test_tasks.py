"""Losses: output shape, a stable metric key set, finite values."""
from __future__ import annotations

import pytest
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


def test_pad_weight_one_matches_the_default_objective():
    """pad_weight=1.0 must be the original loss, so the knob is safe to leave in."""
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch()
    torch.manual_seed(7)
    base = _diffusion_task().compute_loss(model, batch)["loss"]
    torch.manual_seed(7)
    weighted = _diffusion_task(pad_weight=1.0).compute_loss(model, batch)["loss"]
    assert torch.allclose(base, weighted, atol=1e-6)


def test_pad_weight_zero_matches_the_loss_mask():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch()
    torch.manual_seed(7)
    masked = _diffusion_task(mask_padding="loss").compute_loss(model, batch)["mse_loss"]
    torch.manual_seed(7)
    zero = _diffusion_task(pad_weight=0.0).compute_loss(model, batch)["mse_loss"]
    assert torch.allclose(masked, zero, atol=1e-6)


def test_pad_weight_interpolates_between_them():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch()
    out = {}
    for w in (0.0, 0.5, 1.0):
        torch.manual_seed(7)
        out[w] = float(_diffusion_task(pad_weight=w).compute_loss(model, batch)["mse_loss"])
    assert min(out[0.0], out[1.0]) <= out[0.5] <= max(out[0.0], out[1.0])


def test_pad_weight_needs_a_mask():
    model = _tiny_denoiser()
    batch = {"token_ids": _batch()["token_ids"]}
    try:
        _diffusion_task(pad_weight=0.5).compute_loss(model, batch)
    except ValueError as err:
        assert "attention_mask" in str(err)
    else:
        raise AssertionError("a missing attention_mask must be reported")


def test_min_snr_with_a_huge_gamma_is_the_plain_mean():
    """With gamma above every SNR in the batch the weights are all 1."""
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch()
    torch.manual_seed(7)
    base = _diffusion_task().compute_loss(model, batch)["mse_loss"]
    torch.manual_seed(7)
    capped = _diffusion_task(min_snr_gamma=1e12).compute_loss(model, batch)["mse_loss"]
    assert torch.allclose(base, capped, atol=1e-4)


def test_min_snr_downweights_the_easy_low_noise_steps():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch(batch_size=32)
    torch.manual_seed(7)
    base = float(_diffusion_task().compute_loss(model, batch)["mse_loss"])
    torch.manual_seed(7)
    capped = float(_diffusion_task(min_snr_gamma=1.0).compute_loss(model, batch)["mse_loss"])
    assert capped != base
    for value in (base, capped):
        assert value == value and value >= 0.0


def test_min_snr_and_pad_weight_compose():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch()
    out = _diffusion_task(min_snr_gamma=5.0, pad_weight=0.3).compute_loss(model, batch)
    assert torch.isfinite(out["loss"])


def test_ce_ignores_padding_by_default_and_can_include_it():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch(batch_size=16)
    torch.manual_seed(7)
    without = float(_diffusion_task().compute_loss(model, batch)["ce_loss"])
    torch.manual_seed(7)
    with_pad = float(_diffusion_task(ce_include_pad=True).compute_loss(model, batch)["ce_loss"])
    assert without != with_pad


def test_including_padding_teaches_the_readout_to_emit_it():
    """The gradient on the padding row of the readout must be non-zero only then."""
    torch.manual_seed(0)
    batch = _batch(batch_size=16)
    grads = {}
    for include in (False, True):
        model = _tiny_denoiser()
        torch.manual_seed(0)
        for p in model.parameters():
            p.grad = None
        torch.manual_seed(7)
        task = _diffusion_task(ce_include_pad=include, lambda_mse=0.0)
        task.compute_loss(model, batch)["loss"].backward()
        row = model.out_proj.weight.grad[0] if model.out_proj.weight.grad is not None else None
        grads[include] = 0.0 if row is None else float(row.abs().sum())
    assert grads[True] > grads[False]


def test_the_threshold_gate_selects_by_alpha():
    task = _diffusion_task()
    alpha = torch.tensor([0.1, 0.85, 0.5, 0.95])
    mask = task._gate(alpha, 0.8)
    assert mask.tolist() == [False, True, False, True]


def test_the_topk_gate_takes_a_fixed_share_whatever_the_alphas_are():
    task = _diffusion_task(gate_mode="topk", gate_fraction=0.5)
    for alpha in (torch.tensor([0.1, 0.2, 0.3, 0.4]),
                  torch.tensor([0.9, 0.95, 0.99, 0.999]),
                  torch.rand(4)):
        mask = task._gate(alpha, 0.8)
        assert int(mask.sum()) == 2, alpha


def test_the_topk_gate_picks_the_highest_alphas():
    task = _diffusion_task(gate_mode="topk", gate_fraction=0.5)
    mask = task._gate(torch.tensor([0.1, 0.9, 0.2, 0.8]), 0.8)
    assert mask.tolist() == [False, True, False, True]


def test_the_topk_gate_always_selects_at_least_one_sample():
    task = _diffusion_task(gate_mode="topk", gate_fraction=0.01)
    assert int(task._gate(torch.rand(4), 0.8).sum()) == 1


def test_a_bad_gate_setting_is_reported():
    for kwargs in ({"gate_mode": "sometimes"}, {"gate_fraction": 0.0},
                   {"gate_fraction": 1.5}):
        try:
            _diffusion_task(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} should have been rejected")


def test_the_loss_runs_under_the_deterministic_gate():
    torch.manual_seed(0)
    model, batch = _tiny_denoiser(), _batch(batch_size=8)
    out = _diffusion_task(gate_mode="topk", gate_fraction=0.5).compute_loss(model, batch)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["ce_loss"])


# ---------------------------------------- the caption encoder branch of the same loss

TEXT_DIM, TEXT_LEN, CAPTION_VOCAB = 32, 7, 48


def _tiny_encoder(path):
    from transformers import BertConfig, BertModel

    BertModel(BertConfig(
        vocab_size=CAPTION_VOCAB, hidden_size=TEXT_DIM, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=2 * TEXT_DIM,
        max_position_embeddings=4 * TEXT_LEN), add_pooling_layer=False).save_pretrained(path)
    return path


def _text_config(**overrides) -> TransformerConfig:
    params = dict(model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=1,
                  vocab_size=VOCAB, pad_idx=0, max_pos=SEQ_LEN + 4, text_dim=TEXT_DIM)
    params.update(overrides)
    return TransformerConfig(**params)


def _encoder_denoiser(tmp_path) -> DiffusionTransformer:
    pytest.importorskip("transformers")
    return DiffusionTransformer(
        _text_config(text_encoder=str(_tiny_encoder(tmp_path / "enc"))))


def _caption_batch(batch_size: int = 4) -> dict:
    batch = _batch(batch_size)
    attn = torch.zeros(batch_size, TEXT_LEN, dtype=torch.bool)
    attn[:, :5] = True
    batch["caption_ids"] = torch.randint(1, CAPTION_VOCAB, (batch_size, TEXT_LEN)) * attn
    batch["caption_attn"] = attn
    return batch


def _cached_batch(batch_size: int = 4) -> dict:
    batch = _batch(batch_size)
    mask = torch.zeros(batch_size, TEXT_LEN, dtype=torch.bool)
    mask[:, :5] = True
    batch["text"] = torch.randn(batch_size, TEXT_LEN, TEXT_DIM)
    batch["text_mask"] = mask
    return batch


def _capture(model) -> dict:
    """Record the keyword arguments the loss hands the model."""
    seen, original = {}, model.forward
    model.forward = lambda **kwargs: (seen.update(kwargs), original(**kwargs))[1]
    return seen


def test_caption_dropout_blanks_rows_of_the_cross_attention_mask_only(tmp_path):
    """An all-False row into BERT is a softmax over nothing, so the dropout cannot land on
    caption_attn; what must stop seeing the caption is the cross-attention above it."""
    torch.manual_seed(0)
    model = _encoder_denoiser(tmp_path)
    model.train()
    batch = _caption_batch(batch_size=64)
    seen = _capture(model)
    torch.manual_seed(3)
    _diffusion_task(caption_dropout=0.5).compute_loss(model, batch)

    text_mask, caption_attn = seen["text_mask"], seen["caption_attn"]
    dropped = ~text_mask.any(1)
    assert 0 < int(dropped.sum()) < 64, "the dropout fired on every row or on none"
    assert torch.equal(caption_attn, batch["caption_attn"])
    assert bool(caption_attn.any(1).all())
    assert torch.equal(text_mask[~dropped], caption_attn[~dropped])


def test_the_caption_branch_draws_the_same_randomness_as_the_cached_one(tmp_path):
    """Same shape, same place in the sequence: this is what makes the window arm a
    controlled comparison against the cached-states run rather than a second change."""
    torch.manual_seed(0)
    live = _encoder_denoiser(tmp_path)
    cached = DiffusionTransformer(_text_config())
    live.train()
    cached.train()
    seen_live, seen_cached = _capture(live), _capture(cached)
    task = _diffusion_task(caption_dropout=0.3)
    # built before the seed: making a batch draws, and the comparison is about what the
    # loss draws, not about what the fixtures do
    caption_batch, cached_batch = _caption_batch(batch_size=32), _cached_batch(batch_size=32)

    torch.manual_seed(11)
    task.compute_loss(live, caption_batch)
    torch.manual_seed(11)
    task.compute_loss(cached, cached_batch)
    assert torch.equal(~seen_live["text_mask"].any(1), ~seen_cached["text_mask"].any(1))


def test_an_encoder_model_fed_cached_states_is_refused(tmp_path):
    """A silent fallback here leaves the encoder without a gradient, and a parameter with
    no gradient hangs DDP at find_unused_parameters=false on the next all-reduce."""
    model = _encoder_denoiser(tmp_path)
    with pytest.raises(ValueError, match="no 'caption_ids'"):
        _diffusion_task().compute_loss(model, _cached_batch())


def test_caption_ids_against_a_model_with_no_encoder_are_refused():
    pytest.importorskip("transformers")
    model = DiffusionTransformer(_text_config())
    with pytest.raises(ValueError, match="no caption encoder"):
        _diffusion_task().compute_loss(model, _caption_batch())


def test_the_loss_runs_and_trains_the_encoder(tmp_path):
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    # the window arm's shape: the top block takes gradients and nothing below it does
    model = DiffusionTransformer(_text_config(
        text_encoder=str(_tiny_encoder(tmp_path / "enc")),
        text_encoder_trainable_blocks=1))
    out = _diffusion_task().compute_loss(model, _caption_batch())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    top = model.text_encoder.encoder.layer[-1]
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in top.parameters())
    assert all(p.grad is None for p in model.text_encoder.embeddings.parameters())
