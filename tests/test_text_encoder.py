"""SciBERT inside the denoiser: the same conditioning, computed in a different place.

The frozen path reads caption states from disk. This path takes caption ids and runs the
encoder in the forward pass so that gradients reach it. The two must be the same
computation, because every number this project has recorded came from the frozen one and
a silent divergence would invalidate the record rather than extend it. The tests below pin
that equivalence, the zero-initialised graft surviving the new input, the frozen path's
state dict staying byte-compatible, and the freezing arithmetic — which decides what DDP
waits for a gradient on and what the optimizer ever sees.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig  # noqa: E402
from dimol.models.layers import MultiheadCrossAttention  # noqa: E402

SEQ, VOCAB, EMB, HIDDEN, LAYERS, TEXT_LEN, CAPTION_VOCAB = 12, 64, 8, 32, 4, 7, 48


def tiny_encoder(path: Path) -> Path:
    """A four-layer random BERT standing in for SciBERT's 110M parameters."""
    from transformers import BertConfig, BertModel

    BertModel(BertConfig(
        vocab_size=CAPTION_VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=4, intermediate_size=2 * HIDDEN,
        max_position_embeddings=4 * TEXT_LEN), add_pooling_layer=False).save_pretrained(path)
    return path


def _config(**overrides) -> TransformerConfig:
    params = dict(model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
                  vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4, text_dim=HIDDEN,
                  self_conditioning=True, length_conditioning=True)
    params.update(overrides)
    return TransformerConfig(**params)


def _inputs(batch: int = 3) -> dict:
    return dict(input_embeddings=torch.randn(batch, SEQ, EMB),
                time=torch.full((batch, 1), 0.5),
                length=torch.full((batch,), 6, dtype=torch.long))


def _caption(batch: int = 3, real: int = 4):
    ids = torch.randint(1, CAPTION_VOCAB, (batch, TEXT_LEN))
    attn = torch.zeros(batch, TEXT_LEN, dtype=torch.bool)
    attn[:, :real] = True
    ids = ids * attn  # padding ids are zero, as the stored arrays have them
    return ids, attn


def _wake_the_cross_attention(model) -> None:
    """Take the graft out of its zero-initialised state, or the test measures nothing."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MultiheadCrossAttention):
                module.out_layer.weight.normal_(std=0.2)


# ---------------------------------------------------------------- the equivalence


def test_encoding_in_the_loop_equals_encoding_offline(tmp_path):
    """The whole claim: this changes where the states are computed, not what they are."""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")))).eval()
    _wake_the_cross_attention(model)
    inputs, (ids, attn) = _inputs(), _caption()

    with torch.no_grad():
        states = model.text_encoder(input_ids=ids,
                                    attention_mask=attn.long()).last_hidden_state
        live = model(**inputs, caption_ids=ids, caption_attn=attn)
        cached = model(**inputs, text=states, text_mask=attn)
    assert torch.equal(live, cached)


def test_the_text_path_is_still_inert_with_an_encoder_attached(tmp_path):
    """The graft property has to survive: a checkpoint loaded into this shape is unmoved."""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")))).eval()
    inputs, (ids, attn) = _inputs(), _caption()
    with torch.no_grad():
        assert torch.allclose(model(**inputs),
                              model(**inputs, caption_ids=ids, caption_attn=attn),
                              atol=1e-6)


def test_the_caption_reaches_the_model_once_the_graft_is_trained(tmp_path):
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")))).eval()
    _wake_the_cross_attention(model)
    inputs, (ids, attn) = _inputs(), _caption()
    other = torch.randint(1, CAPTION_VOCAB, ids.shape) * attn
    with torch.no_grad():
        assert not torch.allclose(model(**inputs, caption_ids=ids, caption_attn=attn),
                                  model(**inputs, caption_ids=other, caption_attn=attn),
                                  atol=1e-4)


# ---------------------------------------------------------------- the frozen path


def test_a_model_without_an_encoder_carries_no_encoder_weights():
    """Nothing about the frozen checkpoint's state dict may move; the record depends on it."""
    model = DiffusionTransformer(_config())
    assert model.text_encoder is None
    assert not [k for k in model.state_dict() if k.startswith("text_encoder.")]


def test_a_config_predating_these_fields_still_loads_at_the_old_behaviour():
    """from_dict filters unknown keys, so an old config.json must read back unchanged."""
    torch.manual_seed(0)
    model = DiffusionTransformer(_config()).eval()
    old = {k: v for k, v in model.config.to_dict().items()
           if not k.startswith("text_encoder")}
    again = DiffusionTransformer(TransformerConfig.from_dict(old)).eval()
    again.load_state_dict(model.state_dict())

    assert again.config.text_encoder == ""
    assert again.config.text_encoder_trainable_blocks == 0
    assert again.config.text_encoder_train_embeddings is False
    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(model(**inputs), again(**inputs))


def test_caption_ids_without_an_encoder_are_refused():
    model = DiffusionTransformer(_config())
    ids, attn = _caption()
    with pytest.raises(ValueError, match="text_encoder is unset"):
        model(**_inputs(), caption_ids=ids, caption_attn=attn)


def test_states_and_caption_ids_together_are_refused(tmp_path):
    """Two sources of one conditioning signal: one of them would be dropped in silence."""
    model = DiffusionTransformer(_config(text_encoder=str(tiny_encoder(tmp_path / "enc"))))
    ids, attn = _caption()
    with pytest.raises(ValueError, match="both passed"):
        model(**_inputs(), text=torch.randn(3, TEXT_LEN, HIDDEN), text_mask=attn,
              caption_ids=ids, caption_attn=attn)


def test_a_caption_mask_without_its_ids_is_refused(tmp_path):
    model = DiffusionTransformer(_config(text_encoder=str(tiny_encoder(tmp_path / "enc"))))
    _, attn = _caption()
    with pytest.raises(ValueError, match="without caption_ids"):
        model(**_inputs(), caption_attn=attn)


# ---------------------------------------------------------------- freezing


def _trainable(model) -> set:
    return {name for name, p in model.named_parameters()
            if name.startswith("text_encoder.") and p.requires_grad}


def test_only_the_top_blocks_are_trainable(tmp_path):
    """What is frozen here is what never enters the optimizer, and never stalls DDP."""
    encoder = tiny_encoder(tmp_path / "enc")
    model = DiffusionTransformer(
        _config(text_encoder=str(encoder), text_encoder_trainable_blocks=2))
    names = _trainable(model)
    assert names
    assert all(n.startswith(("text_encoder.encoder.layer.2.",
                             "text_encoder.encoder.layer.3.")) for n in names), names
    assert not any("embeddings" in n or "pooler" in n for n in names)


def test_zero_blocks_leaves_the_encoder_completely_frozen(tmp_path):
    """The control arm: a live encoder that cannot move is a comparison, not a change."""
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")),
                text_encoder_trainable_blocks=0))
    assert _trainable(model) == set()


def test_minus_one_trains_every_block_and_still_not_the_embeddings(tmp_path):
    """The table is a fifth of the encoder and overfits 25,574 pairs first."""
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")),
                text_encoder_trainable_blocks=-1))
    names = _trainable(model)
    assert {n.split(".")[3] for n in names} == {str(i) for i in range(LAYERS)}
    assert not any("embeddings" in n for n in names)


def test_the_embeddings_can_be_asked_for_explicitly(tmp_path):
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")),
                text_encoder_trainable_blocks=0, text_encoder_train_embeddings=True))
    names = _trainable(model)
    assert names and all("embeddings" in n for n in names)


def test_a_frozen_encoder_keeps_its_pretrained_weights(tmp_path):
    """self.apply(_init_params) would redraw every one of them from Normal(0, 0.02)."""
    from transformers import AutoModel

    path = tiny_encoder(tmp_path / "enc")
    reference = AutoModel.from_pretrained(str(path), add_pooling_layer=False)
    model = DiffusionTransformer(_config(text_encoder=str(path)))
    for name, param in reference.named_parameters():
        assert torch.equal(param, dict(model.text_encoder.named_parameters())[name]), name


# ---------------------------------------------------------------- determinism


def test_two_forwards_in_training_mode_agree(tmp_path):
    """Dropout off at load, or the frozen-encoder arm is not a control and the export drifts."""
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc"))))
    _wake_the_cross_attention(model)
    model.train()
    inputs, (ids, attn) = _inputs(), _caption()
    with torch.no_grad():
        first = model(**inputs, caption_ids=ids, caption_attn=attn)
        second = model(**inputs, caption_ids=ids, caption_attn=attn)
    assert torch.equal(first, second)


def test_padded_caption_positions_do_not_reach_the_model(tmp_path):
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")))).eval()
    _wake_the_cross_attention(model)
    inputs, (ids, attn) = _inputs(), _caption(real=4)
    moved = ids.clone()
    moved[:, 4:] = torch.randint(1, CAPTION_VOCAB, moved[:, 4:].shape)
    with torch.no_grad():
        before = model(**inputs, caption_ids=ids, caption_attn=attn)
        after = model(**inputs, caption_ids=moved, caption_attn=attn)
    assert torch.allclose(before, after, atol=1e-5)


# ---------------------------------------------------------------- construction


def test_an_encoder_of_the_wrong_width_is_refused_at_construction(tmp_path):
    """Caught here rather than as a matmul shape error nine layers into the first step."""
    with pytest.raises(ValueError, match="hidden size"):
        DiffusionTransformer(_config(text_encoder=str(tiny_encoder(tmp_path / "enc")),
                                     text_dim=HIDDEN + 16))


def test_the_encoder_survives_a_save_and_load(tmp_path):
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _config(text_encoder=str(tiny_encoder(tmp_path / "enc")),
                text_encoder_trainable_blocks=2)).eval()
    _wake_the_cross_attention(model)
    with torch.no_grad():  # move the encoder, so the round trip has something to carry
        for param in model.text_encoder.encoder.layer[-1].parameters():
            param.add_(torch.randn_like(param) * 0.01)
    model.save_model(tmp_path / "ckpt")

    again = DiffusionTransformer.from_pretrained(load_dir=str(tmp_path / "ckpt"),
                                                 map_location="cpu").eval()
    assert again.config.text_encoder_trainable_blocks == 2
    inputs, (ids, attn) = _inputs(), _caption()
    with torch.no_grad():
        assert torch.equal(model(**inputs, caption_ids=ids, caption_attn=attn),
                           again(**inputs, caption_ids=ids, caption_attn=attn))
