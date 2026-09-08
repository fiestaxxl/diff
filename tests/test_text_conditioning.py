"""Text conditioning: it starts inert, it reads the caption, and it respects the mask."""
from __future__ import annotations

import pytest
import torch

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.models.layers import MultiheadCrossAttention

SEQ, VOCAB, TEXT_DIM, TEXT_LEN = 12, 64, 32, 7


def _model(text_dim: int = TEXT_DIM) -> DiffusionTransformer:
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4, text_dim=text_dim,
        self_conditioning=True, length_conditioning=True))


def _inputs(batch: int = 3):
    return dict(
        input_embeddings=torch.randn(batch, SEQ, 8),
        time=torch.full((batch, 1), 0.5),
        length=torch.full((batch,), 6, dtype=torch.long),
    )


def _text(batch: int = 3, real: int = 4):
    text = torch.randn(batch, TEXT_LEN, TEXT_DIM)
    mask = torch.zeros(batch, TEXT_LEN, dtype=torch.bool)
    mask[:, :real] = True
    return text, mask


def test_a_model_without_text_dim_has_no_cross_attention():
    plain = _model(text_dim=0)
    assert all(b.cross_attention is None for b in plain.text_transformer_blocks)
    assert not any(isinstance(m, MultiheadCrossAttention) for m in plain.modules())


def test_the_text_path_starts_completely_inert():
    """Grafting these layers onto a pretrained checkpoint must change nothing."""
    model = _model()
    inputs = _inputs()
    text, mask = _text()
    without = model(**inputs)
    with_caption = model(**inputs, text=text, text_mask=mask)
    assert torch.allclose(without, with_caption, atol=1e-6)


def test_once_trained_the_caption_changes_the_prediction():
    model = _model()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MultiheadCrossAttention):
                module.out_layer.weight.normal_(std=0.2)
    inputs = _inputs()
    a, mask = _text()
    b = torch.randn_like(a)
    assert not torch.allclose(model(**inputs, text=a, text_mask=mask),
                              model(**inputs, text=b, text_mask=mask), atol=1e-4)


def test_masked_caption_tokens_are_ignored():
    model = _model()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MultiheadCrossAttention):
                module.out_layer.weight.normal_(std=0.2)
    inputs = _inputs()
    text, mask = _text(real=4)
    other = text.clone()
    other[:, 4:] = torch.randn_like(other[:, 4:])  # change only the padded tail
    assert torch.allclose(model(**inputs, text=text, text_mask=mask),
                          model(**inputs, text=other, text_mask=mask), atol=1e-5)


def test_a_caption_without_a_text_path_is_refused():
    model = _model(text_dim=0)
    text, mask = _text()
    with pytest.raises(ValueError, match="text_dim"):
        model(**_inputs(), text=text, text_mask=mask)


def test_dropping_the_caption_is_the_unconditional_model():
    """Classifier-free guidance needs the same weights to run with no caption at all."""
    model = _model()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MultiheadCrossAttention):
                module.out_layer.weight.normal_(std=0.2)
    inputs = _inputs()
    text, mask = _text()
    empty = torch.zeros_like(mask)
    assert torch.allclose(model(**inputs), model(**inputs, text=text, text_mask=empty),
                          atol=1e-5)


def test_the_text_path_trains():
    model = _model()
    inputs = _inputs()
    text, mask = _text()
    model(**inputs, text=text, text_mask=mask).square().mean().backward()
    grads = [m.out_layer.weight.grad for m in model.modules()
             if isinstance(m, MultiheadCrossAttention)]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_the_config_round_trips_the_text_width(tmp_path):
    model = _model()
    model.save_model(tmp_path / "ckpt")
    again = DiffusionTransformer.from_pretrained(load_dir=str(tmp_path / "ckpt"),
                                                 map_location="cpu")
    assert again.config.text_dim == TEXT_DIM
