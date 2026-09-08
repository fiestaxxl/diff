"""The caption-to-length head: shape, masking, sampling, and the smoothed loss."""
from __future__ import annotations

import pytest
import torch

from dimol.models.length_head import LengthHead, length_loss

TEXT_DIM, TEXT_LEN, CANVAS = 32, 9, 64


def _inputs(batch: int = 4, real: int = 5):
    text = torch.randn(batch, TEXT_LEN, TEXT_DIM)
    mask = torch.zeros(batch, TEXT_LEN, dtype=torch.bool)
    mask[:, :real] = True
    return text, mask


def test_it_predicts_a_distribution_over_the_canvas():
    head = LengthHead(TEXT_DIM, canvas=CANVAS)
    logits = head(*_inputs())
    assert logits.shape == (4, CANVAS + 1)
    assert torch.isfinite(logits).all()


def test_padded_caption_tokens_are_ignored():
    head = LengthHead(TEXT_DIM, canvas=CANVAS).eval()
    text, mask = _inputs(real=5)
    other = text.clone()
    other[:, 5:] = torch.randn_like(other[:, 5:])
    with torch.no_grad():
        assert torch.allclose(head(text, mask), head(other, mask), atol=1e-5)


def test_the_mode_is_returned_at_temperature_zero():
    head = LengthHead(TEXT_DIM, canvas=CANVAS).eval()
    text, mask = _inputs()
    with torch.no_grad():
        assert torch.equal(head.predict(text, mask), head(text, mask).argmax(-1))


def test_sampling_varies_and_the_mode_does_not():
    head = LengthHead(TEXT_DIM, canvas=CANVAS).eval()
    text, mask = _inputs(batch=64)
    torch.manual_seed(0)
    a = head.predict(text, mask, temperature=1.0)
    b = head.predict(text, mask, temperature=1.0)
    assert not torch.equal(a, b)
    assert torch.equal(head.predict(text, mask), head.predict(text, mask))


def test_predictions_stay_on_the_canvas():
    head = LengthHead(TEXT_DIM, canvas=CANVAS).eval()
    text, mask = _inputs(batch=32)
    for temperature in (0.0, 1.0, 2.0):
        pred = head.predict(text, mask, temperature=temperature)
        assert int(pred.min()) >= 0 and int(pred.max()) <= CANVAS


def test_the_loss_gives_neighbours_partial_credit():
    """Being one token out must cost less than being thirty out."""
    logits = torch.zeros(1, CANVAS + 1)
    target = torch.tensor([20])
    near = torch.zeros(1, CANVAS + 1)
    near[0, 21] = 10.0
    far = torch.zeros(1, CANVAS + 1)
    far[0, 50] = 10.0
    assert length_loss(near, target) < length_loss(far, target)


def test_the_loss_is_lowest_when_it_is_right():
    target = torch.tensor([20])
    right = torch.zeros(1, CANVAS + 1)
    right[0, 20] = 10.0
    near = torch.zeros(1, CANVAS + 1)
    near[0, 21] = 10.0
    assert length_loss(right, target) < length_loss(near, target)


def test_it_trains_on_a_signal_it_can_learn():
    """A length written into the caption states must be learnable."""
    torch.manual_seed(0)
    head = LengthHead(TEXT_DIM, canvas=CANVAS)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    targets = torch.randint(5, 40, (128,))
    text = torch.randn(128, TEXT_LEN, TEXT_DIM) * 0.1
    text[:, 0, 0] = targets.float()  # the answer, in one coordinate
    mask = torch.ones(128, TEXT_LEN, dtype=torch.bool)

    first = length_loss(head(text, mask), targets).item()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = length_loss(head(text, mask), targets)
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.7


def test_it_survives_a_save_and_load(tmp_path):
    head = LengthHead(TEXT_DIM, canvas=CANVAS).eval()
    path = tmp_path / "head.pt"
    head.save(path)
    again = LengthHead.load(path)
    text, mask = _inputs()
    with torch.no_grad():
        assert torch.allclose(head(text, mask), again(text, mask), atol=1e-6)
    assert again.canvas == CANVAS
