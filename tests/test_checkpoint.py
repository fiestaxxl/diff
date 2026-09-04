"""A checkpoint restores both the weights and the optimizer/step state."""
from __future__ import annotations

from pathlib import Path

import torch

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.checkpoint import (
    TrainerState,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)


def _model() -> DiffusionTransformer:
    return DiffusionTransformer(
        TransformerConfig(
            model_dim=16, emb_dim=4, time_dim=16, num_heads=2, num_text_blocks=1,
            vocab_size=16, pad_idx=0, max_pos=8,
        )
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 4, 4)
    t = torch.rand(2, 1)
    model(input_embeddings=x, time=t).sum().backward()
    opt.step()

    state = TrainerState(step=7, epoch=2)
    path = save_checkpoint(tmp_path, model, state, optimizer=opt, keep_last=3)
    assert (path / "config.json").exists()

    model2 = _model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    restored = load_checkpoint(path, model2, optimizer=opt2)

    assert restored.step == 7 and restored.epoch == 2
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)
    assert opt2.state_dict()["state"], "optimizer state was not restored"


def test_keep_last_prunes_old(tmp_path: Path) -> None:
    model = _model()
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, model, TrainerState(step=step, epoch=0), keep_last=2)
    kept = [p.name for p in list_checkpoints(tmp_path)]
    assert kept == ["ep0-ba3", "ep0-ba4"]
    assert latest_checkpoint(tmp_path).name == "ep0-ba4"
