"""Diffusion-DPO: the loss compares a pair against a frozen reference."""
from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

from dimol.builders import build_path
from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
from dimol.training.tasks import DiffusionDPOTask

SEQ, VOCAB, EMB = 12, 64, 8


def _model() -> DiffusionTransformer:
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4,
        self_conditioning=False, length_conditioning=False))


def _path():
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    return build_path(cfg, seq_len=SEQ, emb_dim=EMB, device="cpu")


def _batch(same: bool = False):
    accepted = torch.randint(1, 40, (4, SEQ))
    rejected = accepted.clone() if same else torch.randint(1, 40, (4, SEQ))
    mask = torch.ones(4, SEQ, dtype=torch.long)
    return {"token_ids": accepted, "attention_mask": mask,
            "rejected_token_ids": rejected, "rejected_attention_mask": mask}


def test_the_loss_needs_a_reference():
    task = DiffusionDPOTask(_path(), seq_len=SEQ)
    with pytest.raises(ValueError, match="frozen reference"):
        task.compute_loss(_model(), _batch(), step=0)


def test_an_untrained_model_equal_to_its_reference_scores_log_two():
    """Before any update the model IS the reference, so both margins cancel."""
    torch.manual_seed(0)
    model = _model()
    task = DiffusionDPOTask(_path(), seq_len=SEQ)
    task.set_reference(copy.deepcopy(model))
    torch.manual_seed(1)
    out = task.compute_loss(model, _batch(), step=0)
    assert abs(float(out["loss"]) - torch.log(torch.tensor(2.0)).item()) < 1e-4
    assert abs(float(out["margin"])) < 1e-5


def test_the_reference_is_frozen():
    model = _model()
    task = DiffusionDPOTask(_path(), seq_len=SEQ)
    task.set_reference(copy.deepcopy(model))
    assert all(not p.requires_grad for p in task.reference_model.parameters())


def test_the_loss_has_a_gradient_and_the_reference_does_not():
    torch.manual_seed(0)
    model = _model()
    task = DiffusionDPOTask(_path(), seq_len=SEQ)
    task.set_reference(copy.deepcopy(model))
    out = task.compute_loss(model, _batch(), step=0)
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    assert all(p.grad is None for p in task.reference_model.parameters())


def test_the_pair_shares_one_timestep_and_one_state():
    """An identical pair must give exactly zero margin: any difference would mean the two
    members were evaluated at different diffusion states, which would confound the loss."""
    torch.manual_seed(0)
    model = _model()
    task = DiffusionDPOTask(_path(), seq_len=SEQ, x0_noise_std=0.0)
    task.set_reference(copy.deepcopy(model))
    torch.manual_seed(2)
    out = task.compute_loss(model, _batch(same=True), step=0)
    assert abs(float(out["margin"])) < 1e-5
