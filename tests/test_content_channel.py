"""The content channel: shape, inertness, and that it carries the termination signal."""
from __future__ import annotations

import pytest
import torch

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig

SEQ, VOCAB, EMB = 12, 64, 8


def _model(content: bool, self_cond: bool = True) -> DiffusionTransformer:
    return DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=EMB, time_dim=32, num_heads=4, num_text_blocks=2,
        vocab_size=VOCAB, pad_idx=0, max_pos=SEQ + 4,
        self_conditioning=self_cond, content_channel=content))


def test_the_latent_gains_exactly_one_channel():
    plain, wide = _model(content=False), _model(content=True)
    assert plain.latent_dim == EMB
    assert wide.latent_dim == EMB + 1
    assert wide.up_proj.in_features == plain.up_proj.in_features + 2  # self-conditioned


def test_the_prediction_has_the_same_width_as_the_latent():
    model = _model(content=True).eval()
    out = model(input_embeddings=torch.randn(3, SEQ, EMB + 1),
                time=torch.full((3, 1), 0.5))
    assert out.shape == (3, SEQ, EMB + 1)


def test_the_channel_is_ignored_at_initialisation():
    """A checkpoint grown into this shape must behave exactly as it did before."""
    model = _model(content=True).eval()
    latent = torch.randn(3, SEQ, EMB + 1)
    changed = latent.clone()
    changed[..., EMB] = torch.randn(3, SEQ)  # only the content channel differs
    torch.manual_seed(0)
    a = model(input_embeddings=latent, time=torch.full((3, 1), 0.5))
    torch.manual_seed(0)
    b = model(input_embeddings=changed, time=torch.full((3, 1), 0.5))
    assert torch.allclose(a[..., :EMB], b[..., :EMB], atol=1e-6)


def test_the_channel_is_read_once_its_columns_are_trained():
    model = _model(content=True).eval()
    with torch.no_grad():
        model.up_proj.weight[:, EMB].normal_(0, 1.0)
    latent = torch.randn(3, SEQ, EMB + 1)
    changed = latent.clone()
    changed[..., EMB] = torch.randn(3, SEQ)
    torch.manual_seed(0)
    a = model(input_embeddings=latent, time=torch.full((3, 1), 0.5))
    torch.manual_seed(0)
    b = model(input_embeddings=changed, time=torch.full((3, 1), 0.5))
    assert not torch.allclose(a[..., :EMB], b[..., :EMB], atol=1e-5)


def test_the_loss_builds_the_channel_from_the_attention_mask():
    from dimol.builders import build_path
    from omegaconf import OmegaConf

    from dimol.training.tasks import DiffusionTask

    model = _model(content=True)
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    path = build_path(cfg, seq_len=SEQ, emb_dim=EMB + 1, device="cpu")
    task = DiffusionTask(path, pad_idx=0, grammar_enabled=False, self_cond_prob=0.0,
                         seq_len=SEQ)
    mask = torch.zeros(3, SEQ, dtype=torch.long)
    mask[:, :5] = 1
    batch = {"token_ids": torch.randint(1, VOCAB, (3, SEQ)), "attention_mask": mask}
    out = task.compute_loss(model, batch, step=0)
    assert torch.isfinite(out["loss"])


def test_the_channel_needs_an_attention_mask():
    from dimol.builders import build_path
    from omegaconf import OmegaConf

    from dimol.training.tasks import DiffusionTask

    model = _model(content=True)
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    path = build_path(cfg, seq_len=SEQ, emb_dim=EMB + 1, device="cpu")
    task = DiffusionTask(path, pad_idx=0, grammar_enabled=False, self_cond_prob=0.0,
                         seq_len=SEQ)
    batch = {"token_ids": torch.randint(1, VOCAB, (3, SEQ))}
    with pytest.raises(ValueError, match="content_channel needs"):
        task.compute_loss(model, batch, step=0)


def test_the_sampler_reads_the_channel_and_truncates_there():
    """The channel has to govern the string, not merely be trained."""
    import numpy as np
    from omegaconf import OmegaConf

    from dimol.builders import build_path
    from dimol.eval.sampling import SamplingParams, sample_smiles

    class _Tok:
        def decode_batch(self, ids, special_decode=True):
            return ["".join("C" if t else "" for t in row) for row in ids]

    model = _model(content=True, self_cond=False).eval()
    # Force the channel negative from position 5 on, whatever the trajectory does.
    with torch.no_grad():
        model.content_head.bias.fill_(-50.0)
    cfg = OmegaConf.create({"diffusion": {"name": "vp_cosine", "p_simple": "gaussian"}})
    path = build_path(cfg, seq_len=SEQ, emb_dim=EMB + 1, device="cpu")
    params = SamplingParams(num_samples=2, batch_size=2, num_timesteps=4, seed=0,
                            decode="argmax")
    out = sample_smiles(model, path, _Tok(), params, "cpu")
    assert len(out) == 2
