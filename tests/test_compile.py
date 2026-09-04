"""torch.compile must produce one graph and the same numbers as eager.

The denoiser used to carry @torch._dynamo.disable on apply_rotary, which broke the
graph into fragments and cost most of the fusion benefit. This test pins both the
single-graph property and numerical equality.
"""
from __future__ import annotations

import torch

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig


def _model_and_input():
    torch.manual_seed(0)
    cfg = TransformerConfig(
        model_dim=64, emb_dim=8, time_dim=64, num_heads=4, num_text_blocks=2,
        vocab_size=32, pad_idx=0, max_pos=24,
    )
    model = DiffusionTransformer(cfg).eval()
    return model, torch.randn(2, 12, cfg.emb_dim), torch.rand(2, 1)


def test_no_graph_breaks() -> None:
    model, x, t = _model_and_input()
    explanation = torch._dynamo.explain(model)(input_embeddings=x, time=t)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1


def test_compiled_matches_eager() -> None:
    model, x, t = _model_and_input()
    with torch.no_grad():
        eager = model(input_embeddings=x, time=t)
        compiled = torch.compile(model, backend="inductor")(input_embeddings=x, time=t)
    assert torch.allclose(compiled, eager, atol=1e-5)
