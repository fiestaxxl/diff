"""Registry: components are built by name from config."""
from __future__ import annotations

import pytest
import torch

import dimol.builders  # noqa: F401  (registers models/tasks/schedules)
import dimol.data.datasets  # noqa: F401  (registers datasets)
from dimol import registry


def test_expected_names_registered() -> None:
    assert "diffusion_transformer" in registry.models
    assert "smiles_ar" in registry.models
    assert {"diffusion", "ar"} <= set(registry.tasks.keys())
    assert {"cosine", "linear"} <= set(registry.alphas.keys())
    assert {"cosine", "sqrt"} <= set(registry.betas.keys())
    assert "adamw" in registry.optimizers
    assert "warmup_cosine" in registry.schedulers
    assert {"console", "jsonl"} <= set(registry.loggers.keys())
    assert {"smiles_npy", "random"} <= set(registry.datasets.keys())


def test_build_from_dict() -> None:
    model = registry.models.build(
        {
            "name": "diffusion_transformer",
            "model_dim": 16,
            "emb_dim": 4,
            "time_dim": 16,
            "num_heads": 2,
            "num_text_blocks": 1,
            "vocab_size": 16,
            "pad_idx": 0,
            "max_pos": 8,
        }
    )
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("name", ["cosine", "linear"])
def test_alpha_builds_with_device(name: str) -> None:
    """build_path always passes device, so every schedule must accept it."""
    obj = registry.alphas.build({"name": name}, device="cpu")
    t = torch.zeros(2, 1, 1)
    assert torch.allclose(obj(t), torch.zeros(2, 1, 1), atol=1e-6)


@pytest.mark.parametrize("name", ["cosine", "sqrt"])
def test_beta_builds_with_device(name: str) -> None:
    obj = registry.betas.build({"name": name}, device="cpu")
    t = torch.zeros(2, 1, 1)
    assert torch.allclose(obj(t), torch.ones(2, 1, 1), atol=1e-6)


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        registry.models.build({"name": "no_such_model"})


def test_missing_name_raises() -> None:
    with pytest.raises(KeyError):
        registry.models.build({"model_dim": 16})
