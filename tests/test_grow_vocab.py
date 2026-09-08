"""Growing a checkpoint: trained rows survive, new rows are usable, config follows."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig


def _checkpoint(tmp_path: Path, vocab: int = 64, max_pos: int = 16) -> Path:
    model = DiffusionTransformer(TransformerConfig(
        model_dim=32, emb_dim=8, time_dim=32, num_heads=4, num_text_blocks=1,
        vocab_size=vocab, pad_idx=0, max_pos=max_pos,
        self_conditioning=True, length_conditioning=True))
    with torch.no_grad():  # pretend it trained: the tables grow a long way from init
        model.token_embedding.weight.mul_(50.0)
        model.out_proj.weight.mul_(30.0)
        model.out_proj.bias.add_(torch.linspace(-1, 1, vocab))
    out = tmp_path / "ckpt"
    model.save_model(out)
    return out


def _grow(source: Path, out: Path, vocab: int, max_pos: int | None = None):
    args = [sys.executable, str(ROOT / "scripts" / "grow_vocab.py"),
            "--checkpoint", str(source), "--vocab-size", str(vocab), "--out", str(out)]
    if max_pos is not None:
        args += ["--max-pos", str(max_pos)]
    return subprocess.run(args, capture_output=True, text=True, check=True)


def test_trained_rows_come_through_untouched(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "grown"
    proc = _grow(source, out, 96, 32)
    assert "every trained row is unchanged" in proc.stdout

    from safetensors.torch import load_file

    before = load_file(str(source / "model.safetensors"))
    after = load_file(str(out / "model.safetensors"))
    for key in ("token_embedding.weight", "out_proj.weight", "out_proj.bias"):
        assert after[key].shape[0] == 96
        assert torch.equal(after[key][:64], before[key])


def test_new_rows_are_at_the_scale_of_the_trained_ones(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "grown"
    _grow(source, out, 96)

    from safetensors.torch import load_file

    after = load_file(str(out / "model.safetensors"))["token_embedding.weight"]
    old_std = after[:64].float().std().item()
    new_std = after[64:].float().std().item()
    assert 0.3 < new_std / old_std < 3.0, (old_std, new_std)


def test_new_biases_start_unlikely(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "grown"
    _grow(source, out, 96)

    from safetensors.torch import load_file

    bias = load_file(str(out / "model.safetensors"))["out_proj.bias"]
    assert torch.allclose(bias[64:], bias[:64].min().expand(32))


def test_the_grown_checkpoint_loads_and_runs(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "grown"
    _grow(source, out, 96, 32)

    model = DiffusionTransformer.from_pretrained(load_dir=str(out), map_location="cpu")
    assert model.config.vocab_size == 96
    assert model.config.max_pos == 32
    x = torch.randn(2, 24, 8)  # a canvas longer than the original 16 positions
    out_tensor = model(input_embeddings=x, time=torch.full((2, 1), 0.5),
                       length=torch.tensor([10, 20]))
    assert out_tensor.shape == x.shape
    assert torch.isfinite(out_tensor).all()


def test_the_config_records_both_changes(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "grown"
    _grow(source, out, 128, 64)
    config = json.loads((out / "config.json").read_text())
    assert config["vocab_size"] == 128 and config["max_pos"] == 64


def test_growing_to_the_same_size_changes_nothing(tmp_path):
    source = _checkpoint(tmp_path)
    out = tmp_path / "same"
    proc = _grow(source, out, 64)
    assert "nothing to grow" in proc.stdout


def test_the_length_table_grows_with_the_canvas(tmp_path):
    source = _checkpoint(tmp_path, max_pos=16)
    out = tmp_path / "grown"
    _grow(source, out, 96, 40)

    from safetensors.torch import load_file

    before = load_file(str(source / "model.safetensors"))["length_embedding.weight"]
    after = load_file(str(out / "model.safetensors"))["length_embedding.weight"]
    assert before.shape[0] == 17 and after.shape[0] == 41
    assert torch.equal(after[:17], before)
    # the new lengths start as copies of the longest one the model saw
    assert torch.equal(after[17], before[-1])
