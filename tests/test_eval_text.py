"""The rdkit-free halves of the text evaluation: grammar and token overlap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_text  # noqa: E402
from eval_text import grammatical, token_f1  # noqa: E402

pytest.importorskip("tokenizers")
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402

SAMPLE = ["CCO", "c1ccccc1", "CC(=O)O", "C[C@H](N)C(=O)O", "CC(C)Cc1ccccc1",
          "C1CCCCC1", "CCN(CC)CC", "c1ccc2ccccc2c1", "CC(=O)Nc1ccccc1", "CCOC(C)=O"]


@pytest.fixture(scope="module")
def tokenizer():
    return SmilesTokenizer.train(SAMPLE, vocab_size=200, min_frequency=1,
                                 show_progress=False)


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "C1CC1",
                                    "C[C@H](N)C(=O)O", "c1ccc2ccccc2c1"])
def test_real_molecules_are_grammatical(smiles):
    assert grammatical(smiles)


@pytest.mark.parametrize("smiles", ["CC22", "CC(", "CC)", "C1CC", "CC()C", "CC==C",
                                    "[C@H", ""])
def test_broken_strings_are_not(smiles):
    assert not grammatical(smiles)


def test_a_two_digit_ring_closure_is_read_whole():
    assert grammatical("C%10CCCCC%10")
    assert not grammatical("C%10CCCCC")


def test_token_f1_is_one_for_the_same_molecule(tokenizer):
    assert token_f1("CC(=O)O", "CC(=O)O", tokenizer) == pytest.approx(1.0)


def test_token_f1_is_zero_when_nothing_overlaps(tokenizer):
    assert token_f1("", "CC(=O)O", tokenizer) == 0.0


def test_token_f1_gives_partial_credit(tokenizer):
    """A near miss must score between a hit and a miss, so progress is visible early."""
    exact = token_f1("CC(=O)Nc1ccccc1", "CC(=O)Nc1ccccc1", tokenizer)
    near = token_f1("CC(=O)Nc1ccccc1C", "CC(=O)Nc1ccccc1", tokenizer)
    far = token_f1("CCO", "CC(=O)Nc1ccccc1", tokenizer)
    assert exact == pytest.approx(1.0)
    assert far < near < exact


def test_token_f1_is_segmentation_dependent(tokenizer):
    """Worth pinning because it is a limitation, not a feature.

    The score is a multiset overlap over the tokenizer's own segmentation, and a
    byte-pair vocabulary segments "CCO" and "OCC" into different merges. So the same
    atoms in a different order can score zero. That makes token F1 a cheap progress
    signal inside the job, where rdkit is unavailable, and never a chemical measure:
    the fingerprint similarities in the scoring stage are the ones that mean something.
    """
    assert token_f1("CCO", "OCC", tokenizer) < 1.0


# ------------------------- which caption states a checkpoint is actually scored against

CANVAS, TEXT_LEN, HIDDEN, CAPTION_VOCAB, ROWS = 8, 6, 16, 32, 4

# Whatever a randomly initialised encoder produces, it is not a constant, so a generation
# run scored against this array is visibly reading the file rather than the encoder.
STORED = 0.25


def _tiny_encoder(path: Path) -> Path:
    from transformers import BertConfig, BertModel

    BertModel(BertConfig(
        vocab_size=CAPTION_VOCAB, hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=2 * HIDDEN,
        max_position_embeddings=4 * TEXT_LEN), add_pooling_layer=False).save_pretrained(path)
    return path


def _eval_fixture(tmp_path, tokenizer, encoder_dir=None):
    """A corpus, a checkpoint and the config stage_generate reads, and nothing else."""
    from omegaconf import OmegaConf

    from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig

    data_dir = tmp_path / "tokenized"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    ids = rng.integers(1, CAPTION_VOCAB, size=(ROWS, TEXT_LEN)).astype(np.uint16)
    attn = np.ones((ROWS, TEXT_LEN), dtype=np.uint8)
    attn[:, -1] = 0
    ids[:, -1] = 0
    np.save(data_dir / "val_caption_ids.npy", ids)
    np.save(data_dir / "val_caption_attn.npy", attn)
    np.save(data_dir / "val_text.npy",
            np.full((ROWS, TEXT_LEN, HIDDEN), STORED, dtype=np.float16))
    np.save(data_dir / "val_text_mask.npy", attn)

    tokens = np.zeros((ROWS, CANVAS), dtype=np.uint16)
    masks = np.zeros((ROWS, CANVAS), dtype=np.uint8)
    tokens[:, :3] = np.arange(5, 8)
    masks[:, :3] = 1
    np.save(data_dir / "val_tokens_00000.npy", tokens)
    np.save(data_dir / "val_attn_mask_00000.npy", masks)

    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(tok_path)

    model = DiffusionTransformer(TransformerConfig(
        model_dim=16, emb_dim=4, time_dim=16, num_heads=2, num_text_blocks=1,
        vocab_size=200, pad_idx=0, max_pos=CANVAS + 4, text_dim=HIDDEN,
        text_encoder="" if encoder_dir is None else str(encoder_dir)))
    model.save_model(tmp_path / "ckpt")

    config = tmp_path / "cfg.yaml"
    OmegaConf.save(OmegaConf.create({
        "variables": {"data_dir": str(data_dir), "seq_len": CANVAS},
        "tokenizer": {"path": str(tok_path)},
        "diffusion": {"alpha": {"name": "cosine"}, "beta": {"name": "cosine"}},
    }), config)
    return config, data_dir, tmp_path / "ckpt", model


def _generate(monkeypatch, config, checkpoint, out, *extra):
    """Run the real CLI with the sampler replaced, and report what reached it.

    Stubbing the sampler rather than sampling: the question here is only which states the
    conditioning is built from, and three hundred solver steps would answer it no better.
    """
    import dimol.eval.sampling as sampling

    seen = {}

    def _capture(model, path, tokenizer, params, device, **kwargs):
        seen["params"] = params
        return ["CCO"] * params.num_samples

    monkeypatch.setattr(sampling, "sample_smiles", _capture)
    monkeypatch.setattr(sys, "argv", [
        "eval_text.py", "--stage", "generate", "--config", str(config),
        "--checkpoint", str(checkpoint), "--split", "val", "--out", str(out), *extra])
    eval_text.main()
    return seen["params"]


def test_a_checkpoint_without_an_encoder_is_still_scored_on_the_stored_states(
        tmp_path, tokenizer, monkeypatch):
    """Every recorded number in this project came through this branch, so the detection
    must not be able to divert a checkpoint that carries no encoder — including one whose
    corpus happens to have caption ids sitting beside the states."""
    config, _, checkpoint, _ = _eval_fixture(tmp_path, tokenizer)
    params = _generate(monkeypatch, config, checkpoint, tmp_path / "out")
    assert params.text.shape == (ROWS, TEXT_LEN, HIDDEN)
    assert np.array_equal(params.text, np.full(params.text.shape, STORED, np.float32))


def test_a_checkpoint_with_an_encoder_is_scored_through_that_encoder(
        tmp_path, tokenizer, monkeypatch):
    """The silent failure this branch exists to prevent: a model whose encoder moved,
    scored against the states it moved away from. The run completes, the metrics look
    ordinary, and they belong to a model that does not exist."""
    pytest.importorskip("transformers")
    encoder_dir = _tiny_encoder(tmp_path / "enc")
    config, data_dir, checkpoint, model = _eval_fixture(tmp_path, tokenizer, encoder_dir)
    params = _generate(monkeypatch, config, checkpoint, tmp_path / "out")

    ids = np.load(data_dir / "val_caption_ids.npy").astype(np.int64)
    attn = np.load(data_dir / "val_caption_attn.npy").astype(np.int64)
    with torch.no_grad():
        expected = model.text_encoder(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(attn)).last_hidden_state
    # float16 on the way through, because that is the dtype the stored states have: the
    # two arms then differ by their encoder and not by their precision
    assert np.array_equal(params.text,
                          expected.numpy().astype(np.float16).astype(np.float32))
    assert not np.array_equal(params.text, np.full(params.text.shape, STORED, np.float32))


def test_the_caption_mask_comes_from_the_stored_caption_attention(
        tmp_path, tokenizer, monkeypatch):
    pytest.importorskip("transformers")
    config, data_dir, checkpoint, _ = _eval_fixture(
        tmp_path, tokenizer, _tiny_encoder(tmp_path / "enc"))
    params = _generate(monkeypatch, config, checkpoint, tmp_path / "out")
    assert np.array_equal(params.text_mask,
                          np.load(data_dir / "val_caption_attn.npy").astype(bool))


def test_only_the_captions_that_will_be_sampled_are_encoded(
        tmp_path, tokenizer, monkeypatch):
    """--limit is the whole point of a quick val read; encoding the rest is wasted GPU."""
    pytest.importorskip("transformers")
    config, _, checkpoint, _ = _eval_fixture(
        tmp_path, tokenizer, _tiny_encoder(tmp_path / "enc"))
    params = _generate(monkeypatch, config, checkpoint, tmp_path / "out", "--limit", "2")
    assert params.num_samples == 2 and params.text.shape[0] == 2


def test_a_length_head_against_a_moved_encoder_is_refused(
        tmp_path, tokenizer, monkeypatch):
    """The head was fitted on the frozen states and its inputs have moved underneath it.
    Nothing in a predicted length looks wrong, so this cannot be a warning."""
    pytest.importorskip("transformers")
    config, _, checkpoint, _ = _eval_fixture(
        tmp_path, tokenizer, _tiny_encoder(tmp_path / "enc"))
    with pytest.raises(ValueError, match="moved away"):
        _generate(monkeypatch, config, checkpoint, tmp_path / "out",
                  "--length-head", str(tmp_path / "head.pt"))


def test_an_encoder_checkpoint_without_caption_ids_names_the_script_that_writes_them(
        tmp_path, tokenizer, monkeypatch):
    pytest.importorskip("transformers")
    config, data_dir, checkpoint, _ = _eval_fixture(
        tmp_path, tokenizer, _tiny_encoder(tmp_path / "enc"))
    (data_dir / "val_caption_ids.npy").unlink()
    with pytest.raises(FileNotFoundError, match="tokenize_captions"):
        _generate(monkeypatch, config, checkpoint, tmp_path / "out")
