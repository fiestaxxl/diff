"""Exporting caption states: the output must be a drop-in corpus, not a near-miss.

The whole point of exporting states rather than running a moved encoder inside the
evaluation is that everything downstream — eval_text, score_with_ar, select_candidates,
oracle_ceiling, the analysis scripts — keeps reading ``cfg.variables.data_dir`` and needs
no change. That only holds if the exported directory is byte-compatible with the prepared
one: same dtypes, same row alignment, same filenames, with the molecule arrays carried
over rather than regenerated. These tests pin that by constructing the real dataset class
on the output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("transformers")

from export_caption_states import export_states  # noqa: E402

from dimol.data.datasets import PairedSmilesTextDataset  # noqa: E402

ROWS, TEXT_LEN, HIDDEN, CANVAS, VOCAB = 4, 8, 16, 6, 64


def tiny_encoder(path: Path):
    """A two-layer random BERT: the export's shapes and dtypes, without 400 MB of SciBERT."""
    from transformers import BertConfig, BertModel

    model = BertModel(BertConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=2 * HIDDEN,
        max_position_embeddings=2 * TEXT_LEN), add_pooling_layer=False)
    model.save_pretrained(path)
    return model


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "tokenized"
    corpus.mkdir()
    rng = np.random.default_rng(0)
    ids = rng.integers(1, VOCAB, size=(ROWS, TEXT_LEN)).astype(np.uint16)
    attn = np.zeros((ROWS, TEXT_LEN), dtype=np.uint8)
    for row in range(ROWS):
        attn[row, : TEXT_LEN - row] = 1
        ids[row, TEXT_LEN - row :] = 0
    np.save(corpus / "train_caption_ids.npy", ids)
    np.save(corpus / "train_caption_attn.npy", attn)

    tokens = np.zeros((ROWS, CANVAS), dtype=np.uint16)
    masks = np.zeros((ROWS, CANVAS), dtype=np.uint8)
    tokens[:, :3] = np.arange(1, 4)
    masks[:, :3] = 1
    np.save(corpus / "train_tokens_00000.npy", tokens)
    np.save(corpus / "train_attn_mask_00000.npy", masks)
    (corpus / "train_captions.txt").write_text("\n".join(f"caption {i}" for i in range(ROWS)) + "\n")
    # a stale states file, to prove the export replaces it instead of copying it over
    np.save(corpus / "train_text.npy", np.ones((ROWS, TEXT_LEN, HIDDEN), dtype=np.float16))
    np.save(corpus / "train_text_mask.npy", np.ones((ROWS, TEXT_LEN), dtype=np.uint8))
    (corpus / "meta.json").write_text(json.dumps({"splits": {"train": {"kept": ROWS}}}))
    return corpus


def test_the_exported_directory_loads_as_a_paired_corpus(tmp_path):
    """The acceptance test for 'drop-in': the shipped dataset class must construct on it."""
    corpus = _corpus(tmp_path)
    encoder = tiny_encoder(tmp_path / "enc")
    export_states(encoder, corpus, tmp_path / "out", ["train"])

    dataset = PairedSmilesTextDataset(tmp_path / "out", "train")
    assert len(dataset) == ROWS
    item = dataset[1]
    assert item["text"].shape == (TEXT_LEN, HIDDEN)
    assert item["token_ids"].shape == (CANVAS,)
    assert item["text_mask"].dtype == torch.bool


def test_the_states_keep_the_corpus_dtype(tmp_path):
    """float16, exactly as prepare_paired writes: no new quantisation in the comparison."""
    corpus = _corpus(tmp_path)
    export_states(tiny_encoder(tmp_path / "enc"), corpus, tmp_path / "out", ["train"])
    states = np.load(tmp_path / "out" / "train_text.npy")
    assert states.dtype == np.float16
    assert states.shape == (ROWS, TEXT_LEN, HIDDEN)
    assert not np.array_equal(states, np.load(corpus / "train_text.npy"))


def test_the_mask_comes_from_the_stored_caption_attention(tmp_path):
    """Not from a re-tokenization: the ids on disk are the only source of truth here."""
    corpus = _corpus(tmp_path)
    export_states(tiny_encoder(tmp_path / "enc"), corpus, tmp_path / "out", ["train"])
    assert np.array_equal(np.load(tmp_path / "out" / "train_text_mask.npy"),
                          np.load(corpus / "train_caption_attn.npy"))


def test_the_molecules_and_the_caption_ids_are_carried_over(tmp_path):
    """Carried over, so the only thing that differs from the baseline is the states."""
    corpus = _corpus(tmp_path)
    export_states(tiny_encoder(tmp_path / "enc"), corpus, tmp_path / "out", ["train"])
    out = tmp_path / "out"
    for name in ("train_tokens_00000.npy", "train_attn_mask_00000.npy",
                 "train_caption_ids.npy", "train_caption_attn.npy"):
        assert (out / name).exists() and not (out / name).is_symlink()
        assert np.array_equal(np.load(out / name), np.load(corpus / name))
    assert (out / "train_captions.txt").read_text() == (corpus / "train_captions.txt").read_text()


def test_the_provenance_of_the_states_is_recorded(tmp_path):
    """A corpus that cannot say which encoder wrote it is a corpus nobody can trust."""
    corpus = _corpus(tmp_path)
    export_states(tiny_encoder(tmp_path / "enc"), corpus, tmp_path / "out", ["train"],
                  provenance={"text_encoder": "an encoder", "text_encoder_step": 6000})
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["text_encoder"] == "an encoder" and meta["text_encoder_step"] == 6000
    assert meta["text_hidden"] == HIDDEN
    assert meta["splits"]["train"]["kept"] == ROWS  # the prepared corpus's own report


def test_the_checkpoint_branch_exports_the_encoder_the_model_carries(tmp_path):
    """Scoring an arm against the states its own encoder moved away from is the failure
    this branch exists to prevent, and it leaves no trace in any metric."""
    from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
    from export_caption_states import load_encoder_from_checkpoint

    corpus = _corpus(tmp_path)
    encoder_dir = tmp_path / "enc"
    tiny_encoder(encoder_dir)
    model = DiffusionTransformer(TransformerConfig(
        model_dim=16, emb_dim=4, time_dim=16, num_heads=2, num_text_blocks=1,
        vocab_size=VOCAB, pad_idx=0, max_pos=CANVAS + 4, text_dim=HIDDEN,
        text_encoder=str(encoder_dir), text_encoder_trainable_blocks=1))
    with torch.no_grad():  # move the encoder, so the export has something to carry
        for param in model.text_encoder.encoder.layer[-1].parameters():
            param.add_(torch.randn_like(param) * 0.05)
    model.save_model(tmp_path / "ckpt")

    carried = load_encoder_from_checkpoint(tmp_path / "ckpt", "cpu")
    export_states(carried, corpus, tmp_path / "out", ["train"])

    ids = np.load(corpus / "train_caption_ids.npy").astype(np.int64)
    attn = np.load(corpus / "train_caption_attn.npy").astype(np.int64)
    with torch.no_grad():
        expected = model.text_encoder(input_ids=torch.from_numpy(ids),
                                      attention_mask=torch.from_numpy(attn)).last_hidden_state
    assert np.array_equal(np.load(tmp_path / "out" / "train_text.npy"),
                          expected.numpy().astype(np.float16))


def test_a_checkpoint_with_no_encoder_says_so(tmp_path):
    from dimol.models.diffusion_transformer import DiffusionTransformer, TransformerConfig
    from export_caption_states import load_encoder_from_checkpoint

    DiffusionTransformer(TransformerConfig(
        model_dim=16, emb_dim=4, time_dim=16, num_heads=2, num_text_blocks=1,
        vocab_size=VOCAB, pad_idx=0, max_pos=CANVAS + 4, text_dim=HIDDEN,
    )).save_model(tmp_path / "ckpt")
    with pytest.raises(ValueError, match="holds no caption encoder"):
        load_encoder_from_checkpoint(tmp_path / "ckpt", "cpu")
