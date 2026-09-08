"""The paired dataset and the collate rule that keeps captions off the canvas axis."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dimol.data.collate import trim_collate
from dimol.data.datasets import PairedSmilesTextDataset

CANVAS, TEXT_LEN, TEXT_DIM = 16, 16, 4


def _write(tmp_path, split="train", n=6, real=5):
    tokens = np.zeros((n, CANVAS), dtype=np.uint16)
    masks = np.zeros((n, CANVAS), dtype=np.uint8)
    tokens[:, :real] = np.arange(1, real + 1)
    masks[:, :real] = 1
    np.save(tmp_path / f"{split}_tokens_00000.npy", tokens)
    np.save(tmp_path / f"{split}_attn_mask_00000.npy", masks)
    text = np.random.default_rng(0).normal(size=(n, TEXT_LEN, TEXT_DIM)).astype(np.float16)
    text_mask = np.zeros((n, TEXT_LEN), dtype=np.uint8)
    text_mask[:, : TEXT_LEN - 2] = 1  # captions are longer than the molecules
    np.save(tmp_path / f"{split}_text.npy", text)
    np.save(tmp_path / f"{split}_text_mask.npy", text_mask)
    return tokens, masks, text, text_mask


def test_a_row_carries_its_own_caption(tmp_path):
    tokens, masks, text, text_mask = _write(tmp_path)
    ds = PairedSmilesTextDataset(tmp_path, "train")
    assert len(ds) == len(tokens)
    item = ds[2]
    assert item["token_ids"].tolist() == tokens[2].tolist()
    assert item["text"].shape == (TEXT_LEN, TEXT_DIM)
    # compared through numpy: torch's float16 path consults cpuinfo, which the login
    # node's container cannot parse, and the op raises instead of returning
    assert np.allclose(item["text"].numpy(), text[2].astype(np.float32))
    assert item["text_mask"].tolist() == text_mask[2].astype(bool).tolist()


def test_a_negative_index_takes_the_right_caption(tmp_path):
    _write(tmp_path)
    ds = PairedSmilesTextDataset(tmp_path, "train")
    assert np.array_equal(ds[-1]["text"].numpy(), ds[len(ds) - 1]["text"].numpy())


def test_missing_captions_are_reported(tmp_path):
    _write(tmp_path)
    (tmp_path / "train_text.npy").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_paired"):
        PairedSmilesTextDataset(tmp_path, "train")


def test_misaligned_arrays_are_refused(tmp_path):
    _write(tmp_path, n=6)
    np.save(tmp_path / "train_text.npy",
            np.zeros((5, TEXT_LEN, TEXT_DIM), dtype=np.float16))
    with pytest.raises(ValueError, match="not aligned"):
        PairedSmilesTextDataset(tmp_path, "train")


def test_trimming_shortens_the_molecule_and_leaves_the_caption(tmp_path):
    """The two axes have equal length here, which is exactly when a shape check fails."""
    _write(tmp_path, real=5)
    ds = PairedSmilesTextDataset(tmp_path, "train")
    batch = trim_collate([ds[i] for i in range(4)], multiple_of=8, min_len=8)
    assert batch["token_ids"].shape[1] == 8            # trimmed to the molecules
    assert batch["attention_mask"].shape[1] == 8
    assert batch["text"].shape[1] == TEXT_LEN          # captions untouched
    assert batch["text_mask"].shape[1] == TEXT_LEN


def test_trimming_still_works_without_captions():
    batch = [{"token_ids": torch.ones(16, dtype=torch.long),
              "attention_mask": torch.tensor([1] * 5 + [0] * 11, dtype=torch.bool)}
             for _ in range(3)]
    out = trim_collate(batch, multiple_of=8, min_len=8)
    assert out["token_ids"].shape[1] == 8
