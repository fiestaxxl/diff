"""Caption ids written beside the states: the arrays, and the alignment proof.

Putting SciBERT inside the training loop means the batch carries token ids instead of
precomputed states, and nothing in a tensor's shape says whether those ids are the same
tokenization that produced the states every stored number in this project came from. The
assertion below is that proof, and these tests pin it: a mask that disagrees, or a row
count that disagrees, must stop the write rather than produce a corpus that trains fine
and conditions on the wrong caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tokenize_captions import write_caption_ids  # noqa: E402

CAPTIONS = ["a molecule", "a longer molecule description", "short", "two words"]
TEXT_LEN = 8
CANVAS = 6


class _StubTokenizer:
    """One id per word plus a leading class token, padded to ``max_length``.

    A stub rather than SciBERT, so the alignment logic is testable on a machine with no
    transformers and no 400 MB of weights; the real tokenizer is wired in by ``main``.
    """

    def __call__(self, batch, padding=None, truncation=None, max_length=None,
                 return_tensors=None):
        ids = np.zeros((len(batch), max_length), dtype=np.int64)
        attn = np.zeros((len(batch), max_length), dtype=np.int64)
        for row, text in enumerate(batch):
            words = ([101] + [100 + len(w) for w in text.split()] + [102])[:max_length]
            ids[row, : len(words)] = words
            attn[row, : len(words)] = 1
        return {"input_ids": ids, "attention_mask": attn}


def _corpus(tmp_path: Path, rows: int = len(CAPTIONS), mask_rows: int | None = None):
    """The parts of a prepared corpus this script reads, and nothing else."""
    (tmp_path / "train_captions.txt").write_text("\n".join(CAPTIONS[:rows]) + "\n")
    np.save(tmp_path / "train_tokens_00000.npy", np.zeros((rows, CANVAS), dtype=np.uint16))
    stub = _StubTokenizer()
    truth = stub(CAPTIONS[:rows], max_length=TEXT_LEN)["attention_mask"]
    if mask_rows is not None:
        truth = truth[:mask_rows]
    np.save(tmp_path / "train_text_mask.npy", truth.astype(np.uint8))
    return stub


def test_the_ids_and_the_mask_are_written_in_the_storage_dtypes(tmp_path):
    """uint16 ids and uint8 mask: 12 MB for the corpus instead of 5 GB of states."""
    stub = _corpus(tmp_path)
    write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)

    ids = np.load(tmp_path / "train_caption_ids.npy")
    attn = np.load(tmp_path / "train_caption_attn.npy")
    assert ids.shape == (len(CAPTIONS), TEXT_LEN) and ids.dtype == np.uint16
    assert attn.shape == (len(CAPTIONS), TEXT_LEN) and attn.dtype == np.uint8
    # row 2 is "short": a class token, one word, a separator, then padding
    assert attn[2].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert ids[2, 3:].tolist() == [0] * 5


def test_the_ids_are_row_aligned_with_the_captions(tmp_path):
    stub = _corpus(tmp_path)
    write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)
    ids = np.load(tmp_path / "train_caption_ids.npy")
    for row, caption in enumerate(CAPTIONS):
        assert int(ids[row].astype(bool).sum()) == len(caption.split()) + 2


def test_a_tokenization_that_disagrees_with_the_stored_mask_is_refused(tmp_path):
    """The one failure that leaves no trace: a caption silently re-cut a different way."""
    stub = _corpus(tmp_path)
    moved = np.load(tmp_path / "train_text_mask.npy")
    moved[1, -1] = 1  # one caption one token longer than the states say it was
    np.save(tmp_path / "train_text_mask.npy", moved)
    with pytest.raises(ValueError, match="tokenize differently"):
        write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)
    assert not (tmp_path / "train_caption_ids.npy").exists()


def test_a_row_count_that_disagrees_with_the_molecules_is_refused(tmp_path):
    stub = _corpus(tmp_path)
    np.save(tmp_path / "train_tokens_00000.npy",
            np.zeros((len(CAPTIONS) - 1, CANVAS), dtype=np.uint16))
    with pytest.raises(ValueError, match="no longer row-aligned"):
        write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)


def test_a_mask_of_the_wrong_width_or_height_is_refused(tmp_path):
    stub = _corpus(tmp_path, mask_rows=len(CAPTIONS) - 1)
    with pytest.raises(ValueError, match="has moved since the states were written"):
        write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)


def test_a_corpus_with_no_stored_states_says_which_stage_to_run(tmp_path):
    stub = _corpus(tmp_path)
    (tmp_path / "train_text_mask.npy").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_paired"):
        write_caption_ids(stub, tmp_path, ["train"], TEXT_LEN)
