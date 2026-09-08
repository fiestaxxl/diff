"""Tokenizer: training on a tiny corpus, roundtrip, padding, audit, save/load."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from dimol.tokenization.audit import audit_tokenizer  # noqa: E402
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402

SAMPLE = [
    "CN(N=O)C(=N)O",
    "c1ccc2c(-c3nccs3)c[nH]c2c1",
    "NCC([O-])=NC1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
    "CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-]",
    "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccccc1",
    "c1ccc(C(=O)O)cc1",
] * 20


@pytest.fixture(scope="module")
def tok() -> SmilesTokenizer:
    return SmilesTokenizer.train(
        smiles_iter=SAMPLE,
        vocab_size=300,
        min_frequency=2,
        bracket_min_frequency=1,
        show_progress=False,
    )


def test_special_tokens_first(tok: SmilesTokenizer) -> None:
    assert tok.pad_id == 0
    assert {tok.bos_id, tok.eos_id, tok.unk_id, tok.mask_id} == {1, 2, 3, 4}
    assert tok.vocab_size <= 300


def test_roundtrip_without_specials(tok: SmilesTokenizer) -> None:
    for smi in set(SAMPLE):
        ids = tok.encode(smi, add_special_tokens=False)
        assert tok.decode(ids) == smi, smi


def test_bracket_atoms_are_atomic(tok: SmilesTokenizer) -> None:
    """[...] must stay a single token: that is the key property of this tokenizer."""
    tokens = tok.tokenize("NCC([O-])=NC1O[C@H](CO)O")
    assert "[O-]" in tokens and "[C@H]" in tokens


def test_special_tokens_wrap_sequence(tok: SmilesTokenizer) -> None:
    ids = tok.encode("c1ccccc1", add_special_tokens=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_encode_padded(tok: SmilesTokenizer) -> None:
    ids, mask = tok.encode_padded("c1ccccc1", max_length=32)
    assert len(ids) == 32 and len(mask) == 32
    assert sum(mask) == len(tok.encode("c1ccccc1", add_special_tokens=True))
    assert all(i == tok.pad_id for i in ids[sum(mask):])


def test_encode_padded_too_long_returns_none(tok: SmilesTokenizer) -> None:
    ids, mask = tok.encode_padded("CCCCCCCCCCCCCCCC(=O)O", max_length=4)
    assert ids is None and mask is None


def test_special_decode_stops_at_eos(tok: SmilesTokenizer) -> None:
    """special_decode truncates at the first <eos>, which is how samples are decoded."""
    ids = tok.encode("c1ccccc1", add_special_tokens=True)
    noisy = ids + [tok.bos_id] + tok.encode("CCO", add_special_tokens=False)
    assert tok.decode_batch([noisy], special_decode=True)[0] == "c1ccccc1"


def test_audit(tok: SmilesTokenizer) -> None:
    report = audit_tokenizer(tok, list(set(SAMPLE)))
    assert report["n_molecules"] == len(set(SAMPLE))
    assert report["n_unk_tokens"] == 0
    assert report["roundtrip_failures"] == []
    assert report["chars_per_token"] > 1.0
    assert report["orphan_bracket_tokens"] == []


def test_save_load(tok: SmilesTokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = SmilesTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("c1ccccc1") == tok.encode("c1ccccc1")


def test_added_tokens_extend_the_vocabulary_without_moving_ids(tmp_path):
    """A tokenizer trained on one corpus has to stretch to a second one without
    discarding a pretrained embedding table, so existing ids must not move."""
    tok = SmilesTokenizer.train(SAMPLE, vocab_size=200, min_frequency=1,
                                show_progress=False)
    before = tok.get_vocab()
    added = tok.add_tokens(["[Na+]", "[2H]", ".", "%10"])
    after = tok.get_vocab()

    assert added == 4
    assert len(after) == len(before) + 4
    for token, index in before.items():
        assert after[token] == index, token
    for token in ("[Na+]", "[2H]", ".", "%10"):
        assert after[token] >= len(before)


def test_added_tokens_are_not_split_further(tmp_path):
    tok = SmilesTokenizer.train(SAMPLE, vocab_size=200, min_frequency=1,
                                show_progress=False)
    tok.add_tokens(["[Na+]", ".", "%10"])
    ids = tok.encode("CC.[Na+]", add_special_tokens=False)
    pieces = [tok.decode([i], skip_special_tokens=False) for i in ids]
    assert "[Na+]" in pieces
    assert "." in pieces


def test_adding_nothing_is_a_no_op():
    tok = SmilesTokenizer.train(SAMPLE, vocab_size=200, min_frequency=1,
                                show_progress=False)
    size = len(tok.get_vocab())
    assert tok.add_tokens([]) == 0
    assert len(tok.get_vocab()) == size


def test_an_extended_tokenizer_survives_a_save_and_load(tmp_path):
    tok = SmilesTokenizer.train(SAMPLE, vocab_size=200, min_frequency=1,
                                show_progress=False)
    tok.add_tokens(["[Fe]", "[Se]"])
    path = tmp_path / "extended.json"
    tok.save(path)
    again = SmilesTokenizer.load(path)
    assert again.get_vocab() == tok.get_vocab()
    assert again.encode("C[Fe]C") == tok.encode("C[Fe]C")
