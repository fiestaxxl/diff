"""The rdkit-free halves of the text evaluation: grammar and token overlap."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

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
