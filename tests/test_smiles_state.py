"""The incremental legality check, against rdkit's own verdict."""
from __future__ import annotations

import pytest

from dimol.eval.smiles_state import SmilesState, token_events

Chem = pytest.importorskip("rdkit.Chem")


def place_all(text: str) -> tuple[SmilesState, bool]:
    """Feed a string one character-ish token at a time, as the decoder does."""
    state = SmilesState()
    for token in tokenise(text):
        if not state.place(token_events(token)):
            return state, False
    return state, True


def tokenise(text: str):
    i = 0
    while i < len(text):
        if text[i] == "[":
            end = text.index("]", i)
            yield text[i : end + 1]
            i = end + 1
        elif text[i : i + 2] in ("Cl", "Br"):
            yield text[i : i + 2]
            i += 2
        elif text[i] == "%":
            yield text[i : i + 3]
            i += 3
        else:
            yield text[i]
            i += 1


LEGAL = [
    "CCO", "c1ccccc1", "CC(C)C", "C1CC1", "CC(=O)O", "C[C@H](N)C(=O)O",
    "c1ccc2ccccc2c1", "C1CC2CCC1CC2", "CC.CC", "C[NH+](C)C", "C1=CC=CC=C1",
    "c1ccc(-c2ccccc2)cc1", "[13CH4]", "C%10CCCCC%10",
]

ILLEGAL = [
    ("CC22", "one digit opened and closed on the same atom"),
    ("C(=)C", "a branch holding only a bond symbol"),
    ("C1CC1C1", "a ring left open at the end is not complete"),
    ("CC()C", "an empty branch"),
    ("CC=)C", "a branch ending on a bond symbol"),
    ("CC==C", "two bond symbols in a row"),
    ("(C)C", "a branch before any atom"),
    ("C)C", "closing a branch that was never opened"),
    ("C(1)C", "a ring digit as the whole branch"),
    ("C1CC1CC1", "an unbalanced ring digit"),
    ("=CC", "a bond symbol with nothing before it"),
]


@pytest.mark.parametrize("smiles", LEGAL)
def test_legal_strings_are_accepted_and_complete(smiles):
    state, ok = place_all(smiles)
    assert ok, smiles
    assert state.complete, smiles
    assert Chem.MolFromSmiles(smiles) is not None, smiles  # and rdkit agrees


@pytest.mark.parametrize("smiles,why", ILLEGAL)
def test_illegal_strings_are_rejected_or_left_incomplete(smiles, why):
    state, ok = place_all(smiles)
    assert not (ok and state.complete), f"{smiles} should be refused: {why}"


def test_the_two_ring_rules_that_the_parity_check_missed():
    """Both have even parity, so counting digits accepts them; rdkit does not."""
    for smiles in ("CC22", "C1CC1"):
        assert Chem.MolFromSmiles(smiles) is not None or smiles == "CC22"
    state, ok = place_all("CC22")
    assert not ok
    # a closure duplicating an existing chain bond
    state, ok = place_all("C1C1")
    assert not ok


def test_a_duplicate_bond_between_two_atoms_is_refused():
    state = SmilesState()
    assert state.place(token_events("C"))
    assert state.place(token_events("1"))
    assert state.place(token_events("C"))
    # closing ring 1 here would repeat the chain bond between atom 0 and atom 1
    assert not state.place(token_events("1"))


def test_closing_tokens_finish_a_legal_string():
    state, ok = place_all("C1CC(C")
    assert ok and not state.complete
    suffix = state.closing_tokens()
    assert suffix is not None
    finished = "C1CC(C" + "".join(suffix)
    assert Chem.MolFromSmiles(finished) is not None, finished


def test_closing_tokens_refuse_a_string_that_cannot_be_finished():
    state = SmilesState()
    for token in ("C", "1", "C"):
        state.place(token_events(token))
    # ring 1 can only close onto the atom it is already bonded to
    assert state.closing_tokens() is None


def test_a_two_letter_element_is_one_atom():
    assert [e.kind for e in token_events("Cl")] == ["atom"]
    assert [e.kind for e in token_events("CCl")] == ["atom", "atom"]


def test_a_bracket_atom_is_one_atom_and_its_digits_are_not_rings():
    assert [e.kind for e in token_events("[13C]")] == ["atom"]
    assert [e.kind for e in token_events("[C@@H]")] == ["atom"]


def test_a_truncated_bracket_cannot_be_placed():
    assert token_events("[C@@H")[0].kind == "invalid"
    assert not SmilesState().place(token_events("[C@@H"))


def test_multi_character_fragments_replay_every_character():
    kinds = [e.kind for e in token_events("c1cc(")]
    assert kinds == ["atom", "ring", "atom", "atom", "open"]


def test_placing_a_rejected_token_leaves_the_state_untouched():
    state = SmilesState()
    state.place(token_events("C"))
    before = state.copy()
    assert not state.place(token_events(")"))
    assert state.atoms == before.atoms and state.last == before.last


@pytest.mark.parametrize("smiles", [
    "O=C(COCCOc1ccccc11)C2CC2N1Cc1ccccc1",
    "C[C@H](c1ccc(-c2ccccc2)cc1)c1nnc([NH+]2CCC[C@H]22)cc12",
    "CCc1cn(C(=O)N1CCCN(C(N)=O)CC1)c2ccccc2C11CCOCC1",
])
def test_one_digit_twice_on_one_atom_is_legal_when_it_closes_then_opens(smiles):
    """Spiro and fused atoms do exactly that, and rdkit accepts them."""
    assert Chem.MolFromSmiles(smiles) is not None, smiles
    state, ok = place_all(smiles)
    assert ok and state.complete, smiles


def test_but_opening_and_closing_on_the_same_atom_is_still_refused():
    state, ok = place_all("CC22")
    assert not ok
