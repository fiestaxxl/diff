"""Grammar repair: what it fixes, and what it must never invent.

The regression these tests exist for: an earlier repair, when the model asked for a ")"
that closes nothing, fell back to the next most likely token, and that token is often an
atom. Validity went up and the molecules acquired atoms the model never asked for.
"""
from __future__ import annotations

import torch

from dimol.eval.decoding import GrammarConstrainedDecoder, _token_grammar, build_decoder

TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "C", "N", "O", "(", ")", "1", "2",
          "[P@@]", "[13C]", "c1ccccc1"]
VOCAB = {token: i for i, token in enumerate(TOKENS)}


class StubTokenizer:
    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]
    eos_id = VOCAB["<eos>"]
    unk_id = VOCAB["<unk>"]

    def get_vocab(self):
        return dict(VOCAB)


def logits_for(sequence, second=None, canvas=None):
    """Logits whose argmax is `sequence`, with `second` as the runner-up per position."""
    canvas = canvas or len(sequence)
    out = torch.full((1, canvas, len(TOKENS)), -10.0)
    for pos in range(canvas):
        token = sequence[pos] if pos < len(sequence) else "<pad>"
        out[0, pos, VOCAB[token]] = 5.0
        runner = (second[pos] if second and pos < len(second) else None)
        if runner:
            out[0, pos, VOCAB[runner]] = 4.0
    return out


def decode(sequence, second=None, canvas=None, **kwargs):
    decoder = GrammarConstrainedDecoder(StubTokenizer(), canvas=canvas or 32, **kwargs)
    return decoder.decode(logits_for(sequence, second, canvas or 32))[0]


def test_a_legal_string_passes_through_untouched():
    assert decode(["C", "C", "O", "<eos>"]) == "CCO"


def test_bos_is_skipped_and_eos_ends_the_molecule():
    assert decode(["<bos>", "C", "N", "<eos>", "C"]) == "CN"


def test_padding_also_ends_the_molecule():
    assert decode(["C", "O", "<pad>", "C"]) == "CO"


def test_a_stray_closing_paren_is_dropped_not_replaced():
    """The regression test: the runner-up is an atom and must not appear."""
    out = decode(["C", ")", "O"], second=[None, "[P@@]", None])
    assert out == "CO"
    assert "P" not in out


def test_next_best_substitution_does_insert_the_atom():
    """Documents the behaviour that is now off by default."""
    out = decode(["C", ")", "O"], second=[None, "[P@@]", None],
                 substitution="next_best")
    assert out == "C[P@@]O"


def test_closing_appends_the_missing_ring_digit_and_paren():
    out = decode(["C", "1", "C", "(", "C"], repair="close")
    assert out == "C1C(C1)"


def test_trimming_cuts_back_to_the_last_balanced_point():
    out = decode(["C", "C", "(", "N", "1"], repair="trim")
    assert out == "CC"


def test_mixed_keeps_the_last_paired_ring_then_closes_branches():
    out = decode(["C", "1", "C", "1", "(", "N", "2"], repair="mixed")
    assert out == "C1C1(N)"


def test_a_string_that_needs_no_repair_is_identical_in_every_mode():
    sequence = ["C", "1", "C", "C", "1", "<eos>"]
    outs = {mode: decode(sequence, repair=mode) for mode in ("close", "trim", "mixed")}
    assert set(outs.values()) == {"C1CC1"}


def test_digits_inside_brackets_are_isotopes_not_ring_closures():
    delta, lowest, parity = _token_grammar("[13C]")
    assert delta == 0 and lowest == 0 and parity.sum() == 0
    assert decode(["[13C]", "C", "<eos>"], repair="close") == "[13C]C"


def test_multi_character_tokens_are_accounted_for_as_a_whole():
    delta, lowest, parity = _token_grammar("c1ccccc1")
    assert parity.sum() == 0  # the ring opens and closes inside the token
    assert decode(["c1ccccc1", "<eos>"]) == "c1ccccc1"


def test_the_mode_names_map_to_the_documented_behaviour():
    assert build_decoder(StubTokenizer(), 32, "argmax") is None
    for name, repair in (("grammar", "close"), ("grammar_close", "close"),
                         ("grammar_trim", "trim"), ("grammar_mixed", "mixed")):
        decoder = build_decoder(StubTokenizer(), 32, name)
        assert decoder.repair == repair and decoder.substitution == "skip"
    subst = build_decoder(StubTokenizer(), 32, "grammar_close_subst")
    assert subst.substitution == "next_best"


def test_an_unknown_mode_is_reported():
    try:
        build_decoder(StubTokenizer(), 32, "grammar_magic")
    except ValueError as err:
        assert "expected" in str(err)
    else:
        raise AssertionError("an unknown decode mode must be rejected")
