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


def test_atoms_outside_the_corpus_vocabulary_are_replaced():
    """[P@@] never occurs in the corpus, so the model's next choice is used instead."""
    out = decode(["C", "[P@@]", "O"], second=[None, "N", None],
                 allowed_brackets={"[13C]"})
    assert out == "CNO"


def test_atoms_outside_the_vocabulary_can_be_dropped_instead():
    out = decode(["C", "[P@@]", "O"], second=[None, "N", None],
                 allowed_brackets={"[13C]"}, on_disallowed="skip")
    assert out == "CO"


def test_an_allowed_bracket_atom_is_left_alone():
    out = decode(["C", "[13C]", "O"], second=[None, "N", None],
                 allowed_brackets={"[13C]"})
    assert out == "C[13C]O"


def test_without_a_vocabulary_nothing_is_banned():
    assert decode(["C", "[P@@]", "O"]) == "C[P@@]O"


def test_the_grammar_substitution_also_respects_the_vocabulary():
    """A grammar fix must not reach for a banned atom either."""
    out = decode(["C", ")", "O"], second=[None, "[P@@]", None],
                 substitution="next_best", allowed_brackets={"[13C]"})
    assert "P" not in out


def test_corpus_brackets_reads_a_file_and_a_directory(tmp_path):
    from dimol.eval.decoding import corpus_brackets

    one = tmp_path / "train.txt"
    one.write_text("C[NH+](C)C\nc1cc[nH]c1\nCCO\n")
    assert corpus_brackets(one) == {"[NH+]", "[nH]"}

    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "a.txt").write_text("CC(=O)[O-]\n")
    (shards / "b.txt").write_text("C[C@H](N)C\n")
    assert corpus_brackets(shards) == {"[O-]", "[C@H]"}
    assert corpus_brackets(one, limit=1) == {"[NH+]"}


def test_strict_decoding_refuses_a_ring_closed_onto_its_own_atom():
    """CC22 has even digit parity, so only the strict state catches it."""
    loose = decode(["C", "C", "2", "2", "<eos>"])
    strict = decode(["C", "C", "2", "2", "<eos>"], strict=True)
    assert loose == "CC22"
    assert strict == "CC"  # the second 2 is refused, then the first one has to go


def test_strict_decoding_refuses_a_closure_that_duplicates_a_bond():
    """The closure is dropped, and the ring opener it left behind forces a trim back."""
    assert decode(["C", "1", "C", "1", "<eos>"], strict=True) == "C"


def test_strict_decoding_keeps_a_legal_ring():
    assert decode(["C", "1", "C", "C", "1", "<eos>"], strict=True) == "C1CC1"


def test_strict_decoding_closes_what_it_can_at_the_end():
    out = decode(["C", "1", "C", "C", "(", "C", "<eos>"], strict=True, repair="close")
    assert out == "C1CC(C1)"


def test_strict_decoding_trims_when_nothing_legal_closes_the_string():
    out = decode(["C", "1", "C", "<eos>"], strict=True, repair="close")
    assert out == "C"  # closing ring 1 here would repeat the chain bond


def test_strict_decoding_still_honours_the_atom_vocabulary():
    out = decode(["C", "[P@@]", "O", "<eos>"], second=[None, "N", None],
                 strict=True, allowed_brackets={"[13C]"})
    assert out == "CNO"


def test_a_stop_token_before_the_floor_is_ignored():
    """Length conditioning needs the decoder to refuse an early end."""
    decoder = GrammarConstrainedDecoder(StubTokenizer(), canvas=6)
    logits = logits_for(["C", "<eos>", "C", "C", "<eos>", "<pad>"],
                        second=[None, "N", None, None, None, None], canvas=6)
    assert decoder.decode(logits)[0] == "C"
    assert decoder.decode(logits, min_length=torch.tensor([3]))[0] == "CNCC"


def test_the_floor_does_not_extend_past_what_the_model_gives():
    decoder = GrammarConstrainedDecoder(StubTokenizer(), canvas=4)
    logits = logits_for(["C", "C", "<eos>", "<pad>"], canvas=4)
    out = decoder.decode(logits, min_length=torch.tensor([99]))[0]
    assert out.startswith("CC") and len(out) <= 4


def test_the_floor_is_per_sample():
    decoder = GrammarConstrainedDecoder(StubTokenizer(), canvas=5)
    a = logits_for(["C", "<eos>", "C", "C", "<pad>"], second=[None, "N", None, None, None],
                   canvas=5)
    logits = torch.cat([a, a], dim=0)
    out = decoder.decode(logits, min_length=torch.tensor([0, 3]))
    assert out[0] == "C" and out[1] == "CNCC"


def test_strict_decoding_honours_the_floor_too():
    decoder = GrammarConstrainedDecoder(StubTokenizer(), canvas=6, strict=True)
    logits = logits_for(["C", "<eos>", "C", "C", "<eos>", "<pad>"],
                        second=[None, "N", None, None, None, None], canvas=6)
    assert decoder.decode(logits)[0] == "C"
    assert decoder.decode(logits, min_length=torch.tensor([3]))[0] == "CNCC"
