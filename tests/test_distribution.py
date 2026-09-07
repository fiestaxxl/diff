"""Distribution metrics: descriptors, the trivial-molecule rule, scaffolds."""
from __future__ import annotations

import math

from dimol.eval.distribution import (
    DESCRIPTOR_NAMES,
    compare_descriptors,
    describe,
    evaluate_distribution,
    trivial_share,
)

DRUGLIKE = [
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "COc1cc2c(cc1OC)CCN(C)C2",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2[nH]c3ccccc3c2c1",
]
TINY = ["CCO", "CC", "CCN", "C", "CO", "CCC"]


def test_descriptors_are_all_present_and_finite():
    out = describe(DRUGLIKE)
    for name in DESCRIPTOR_NAMES:
        column = out["columns"][name]
        assert len(column) == len(DRUGLIKE), name
        assert all(math.isfinite(v) for v in column), name


def test_invalid_strings_are_dropped_not_counted():
    out = describe(["CCO", "not-a-molecule", "c1ccccc1", ""])
    assert len(out["canonical"]) == 2


def test_the_trivial_rule_separates_tiny_from_druglike():
    tiny = trivial_share(describe(TINY)["columns"])
    real = trivial_share(describe(DRUGLIKE)["columns"])
    assert tiny["trivial"] == 1.0
    assert real["trivial"] == 0.0


def test_a_shift_is_measured_in_reference_sigmas():
    sample = describe(TINY)["columns"]
    reference = describe(DRUGLIKE)["columns"]
    rows = compare_descriptors(sample, reference)
    assert rows["heavy_atoms"]["shift_sigma"] < -1.0  # much smaller molecules
    assert rows["rings"]["ks"] > 0.9  # and a completely different ring distribution


def test_the_same_set_against_itself_has_no_shift():
    columns = describe(DRUGLIKE)["columns"]
    rows = compare_descriptors(columns, columns)
    for name in DESCRIPTOR_NAMES:
        assert abs(rows[name]["shift_sigma"]) < 1e-9, name
        assert rows[name]["ks"] < 1e-9, name


def test_scaffolds_and_the_report_shape():
    report = evaluate_distribution(DRUGLIKE, DRUGLIKE, with_fcd=False)
    assert report["n_valid"] == len(DRUGLIKE)
    assert report["scaffolds"]["scaffolds"] >= 3
    assert report["scaffolds"]["scaffold_novelty"] == 0.0  # same set, nothing unseen
    assert "fcd" not in report


def test_a_molecule_that_parses_but_breaks_a_descriptor_is_counted_not_crashed():
    """rdkit accepts this string and then QED refuses to re-kekulize it."""
    awkward = "Cc1ccc(C)n([C@@H](C)C)=C(Nc1ccccc1)N2CCOCC21CCCN"
    out = describe(["CCO", awkward, "c1ccccc1"])
    assert len(out["canonical"]) == 3  # all three parse
    assert out["undescribed"] == 1  # one has no descriptors
    assert len(out["columns"]["heavy_atoms"]) == 2  # and is left out of the statistics

    report = evaluate_distribution(["CCO", awkward, "c1ccccc1"], DRUGLIKE, with_fcd=False)
    assert report["n_valid"] == 3 and report["n_described"] == 2
    assert report["n_undescribed"] == 1


def test_bracket_atoms_the_corpus_never_uses_are_flagged():
    from dimol.eval.distribution import bracket_report

    corpus = ["C[NH+](C)C", "c1cc[nH]c1", "CC(=O)[O-]"]
    generated = ["C[NH+](C)C", "CC[P]C", "C[CH]C", "CC(=O)[O-]"]
    out = bracket_report(generated, corpus)
    assert out["unseen_kinds"] == 2
    assert abs(out["molecules_with_unseen"] - 0.5) < 1e-9
    assert dict(out["top_unseen"]).keys() == {"[P]", "[CH]"}


def test_a_set_drawn_from_the_corpus_flags_nothing():
    from dimol.eval.distribution import bracket_report

    corpus = ["C[NH+](C)C", "c1cc[nH]c1"]
    out = bracket_report(["C[NH+](C)C"], corpus)
    assert out["unseen_kinds"] == 0 and out["molecules_with_unseen"] == 0.0


def test_fcd_input_is_filtered_to_strings_rdkit_can_read_back():
    from dimol.eval.distribution import roundtrip_safe

    awkward = "Cc1ccc(C)n(C(C)C)c2[nH]c1=CC=CC=CC=NC1COCCN21"
    kept = roundtrip_safe(["CCO", awkward, "", "c1ccccc1"])
    assert kept == ["CCO", "c1ccccc1"]


def test_usable_counts_only_molecules_a_chemist_would_accept():
    from dimol.eval.distribution import usable_molecules

    corpus = ["C[NH+](C)C", "c1ccc2[nH]c3ccccc3c2c1"]
    attempts = [
        "CC(=O)Nc1ccc(O)cc1",   # fine
        "CC(=O)Nc1ccc(O)cc1",   # duplicate
        "CCO",                  # too small, no ring
        "CC(=O)Nc1ccc(O)cc1[P]",  # bracket atom the corpus never uses
        "not-a-molecule",       # invalid
        "",                     # empty
        "COc1cc2c(cc1OC)CCN(C)C2",  # fine
    ]
    out = usable_molecules(attempts, corpus)
    assert out["count"] == 2
    assert out["attempts"] == 7
    assert out["rejected"] == {"invalid": 2, "duplicate": 1, "too_small": 1,
                               "unseen_atom": 1}


def test_usable_is_a_share_of_attempts_not_of_valid_strings():
    from dimol.eval.distribution import usable_molecules

    out = usable_molecules(["CC(=O)Nc1ccc(O)cc1"] + [""] * 9, ["CCO"])
    assert out["count"] == 1 and abs(out["rate"] - 0.1) < 1e-9
