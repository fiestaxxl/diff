#!/usr/bin/env python3
"""Score a generated set against the ChEBI-20 text-to-molecule benchmark table.

    python scripts/score_benchmark.py --dir scratch/ev/test_48m

The published table (MolT5, TGM-DLM and the rest) reports nine columns, and the five we
had been reporting are not enough to place a row in it. This adds the four missing ones
and fixes the denominators, which differ per metric in the reference implementation:

* BLEU, Levenshtein and exact match are computed over **every** caption, invalid outputs
  included, because a model that emits nothing must not be rewarded for it.
* The three fingerprint similarities are computed over pairs where **both** molecules
  parse, which is the convention the baselines use; the count is reported so a row can
  never hide how few pairs it was averaged over.
* FCD is computed between the valid generations and the references of those same pairs,
  so the reference distribution is built per row and changes with what the row got
  valid. This makes the column non-comparable across rows, including our own. See
  frechet_distance() before quoting an FCD anywhere.
* Validity is the share of outputs RDKit accepts **and** that carry at least three heavy
  atoms. An empty string parses in RDKit and would otherwise count as a valid molecule.

Text2Mol is the one column this cannot produce: it needs the external retrieval model
from Edwards et al., which is not on either machine. The column is emitted as None so
the shape of the table is honest about it. It is not printed, so the table this script
prints is eight of the published nine.

This scorer does not reproduce published numbers. We ran the published MolT5-base
checkpoint (laituan245/molt5-base-caption2smiles) ourselves, on our split, through this
file: BLEU 0.527 against its published 0.769, Levenshtein 40.34 against 24.46, FCD 1.63
against 2.18. Same checkpoint, same split, same scorer, and six of the eight columns
move. The comparison this script prints is therefore indicative only. The full statement
of what that costs is in the comment above REFERENCE_ROWS; read it before quoting a rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import Levenshtein
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

MIN_HEAVY_ATOMS = 3


def canonical(smiles: str):
    """Parsed molecule and canonical string, or (None, None) if RDKit refuses it."""
    if not smiles or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        return mol, Chem.MolToSmiles(mol)
    except Exception:
        return None, None


def fingerprint_similarities(gen, ref):
    maccs = TanimotoSimilarity(MACCSkeys.GenMACCSKeys(gen), MACCSkeys.GenMACCSKeys(ref))
    rdk = TanimotoSimilarity(Chem.RDKFingerprint(gen), Chem.RDKFingerprint(ref))
    morgan = TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(gen, 2, 2048),
        AllChem.GetMorganFingerprintAsBitVect(ref, 2, 2048),
    )
    return maccs, rdk, morgan


def frechet_distance(generated, references):
    """FCD between the valid generations and the references of the pairs they came from.

    The reference set is built per row, not fixed. `references` is `paired_references`
    from score(): one canonical reference for each caption where *this* row produced a
    valid molecule and the reference itself parsed. A row that answers fewer captions is
    scored against fewer references, so two rows are measured against two different
    target distributions and their FCDs are not comparable: not against a published row,
    and not against each other. Recorded examples, all from this scorer: the
    1,200-step single-trajectory row scores 0.714 against 2,505 references, the
    10-candidate row's worst seed scores 0.744 against 3,178, and the TF-IDF retrieval
    floor, which generates nothing, scores 0.481 against 3,182. The row with the easier
    target is also the row with the larger one, and nothing in the number says so.

    One recorded FCD has already moved this way: 2.20 on 1,000 captions against 1.23 on
    the full split. The retracted-claims ledger attributes that to the Frechet estimator
    being biased upward on a small sample. The reference set also changed size between
    those two scorings, so the two explanations have never been separated.

    The fix is a fixed reference set: compute FCD against all canonical test references
    for every row, independent of which captions the row got valid, and report the
    generation count beside it. A row that answers fewer captions is then penalised by
    the distance instead of rewarded with a smaller target.

    Do not make that change in place. Every FCD in EXPERIMENTS.md, PAPER.md and
    scratchpad/metrics.txt was produced by the computation below; changing it silently
    would make the whole record unreadable, because nothing would mark which numbers came
    from which definition. A fixed-reference FCD goes in beside this one, under its own
    key, with every row re-scored before either number is quoted.
    """
    try:
        from fcd_torch import FCD
    except ImportError:
        return None
    if len(generated) < 2:
        return None
    try:
        return float(FCD(device="cpu", n_jobs=1)(ref=references, gen=generated))
    except Exception:
        return None


def score(generated: list[str], references: list[str]) -> dict:
    n = min(len(generated), len(references))
    generated, references = generated[:n], references[:n]

    bleu_refs, bleu_hyps = [], []
    distances, exact = [], 0
    valid = 0
    maccs_scores, rdk_scores, morgan_scores = [], [], []
    valid_generated, paired_references = [], []
    unparseable_reference = 0

    for gen_text, ref_text in zip(generated, references):
        gen_mol, gen_canon = canonical(gen_text)
        ref_mol, ref_canon = canonical(ref_text)
        if ref_mol is None:
            # A health check on the reference decode, not a property of ChEBI-20. This
            # counter was documented here as "a handful of captions with no structure at
            # all (isotope entries)", unanswerable by construction. That was wrong. The
            # evaluation tokenizer was reading the corpus with the 391-token ZINC
            # vocabulary instead of the 635-token shared one, and 95 of the 3,202
            # references decoded to strings RDKit refuses; the affected rows are salts
            # and complexes, not isotopes. After the tokenizer fix the count is 0 on the
            # full split. Read a non-zero count as "this row was scored against a damaged
            # reference set", and two rows with different counts as not comparable.
            unparseable_reference += 1

        # String metrics: every caption counts, an empty output included.
        bleu_refs.append([list(ref_text)])
        bleu_hyps.append(list(gen_text))
        distances.append(Levenshtein.distance(gen_text, ref_text))
        if gen_canon is not None and ref_canon is not None and gen_canon == ref_canon:
            exact += 1

        if gen_mol is not None and gen_mol.GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS:
            valid += 1
            if ref_mol is not None:
                valid_generated.append(gen_canon)
                paired_references.append(ref_canon)
                a, b, c = fingerprint_similarities(gen_mol, ref_mol)
                maccs_scores.append(a)
                rdk_scores.append(b)
                morgan_scores.append(c)

    smoothing = SmoothingFunction().method1
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

    return {
        "n": n,
        "bleu": corpus_bleu(bleu_refs, bleu_hyps, smoothing_function=smoothing),
        "exact": exact / n,
        "levenshtein": mean(distances),
        "maccs": mean(maccs_scores),
        "rdk": mean(rdk_scores),
        "morgan": mean(morgan_scores),
        "fcd": frechet_distance(valid_generated, paired_references),
        "text2mol": None,  # needs the external retrieval model
        "validity": valid / n,
        "pairs_scored": len(maccs_scores),
        "unparseable_reference": unparseable_reference,
    }


# Published rows. Read this before quoting any of them against one of ours.
#
# PROVENANCE. Eleven of the thirteen rows below are Table 1 of the LDMol paper: Jinho
# Chang and Jong Chul Ye, "LDMol: A Text-to-Molecule Diffusion Model with Structurally
# Informative Latent Space Surpasses AR Models", ICML 2025, arXiv:2405.17829v4; code at
# https://github.com/jinhojsk515/LDMol. Caption-to-molecule on the standard 3,300-pair
# ChEBI-20 test split. The PDF was supplied by the project owner, and this is the first
# table in the project with a source. The rows below keep that table's order: the nine
# autoregressive systems from Transformer to BioT5+, then the two diffusion ones, TGM-DLM
# and LDMol. Its MACCS, RDK and Morgan are Tanimoto similarities, the same quantity this
# file computes.
#
# The two rows that are not from it:
#   Ground truth      - definitional rather than transcribed: the references scored
#                       against themselves. The ceiling of each column, not a system.
#   TGM-DLM w/o corr  - the correction-free variant, which LDMol Table 1 does not report.
#                       It is all that is left of the five rows transcribed from a chat
#                       message in commit 29ad1f6 (2026-09-25, "Score the published
#                       ChEBI-20 columns, and take lengths from a file"), which recorded
#                       no paper, table number, year or URL. Its likely source is the
#                       TGM-DLM paper (Gong et al., AAAI 2024), still UNVERIFIED in-repo.
#                       It is now the only unsourced row here; quote it as such.
#
# WHAT THE SOURCE CHANGED. Of the 40 numbers transcribed in 29ad1f6, the Transformer,
# MolT5-base and TGM-DLM rows agree with the source cell for cell, and T5-base agrees in
# seven of its eight. The eighth does not: T5-base BLEU is 0.765 in the source and was
# transcribed as 0.762. The sourced value stands. 0.762 is the figure the "third on BLEU"
# ranks elsewhere in the repo were computed against; on either figure those margins are
# 0.012 to 0.019, well inside the 0.242 BLEU shift below. Six systems in the source were
# never in this table at all - GIT-Mol, T5-large, MolT5-large, MolXPT, BioT5, BioT5+ - and
# five of the six sit above both our best row and the TF-IDF retrieval floor on all three
# fingerprint columns. GIT-Mol is the one that does not.
#
# CALIBRATION, and this is the thing to know about this file. The scorer does not
# reproduce published numbers. We ran the published MolT5-base checkpoint
# (laituan245/molt5-base-caption2smiles) ourselves, on our split, through this code:
# BLEU 0.527 against the published 0.769, Levenshtein 40.34 against 24.46, FCD 1.63
# against 2.18. Same checkpoint, same split, same scorer, and six of the eight columns
# move. Column by column the shift is BLEU 0.242, Levenshtein 15.9, FCD 0.553, exact
# 0.038, MACCS 0.033, Morgan 0.017, validity 0.010 and RDK 0.005. A cross-paper margin
# smaller than its own column's shift is noise and has to be written as noise. That is
# the only published system we have ever run; no TGM-DLM, LDMol, BioT5, MolXPT, T5 or
# Transformer output has been through this code at all. So what gets printed is two
# tables stacked, not one ranking: our rows are measured, the published rows come from
# other scorers, and no cross-paper rank read off the printout is sound until the
# calibration gap is explained.
#
# SPLIT. The published rows are on the standard 3,300-pair ChEBI-20 test split. Our rows
# are on our curated 3,202-caption split: scripts/prepare_paired.py drops a pair whole
# when the reference does not parse or exceeds the 128-token canvas, 98 pairs, 3.0%, and
# those are the unparsable and the longest molecules, the ones a 128-token model cannot
# emit. The drop is not neutral with respect to validity or exact match, and it is
# accounted for: counting the 98 as failures moves our worst-seed validity from 0.993 to
# 0.963 and our exact match from 0.143 to 0.138, and it takes the K=5 AR-selection row's
# 0.949 to 0.921, below LDMol's 0.941 - a validity ranking that survives the raw numbers
# and not the adjustment. Apply it before quoting any row of ours against a published one.
# It is still nowhere near large enough to explain the calibration gap above.
REFERENCE_ROWS = [
    # (name, BLEU, exact, Levenshtein, MACCS, RDK, Morgan, FCD, validity). None where a
    # paper does not report the column; cell() prints it as "n/a".
    ("Ground truth", 1.000, 1.000, 0.000, 1.000, 1.000, 1.000, 0.00, 1.000),
    # LDMol Table 1, autoregressive rows, in the order that table prints them.
    ("Transformer", 0.499, 0.000, 57.660, 0.480, 0.320, 0.217, 11.32, 0.906),
    ("GIT-Mol", 0.756, 0.051, 26.315, 0.738, 0.582, 0.519, None, 0.928),  # no FCD
    ("T5-base", 0.765, 0.069, 24.950, 0.731, 0.605, 0.545, 2.48, 0.660),
    ("MolT5-base", 0.769, 0.081, 24.458, 0.721, 0.588, 0.529, 2.18, 0.772),
    ("T5-large", 0.854, 0.279, 16.721, 0.823, 0.731, 0.670, 1.22, 0.902),
    ("MolT5-large", 0.854, 0.311, 16.071, 0.834, 0.746, 0.684, 1.20, 0.905),
    ("MolXPT", None, 0.215, None, 0.859, 0.757, 0.667, 0.45, 0.983),  # no BLEU, no Leven
    ("BioT5", 0.867, 0.413, 15.097, 0.886, 0.801, 0.734, 0.43, 1.000),
    ("BioT5+", 0.872, 0.522, 12.776, 0.907, 0.835, 0.779, 0.35, 1.000),
    # LDMol Table 1, diffusion rows. BioT5 and BioT5+ reach validity 1.000 by emitting
    # SELFIES, which cannot encode an invalid molecule; that column is saturable and not
    # a measure of the same thing for every row here.
    ("TGM-DLM", 0.826, 0.242, 17.003, 0.854, 0.739, 0.688, 0.77, 0.871),
    # Not from LDMol Table 1, which reports TGM-DLM without the correction-free variant.
    # Transcribed in 29ad1f6 with no source; the one unsourced row in this table.
    ("TGM-DLM w/o corr", 0.828, 0.242, 16.897, 0.874, 0.771, 0.722, 0.89, 0.789),
    # The LDMol row of its own Table 1. Cite v4, the ICML version, which is the row below.
    # The v1 preprint (May 2024) reports a weaker row - validity 0.928, BLEU 0.900,
    # Levenshtein 9.259, MACCS 0.931, RDK 0.860, Morgan 0.820, exact 0.363, FCD 0.25 -
    # and must not be quoted; it is kept here only so that a number found elsewhere in
    # the project can be identified as the stale version rather than re-transcribed.
    # LDMol reports no Text2Mol column, which costs nothing here: Text2Mol is not one of
    # the eight columns printed (see the printer below).
    ("LDMol", 0.926, 0.530, 6.750, 0.973, 0.950, 0.931, 0.20, 0.941),
]

# What this table does to this project's claims. The ranks below are over fourteen rows:
# the eleven of LDMol Table 1, our ten-candidate worst seed, our single-trajectory row and
# the TF-IDF retrieval floor, counting only the rows that report the column. Ground truth
# and the unsourced TGM-DLM w/o corr row are not in the pool.
#
# Our strongest configuration, quoted at the worse of its two seeds in every column
# because that is the rule: BLEU 0.781, exact 0.143, Levenshtein 18.18, MACCS 0.771,
# RDK 0.629, Morgan 0.545, FCD 0.744, validity 0.993. That row is 4th of 14 on validity,
# 6th of 13 on FCD, 7th of 13 on Levenshtein, 8th of 13 on BLEU, 8th of 14 on exact match
# and 9th of 14 on MACCS, RDK and Morgan; of the published rows it is ahead of
# Transformer, GIT-Mol, T5-base and MolT5-base, and of nothing else, and on Morgan it is
# level with T5-base rather than ahead - 0.544985 against 0.545, a difference three orders
# of magnitude below the 0.017 the Morgan column moves between scorers. LDMol is ahead of
# it on seven of the eight columns, and ahead of its better seed on the same seven, but
# only four of those seven have a margin larger than the shift for their column:
# exact match 0.387, Morgan 0.386, RDK 0.321 and MACCS 0.202, against shifts of 0.038,
# 0.017, 0.005 and 0.033. The other three are inside the gap and are not results: BLEU
# 0.145 against a 0.242 shift, Levenshtein 11.43 against 15.9, FCD 0.544 against 0.553.
# Four of eight beyond the calibration gap and three more inside it is the form to quote.
# We lead on validity, 0.993 against 0.941, by 0.052 against a 0.010 shift, and that one
# survives the split adjustment above (0.963 against 0.941). We lead on nothing else. In
# particular "FCD 0.73 is first in the published table" is false: LDMol reports 0.20,
# BioT5+ 0.35, BioT5 0.43 and MolXPT 0.45. Subject, as always, to the calibration note
# above - no row of this table has been through this scorer.
#
# It bounds the retrieval argument rather than killing it. The TF-IDF retrieval floor
# (BLEU 0.786, exact 0.004, Levenshtein 20.57, MACCS 0.785, RDK 0.690, Morgan 0.576,
# FCD 0.481, validity 0.994) does NOT beat TGM-DLM on the fingerprint metrics. TGM-DLM
# reads MACCS 0.854, RDK 0.739 and Morgan 0.688 against the floor's 0.785, 0.690 and
# 0.576, and each of those three margins is larger than its column's calibration shift.
# The floor sits 3rd of 14 on validity, 5th of 13 on FCD, 7th of 13 on BLEU, 8th on MACCS,
# RDK, Morgan and Levenshtein, and 13th of 14 on exact match, where it beats only the
# Transformer's 0.000. On the three fingerprints it beats Transformer, GIT-Mol, T5-base,
# MolT5-base and both of our rows, and nothing else: not TGM-DLM, T5-large, MolT5-large,
# MolXPT, BioT5, BioT5+ or LDMol. Those fingerprint margins all clear their columns'
# shifts, the smallest being Morgan over T5-base at 0.031 against 0.017. On FCD, which
# GIT-Mol does not report, it beats six published rows - Transformer, T5-base, MolT5-base,
# T5-large, MolT5-large and TGM-DLM - and loses to MolXPT, BioT5, BioT5+ and LDMol, which
# is all ten of the published rows that carry the column; the margin over TGM-DLM, 0.29,
# is inside the 0.553 FCD shift. On BLEU it beats the same four, but only the Transformer
# by more than the 0.242 BLEU shift: the margins over GIT-Mol, T5-base and MolT5-base are
# 0.030, 0.021 and 0.017, inside the gap and not results.
#
# So the claim in its honest form: a lookup table that generates nothing outranks four
# published systems on the fingerprint metrics and six on FCD, sits third on validity, and
# collapses to last-but-one on exact match. Ordering the eight columns by how well the
# floor does in them is a ranking of how much each column rewards lookup. Validity is
# saturable, to 1.000 by emitting SELFIES as BioT5 and BioT5+ do; FCD rewards returning
# real training molecules; exact match is the one column where the floor collapses.
# TGM-DLM and LDMol both clear the floor on the fingerprints, and LDMol clears it on seven
# of eight, losing only validity - four of those seven beyond the calibration gap (exact
# match, MACCS, RDK, Morgan) and three inside it (BLEU 0.140, Levenshtein 13.82, FCD
# 0.281). So "ChEBI-20 measures lookup" is not a property of the benchmark and cannot be
# stated as one. It is a true statement about a set of systems, ours among them, and the
# rows that clear the floor are what make it a claim rather than a tautology.
#
# Network evaluations, which invert the test-time framing. LDMol spends about 200 network
# evaluations per caption: 100 DDIM steps, classifier-free guidance, one sample. Our
# 10-candidate row spends 10 x 300 solver steps, about 3,000 backbone passes, plus 10
# verifier scorings. That is about fifteen times the network evaluations for the worse
# row. It is not fifteen times the compute, and must not be written as compute: nothing
# in this repo measures FLOPs or wall-clock for either system, and the passes being
# counted are not the same size. Our backbone is 47.9M parameters; LDMol's DiT is roughly
# 160M, and their frozen ~335M text encoder and ~120M autoregressive latent-to-SMILES
# decoder are not counted on their side at all. In FLOPs the ratio could narrow or
# invert. Quote it as a count of network evaluations with the parameter asymmetry beside
# it, or not at all.

HEADER = ("%-20s%7s%8s%9s%8s%8s%8s%8s%10s"
          % ("model", "BLEU", "Exact", "Leven", "MACCS", "RDK", "Morgan", "FCD", "Validity"))


def cell(value, places: int) -> str:
    """One numeric cell, or "n/a" where a paper does not report that column."""
    return "n/a" if value is None else f"{value:.{places}f}"


def format_row(name: str, m: dict) -> str:
    return ("%-20s%7.3f%8.3f%9.3f%8.3f%8.3f%8.3f%8s%10.3f"
            % (name, m["bleu"], m["exact"], m["levenshtein"], m["maccs"], m["rdk"],
               m["morgan"], cell(m["fcd"], 2), m["validity"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path,
                        help="directory holding reference.txt and one guidance_*.txt")
    parser.add_argument("--name", default=None, help="label for the printed row")
    parser.add_argument("--no-table", action="store_true",
                        help="print only this row, without the published baselines")
    args = parser.parse_args()

    references = (args.dir / "reference.txt").read_text().splitlines()
    files = sorted(p for p in args.dir.glob("guidance_*.txt"))
    if not files:
        raise FileNotFoundError(f"no guidance_*.txt in {args.dir}")

    rows = {}
    for file in files:
        generated = file.read_text().splitlines()
        rows[file.stem] = score(generated, references)

    print(HEADER)
    if not args.no_table:
        # Eight columns of the published nine. Text2Mol is neither stored in
        # REFERENCE_ROWS nor printed, so a published row that does not report it - LDMol
        # does not - needs no placeholder and leaves no gap. The column it is missing is
        # one this table never had. A row missing one of the eight that are printed is a
        # different case: it carries None for that column and cell() prints "n/a" instead
        # of raising. Three cells in the table are like that - GIT-Mol reports no FCD,
        # MolXPT no BLEU and no Levenshtein - and it is also how a row of ours with too
        # few valid generations prints its FCD.
        for row in REFERENCE_ROWS:
            name, bleu, ex, lev, maccs, rdk, morgan, fcd, validity = row
            print("%-20s%7s%8s%9s%8s%8s%8s%8s%10s"
                  % (name, cell(bleu, 3), cell(ex, 3), cell(lev, 3), cell(maccs, 3),
                     cell(rdk, 3), cell(morgan, 3), cell(fcd, 2), cell(validity, 3)))
    for stem, m in rows.items():
        print(format_row(args.name or stem, m))
        print("    %d captions, %d scoreable pairs, %d references without a structure"
              % (m["n"], m["pairs_scored"], m["unparseable_reference"]))

    out = args.dir / "benchmark_metrics.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
