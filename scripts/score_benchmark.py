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
* FCD is computed between the valid generations and their own references.
* Validity is the share of outputs RDKit accepts **and** that carry at least three heavy
  atoms. An empty string parses in RDKit and would otherwise count as a valid molecule.

Text2Mol is the one column this cannot produce: it needs the external retrieval model
from Edwards et al., which is not on either machine. The column is emitted as None so
the shape of the table is honest about it.
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
    """FCD between the valid generations and the references they were asked for."""
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
            # ChEBI-20 carries a handful of captions with no structure at all (isotope
            # entries). They are unanswerable, so they are counted and reported, never
            # silently dropped.
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


REFERENCE_ROWS = [
    ("Ground truth", 1.000, 1.000, 0.000, 1.000, 1.000, 1.000, 0.00, 1.000),
    ("Transformer", 0.499, 0.000, 57.660, 0.480, 0.320, 0.217, 11.32, 0.906),
    ("T5-Base", 0.762, 0.069, 24.950, 0.731, 0.605, 0.545, 2.48, 0.660),
    ("MolT5-Base", 0.769, 0.081, 24.458, 0.721, 0.588, 0.529, 2.18, 0.772),
    ("TGM-DLM w/o corr", 0.828, 0.242, 16.897, 0.874, 0.771, 0.722, 0.89, 0.789),
    ("TGM-DLM", 0.826, 0.242, 17.003, 0.854, 0.739, 0.688, 0.77, 0.871),
]

HEADER = ("%-20s%7s%8s%9s%8s%8s%8s%8s%10s"
          % ("model", "BLEU", "Exact", "Leven", "MACCS", "RDK", "Morgan", "FCD", "Validity"))


def format_row(name: str, m: dict) -> str:
    fcd = f"{m['fcd']:.2f}" if m["fcd"] is not None else "n/a"
    return ("%-20s%7.3f%8.3f%9.3f%8.3f%8.3f%8.3f%8s%10.3f"
            % (name, m["bleu"], m["exact"], m["levenshtein"], m["maccs"], m["rdk"],
               m["morgan"], fcd, m["validity"]))


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
        for row in REFERENCE_ROWS:
            name, bleu, ex, lev, maccs, rdk, morgan, fcd, validity = row
            print("%-20s%7.3f%8.3f%9.3f%8.3f%8.3f%8.3f%8.2f%10.3f"
                  % (name, bleu, ex, lev, maccs, rdk, morgan, fcd, validity))
    for stem, m in rows.items():
        print(format_row(args.name or stem, m))
        print("    %d captions, %d scoreable pairs, %d references without a structure"
              % (m["n"], m["pairs_scored"], m["unparseable_reference"]))

    out = args.dir / "benchmark_metrics.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
