#!/usr/bin/env python3
"""What the candidate pool contains, before any selection rule is allowed to look at it.

    python scripts/oracle_ceiling.py --dirs scratch/test/p10_s42/p0 ... --name "10 particles"

Candidate decoding is the only intervention in this project that moved every metric at once, and
every proposal to improve it is a proposal to improve the *selection* rule. None of them can be
priced without knowing what is in the pool to begin with, so this computes the ceiling: for each
caption, the best candidate the pool holds, chosen with the reference in hand.

That number is not a result. It is an upper bound on everything selection can ever buy, and the gap
between it and the deployed rule is the only honest measure of how much room is left. If the oracle
is close to what we already get, better selection is not where the remaining metric gap lives and
the effort belongs in the generator; if it is far above, selection is underexploited.

Three oracles are reported because they disagree, and the disagreement is itself informative:

  exact   pick a candidate that canonicalises to the reference, if the pool holds one
  maccs   pick the candidate with the highest MACCS Tanimoto to the reference
  each    choose independently per metric, which is not a system -- no single string achieves this
          row -- but it bounds every per-metric selection rule at once

The deployed rule is reported beside them on the same captions, so the three numbers are comparable
without re-reading another table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def read_dir(directory: Path) -> list[str]:
    files = sorted(directory.glob("guidance_*.txt"))
    if not files:
        raise FileNotFoundError(f"no guidance_*.txt in {directory}")
    return read_lines(files[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", required=True, nargs="+", type=Path,
                        help="one directory per candidate slot, in pool order")
    parser.add_argument("--selected", type=Path, default=None,
                        help="the directory the deployed selection rule wrote, for comparison")
    parser.add_argument("--name", default="pool")
    parser.add_argument("--out", type=Path, default=None, help="write the per-caption record here")
    args = parser.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys

    RDLogger.DisableLog("rdApp.*")

    pools = [read_dir(d) for d in args.dirs]
    references = read_lines(args.dirs[0] / "reference.txt")
    n = min(len(references), min(len(p) for p in pools))
    k = len(pools)
    selected = read_dir(args.selected)[:n] if args.selected is not None else None
    print(f"{args.name}: {n} captions, {k} candidates each")

    def mol(s: str):
        if not s or not s.strip():
            return None
        m = Chem.MolFromSmiles(s.strip())
        if m is None or m.GetNumHeavyAtoms() < 3:
            return None
        return m

    def fingerprints(m):
        return (MACCSkeys.GenMACCSKeys(m),
                Chem.RDKFingerprint(m),
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

    def canonical(m) -> str:
        return Chem.MolToSmiles(m)

    best_maccs, best_rdk, best_morgan = [], [], []
    oracle_exact, deployed_exact = [], []
    deployed_maccs = []
    pool_has_exact, pool_valid_count = [], []
    rank_of_best = []

    for i in range(n):
        ref = mol(references[i])
        if ref is None:
            continue  # the reference itself does not parse; every row in this project drops these
        ref_canon = canonical(ref)
        ref_fp = fingerprints(ref)

        scores, exacts = [], []
        for j in range(k):
            cand = mol(pools[j][i])
            if cand is None:
                scores.append(None)
                exacts.append(False)
                continue
            cand_fp = fingerprints(cand)
            scores.append(tuple(DataStructs.TanimotoSimilarity(a, b)
                                for a, b in zip(ref_fp, cand_fp)))
            exacts.append(canonical(cand) == ref_canon)

        usable = [j for j in range(k) if scores[j] is not None]
        pool_valid_count.append(len(usable))
        if not usable:
            # no candidate parses: the pool contributes nothing for this caption, and the
            # deployed rule falls back, so the oracle must score zero here too rather than
            # quietly dropping the caption and flattering itself with a smaller denominator
            best_maccs.append(0.0); best_rdk.append(0.0); best_morgan.append(0.0)
            oracle_exact.append(False); pool_has_exact.append(False); rank_of_best.append(-1)
        else:
            best_maccs.append(max(scores[j][0] for j in usable))
            best_rdk.append(max(scores[j][1] for j in usable))
            best_morgan.append(max(scores[j][2] for j in usable))
            hit = any(exacts[j] for j in usable)
            oracle_exact.append(hit)
            pool_has_exact.append(hit)
            rank_of_best.append(max(usable, key=lambda j: scores[j][0]))

        if selected is not None:
            chosen = mol(selected[i])
            if chosen is None:
                deployed_maccs.append(0.0)
                deployed_exact.append(False)
            else:
                deployed_maccs.append(DataStructs.TanimotoSimilarity(
                    ref_fp[0], fingerprints(chosen)[0]))
                deployed_exact.append(canonical(chosen) == ref_canon)

    scored = len(best_maccs)
    row = {
        "name": args.name,
        "captions_with_a_parsing_reference": scored,
        "candidates_per_caption": k,
        "mean_usable_candidates": float(np.mean(pool_valid_count)),
        "oracle_exact": float(np.mean(oracle_exact)),
        "oracle_maccs": float(np.mean(best_maccs)),
        "oracle_rdk": float(np.mean(best_rdk)),
        "oracle_morgan": float(np.mean(best_morgan)),
        "pool_contains_the_answer": float(np.mean(pool_has_exact)),
    }
    if selected is not None:
        row["deployed_exact"] = float(np.mean(deployed_exact))
        row["deployed_maccs"] = float(np.mean(deployed_maccs))

    print(f"  references that parse:        {scored}")
    print(f"  usable candidates per caption {row['mean_usable_candidates']:.2f} of {k}")
    print(f"  the pool CONTAINS the answer  {row['pool_contains_the_answer'] * 100:.1f}% of captions")
    if selected is not None:
        print(f"  the rule FINDS it             {row['deployed_exact'] * 100:.1f}%"
              f"  ({row['deployed_exact'] / max(row['pool_contains_the_answer'], 1e-9) * 100:.0f}%"
              f" of what is there)")
        print(f"  MACCS, deployed               {row['deployed_maccs']:.3f}")
    print(f"  MACCS, oracle per metric      {row['oracle_maccs']:.3f}")
    print(f"  RDK,   oracle per metric      {row['oracle_rdk']:.3f}")
    print(f"  Morgan,oracle per metric      {row['oracle_morgan']:.3f}")

    counts = np.bincount([r for r in rank_of_best if r >= 0], minlength=k)
    print("  rank holding the best MACCS:  "
          + "  ".join(f"{j}:{c}" for j, c in enumerate(counts)))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2) + "\n")
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
