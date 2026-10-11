#!/usr/bin/env python3
"""Choose one molecule per caption from candidates generated at different lengths.

    python scripts/select_candidates.py --dirs scratch/test/k0 scratch/test/k1 ... \
        --probs data/chebi20/lenc_5e5/test_top5_probs.npy --out scratch/test/k_selected

Exact match needs the length exactly right. The length encoder's argmax is exactly right
19.5% of the time, but the true length is inside its top-5 for 63.2% of captions, so the
information is there and the argmax throws it away.

This generates one candidate per candidate length and picks between them WITHOUT looking
at the reference, which is what makes it a legitimate decoding procedure rather than an
oracle. The selection rule, in order:

1. Drop candidates RDKit cannot parse, and candidates below a minimum heavy-atom count.
   An empty string parses, so the floor matters.
2. Among the survivors prefer the one with the highest score. With length
   probabilities that means the most likely length; with autoregressive log-likelihoods
   it means the molecule the decoder finds most plausible GIVEN THE CAPTION, which is the
   term the length-only rule was missing.
3. If nothing survives, fall back to the top-1 candidate, so the procedure never produces
   fewer molecules than the baseline.

The fallback is what keeps the comparison honest: every caption still yields exactly one
string, so validity is measured over the same denominator as every other row in the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MIN_HEAVY_ATOMS = 3


def usable(smiles: str) -> bool:
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS


def read_generations(directory: Path) -> list[str]:
    files = sorted(directory.glob("guidance_*.txt"))
    if not files:
        raise FileNotFoundError(f"no guidance_*.txt in {directory}")
    if len(files) > 1:
        raise ValueError(f"{directory} holds {len(files)} generation files; expected one")
    return files[0].read_text().splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", required=True, nargs="+", type=Path,
                        help="one directory per candidate rank, in rank order")
    parser.add_argument("--probs", required=True, type=Path,
                        help="(N, K) array to rank candidates by: length probabilities "
                             "from the predictor, or per-token log-likelihoods from the "
                             "autoregressive scorer. Higher is better either way")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    candidates = [read_generations(d) for d in args.dirs]
    reference = (args.dirs[0] / "reference.txt").read_text().splitlines()
    probs = np.load(args.probs)

    n = min(len(reference), min(len(c) for c in candidates), len(probs))
    k = len(candidates)
    probs = probs[:n, :k]

    chosen, stats = [], {"fallback": 0, "rank_used": [0] * k, "n_usable_mean": 0.0}
    for i in range(n):
        options = [(rank, candidates[rank][i]) for rank in range(k)]
        survivors = [(rank, s) for rank, s in options if usable(s)]
        stats["n_usable_mean"] += len(survivors)
        if not survivors:
            stats["fallback"] += 1
            chosen.append(candidates[0][i])
            stats["rank_used"][0] += 1
            continue
        best = max(survivors, key=lambda rs: float(probs[i, rs[0]]))
        chosen.append(best[1])
        stats["rank_used"][best[0]] += 1
    stats["n_usable_mean"] /= max(n, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guidance_0.0_selected.txt").write_text("\n".join(chosen) + "\n")
    (args.out / "reference.txt").write_text("\n".join(reference[:n]) + "\n")
    (args.out / "selection.json").write_text(json.dumps(stats, indent=2))

    print(f"{n} captions, {k} candidates each")
    print(f"  usable candidates per caption: {stats['n_usable_mean']:.2f} of {k}")
    print(f"  captions with no usable candidate (fell back to rank 0): "
          f"{stats['fallback']} ({stats['fallback'] / n * 100:.1f}%)")
    print("  rank chosen: " + "  ".join(
        f"{r}:{c} ({c / n * 100:.0f}%)" for r, c in enumerate(stats["rank_used"])))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
