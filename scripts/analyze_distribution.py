#!/usr/bin/env python3
"""Check that validity was not bought by generating simpler molecules.

    python scripts/analyze_distribution.py \
        --reference data/zinc250k/corpus/val.txt \
        --limit 5000 \
        runs/r_tl0_ce3/samples_mixed/samples.txt runs/r_base80/samples/samples.txt

For every file: the Frechet ChemNet Distance to the reference set, the share of molecules
too small or too flat to belong in this corpus, Bemis-Murcko scaffold counts, and thirteen
descriptor distributions against the corpus in units of its own standard deviation.

A model that games validity shows up as a large negative shift in heavy atoms, rings and
molecular weight, a low scaffold count, and a high FCD. A model that got better shows up
as a lower FCD with the descriptor shifts near zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.eval.distribution import evaluate_distribution, format_distribution  # noqa: E402


def read_smiles(path: Path, limit: int | None = None) -> list[str]:
    out = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(line)
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--reference", required=True, type=Path,
                        help="one SMILES per line, the corpus split to compare against")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on molecules read from each file, reference included")
    parser.add_argument("--no-fcd", action="store_true",
                        help="skip the ChemNet distance, which is the slow part")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    reference = read_smiles(args.reference, args.limit)
    if not reference:
        raise SystemExit(f"the reference file {args.reference} held no SMILES")
    print(f"reference: {len(reference)} molecules from {args.reference}\n")

    reports = {}
    for path in args.samples:
        if not path.exists():
            raise SystemExit(f"no such sample file: {path}")
        smiles = read_smiles(path, args.limit)
        report = evaluate_distribution(smiles, reference,
                                       with_fcd=not args.no_fcd, device=args.device)
        report["attempts"] = len(smiles)
        reports[path.stem if path.stem != "samples" else path.parent.name] = report
        print(format_distribution(report, name=str(path)))
        print()

    if len(reports) > 1:
        print(f"{'set':<26}{'FCD':>8}{'trivial':>9}{'scaffolds':>11}"
              f"{'heavy atoms':>14}{'rings':>9}{'weight':>9}")
        for name, r in sorted(reports.items(), key=lambda kv: kv[1].get("fcd", float("inf"))):
            d = r["descriptors"]
            fcd_value = r.get("fcd", float("nan"))
            print(f"{name:<26}{fcd_value:>8.2f}{r['trivial']['trivial'] * 100:>8.1f}%"
                  f"{r['scaffolds']['scaffolds']:>11.0f}"
                  f"{d['heavy_atoms']['sample_mean']:>10.1f}"
                  f"{d['heavy_atoms']['shift_sigma']:>+5.1f}s"
                  f"{d['rings']['sample_mean']:>9.2f}"
                  f"{d['mol_weight']['sample_mean']:>9.0f}")
        print("trivial = under 10 heavy atoms or no ring; s = shift in corpus sigmas")

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, default=float))
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
