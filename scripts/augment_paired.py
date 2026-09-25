#!/usr/bin/env python3
"""Write an augmented copy of a paired split: many SMILES spellings per molecule.

    python scripts/augment_paired.py --corpus data/chebi20/tokenized \
        --out data/chebi20/aug10_mixed --copies 10 --keep-canonical

Why. ChEBI-20's training split is 25,574 molecules, and training the text stage on it
from scratch collapses outright (validity 6.9%, strings of 400 characters against
references of 50), so the corpus is the binding constraint on the second stage exactly as
ZINC-250k was on the first. Meanwhile a quarter of the test set has a training molecule
with an identical Morgan fingerprint, differing only in stereochemistry or tautomer, and
86.9% share a Bemis-Murcko scaffold. So the benchmark rewards recognising that two
spellings denote the same molecule, and the model currently sees each molecule written
exactly one way.

RDKit's random atom ordering gives many valid SMILES for one structure. The molecule is
the diffusion target, not an input, so augmenting means the training set contains several
spellings of each molecule as targets. Two variants are worth separating:

* ``--keep-canonical``: the canonical spelling plus copies-1 random ones. The canonical
  form stays the most frequent target, which matters because BLEU and Levenshtein compare
  raw strings against a canonical reference and are not invariant to spelling.
* without it: copies random spellings and no canonical form at all. Exact match and the
  fingerprint similarities canonicalise both sides and cannot see the difference, so this
  isolates how much of BLEU and Levenshtein is spelling convention rather than chemistry.

The caption states are not copied. They are 5 GB for this split, so the output carries a
``<split>_text_index.npy`` mapping each molecule row to its caption row, which
PairedSmilesTextDataset reads when present.

Only the training split is augmented. Validation and test are the benchmark and are
symlinked through unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from tokenizers import Tokenizer

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def spellings(smiles: str, copies: int, keep_canonical: bool, rng) -> list[str]:
    """Distinct SMILES strings for one molecule, canonical first if asked for."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    if keep_canonical:
        canonical = Chem.MolToSmiles(mol)
        out.append(canonical)
        seen.add(canonical)
    # Ask for more than needed: doRandom repeats itself on small or symmetric molecules,
    # and a molecule with few distinct spellings should contribute few rows rather than
    # the same row many times.
    for _ in range(copies * 4):
        if len(out) >= copies:
            break
        try:
            variant = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            continue
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="the tokenized paired directory to augment")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("data/shared/tokenizers/zinc20_chebi.json"))
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--keep-canonical", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.out.mkdir(parents=True, exist_ok=True)

    tokens = np.load(args.corpus / f"{args.split}_tokens_00000.npy")
    masks = np.load(args.corpus / f"{args.split}_attn_mask_00000.npy")
    pad_id = 0

    originals = []
    for row, mask in zip(tokens, masks):
        ids = [int(t) for t, m in zip(row, mask) if m]
        originals.append(tokenizer.decode(ids, skip_special_tokens=True).replace(" ", ""))
    print(f"{args.split}: {len(originals)} molecules, "
          f"{args.copies} spellings each, keep_canonical={args.keep_canonical}")

    out_tokens, out_masks, out_index = [], [], []
    too_long = unparseable = 0
    for i, smiles in enumerate(originals):
        variants = spellings(smiles, args.copies, args.keep_canonical, rng)
        if not variants:
            unparseable += 1
            continue
        for variant in variants:
            ids = tokenizer.encode(variant).ids
            if len(ids) > args.seq_len:
                too_long += 1
                continue
            row = np.full(args.seq_len, pad_id, dtype=tokens.dtype)
            mask = np.zeros(args.seq_len, dtype=masks.dtype)
            row[: len(ids)] = ids
            mask[: len(ids)] = 1
            out_tokens.append(row)
            out_masks.append(mask)
            out_index.append(i)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1} molecules -> {len(out_tokens)} rows")

    if not out_tokens:
        raise RuntimeError("no rows produced; check the tokenizer and the corpus")

    tokens_out = np.stack(out_tokens)
    masks_out = np.stack(out_masks)
    index_out = np.asarray(out_index, dtype=np.int64)

    np.save(args.out / f"{args.split}_tokens_00000.npy", tokens_out)
    np.save(args.out / f"{args.split}_attn_mask_00000.npy", masks_out)
    np.save(args.out / f"{args.split}_text_index.npy", index_out)

    # The caption states stay where they are; a symlink keeps the loader's layout without
    # copying 5 GB per variant.
    for name in (f"{args.split}_text.npy", f"{args.split}_text_mask.npy"):
        link = args.out / name
        if not link.exists():
            os.symlink((args.corpus / name).resolve(), link)

    # Validation and test are the benchmark: pass them through untouched.
    for split in ("val", "test"):
        for name in (f"{split}_tokens_00000.npy", f"{split}_attn_mask_00000.npy",
                     f"{split}_text.npy", f"{split}_text_mask.npy",
                     f"{split}_captions.txt"):
            source = args.corpus / name
            link = args.out / name
            if source.exists() and not link.exists():
                os.symlink(source.resolve(), link)

    unique_molecules = len(set(index_out.tolist()))
    per_molecule = len(index_out) / max(unique_molecules, 1)
    meta = {
        "source": str(args.corpus),
        "copies_requested": args.copies,
        "keep_canonical": bool(args.keep_canonical),
        "splits": {args.split: {"kept": int(len(index_out)),
                                "unique_molecules": unique_molecules,
                                "spellings_per_molecule": round(per_molecule, 2),
                                "dropped_too_long": too_long,
                                "unparseable": unparseable}},
        "seq_len": args.seq_len,
        "tokenizer": str(args.tokenizer),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(index_out)} rows for {unique_molecules} molecules "
          f"({per_molecule:.2f} spellings each), {too_long} dropped as too long, "
          f"{unparseable} unparseable")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
