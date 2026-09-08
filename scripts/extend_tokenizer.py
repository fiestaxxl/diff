#!/usr/bin/env python3
"""Extend a trained SMILES tokenizer to cover a second corpus, keeping every existing id.

    python scripts/extend_tokenizer.py \
        --tokenizer data/zinc250k/tokenizers/bpe_384_alpha.json \
        --corpus data/zinc20/corpus/val_sample.txt --hf-dataset liupf/ChEBI-20-MM \
        --out data/shared/tokenizers/zinc20_chebi.json

Why extend rather than retrain: ChEBI-20 needs 219 bracket atoms that ZINC never uses
([Na+], [Fe], [2H], [Si], [Se]), the disconnection dot for salts, and two-digit ring
closures. Retraining reassigns every id, which throws away a pretrained embedding table.
`tokenizers` appends added tokens after the existing vocabulary, so the 386 ids we already
trained on keep their meaning and a checkpoint can be warm-started into the larger table.

Two-digit ring closures are added whole (`%10`, not `%` and `1` and `0`), because the
decoder's legality checker reads one token at a time and a bare `%` means nothing to it.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402

BRACKET = re.compile(r"\[[^\]]+\]")
TWO_DIGIT_RING = re.compile(r"%\d\d")


def iter_corpus(path: Path):
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    for file in files:
        with file.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line


def iter_hf(name: str, column: str = "SMILES"):
    from datasets import load_dataset

    data = load_dataset(name)
    for split in data:
        for row in data[split]:
            value = row.get(column) or row.get(column.lower())
            if value:
                yield value


def missing_pieces(smiles_iter, vocab: set) -> tuple[Counter, Counter, Counter]:
    """Bracket atoms, %NN ring closures and single characters the vocabulary lacks."""
    brackets, rings, chars = Counter(), Counter(), Counter()
    for smiles in smiles_iter:
        for token in BRACKET.findall(smiles):
            if token not in vocab:
                brackets[token] += 1
        rest = BRACKET.sub("", smiles)
        for token in TWO_DIGIT_RING.findall(rest):
            if token not in vocab:
                rings[token] += 1
        for char in TWO_DIGIT_RING.sub("", rest):
            if char not in vocab:
                chars[char] += 1
    return brackets, rings, chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, action="append", default=[],
                        help="a text file or a directory of shards; may be repeated")
    parser.add_argument("--hf-dataset", action="append", default=[],
                        help="a HuggingFace dataset to scan as well; may be repeated")
    parser.add_argument("--column", default="SMILES")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-count", type=int, default=1,
                        help="ignore pieces rarer than this")
    args = parser.parse_args()

    tokenizer = SmilesTokenizer.load(str(args.tokenizer))
    before = tokenizer.get_vocab()
    print(f"loaded {args.tokenizer} with {len(before)} tokens")

    brackets, rings, chars = Counter(), Counter(), Counter()
    vocab = set(before)
    for path in args.corpus:
        b, r, c = missing_pieces(iter_corpus(path), vocab)
        brackets += b
        rings += r
        chars += c
        print(f"  scanned {path}")
    for name in args.hf_dataset:
        b, r, c = missing_pieces(iter_hf(name, args.column), vocab)
        brackets += b
        rings += r
        chars += c
        print(f"  scanned {name}")

    keep = lambda counter: [t for t, n in counter.most_common() if n >= args.min_count]
    new_brackets, new_rings, new_chars = keep(brackets), keep(rings), keep(chars)
    print(f"missing: {len(new_brackets)} bracket atoms, {len(new_rings)} two-digit ring "
          f"closures, {len(new_chars)} single characters {new_chars}")
    if not (new_brackets or new_rings or new_chars):
        print("nothing to add")
        return

    added = tokenizer.add_tokens(new_chars + new_rings + new_brackets)
    after = tokenizer.get_vocab()
    print(f"added {added} tokens, vocabulary {len(before)} -> {len(after)}")

    for token, index in before.items():
        if after.get(token) != index:
            raise SystemExit(f"id of {token!r} moved from {index} to {after.get(token)}")
    print("every existing id is unchanged, so a checkpoint can be warm-started")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
