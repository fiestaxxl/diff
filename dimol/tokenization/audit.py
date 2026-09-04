"""Tokenizer audit: UNK rate, roundtrip failures, compression, vocab usage.

Moved verbatim from the root-level smiles_tokenizer.py (the copy that had it);
the tokenizer itself now lives in dimol/tokenization/smiles_tokenizer.py.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Set

from dimol.tokenization.smiles_tokenizer import SmilesTokenizer


# ------------------------------------------------------------------
# Audit
# ------------------------------------------------------------------
def audit_tokenizer(
    tokenizer: SmilesTokenizer,
    smiles_iter: Iterable[str],
    n_examples: int = 5,
) -> dict:
    n_mol = 0
    n_unk = 0
    mol_with_unk = 0
    orphans: Set[str] = set()
    rt_failures: List[tuple] = []
    token_counter: Counter[str] = Counter()
    total_tokens = 0
    total_chars = 0
    specials = set(tokenizer.SPECIAL_TOKENS)

    for smi in smiles_iter:
        n_mol += 1
        toks = tokenizer.tokenize(smi)
        token_counter.update(toks)
        total_tokens += len(toks)
        total_chars += len(smi)

        had_unk = False
        for t in toks:
            if t == tokenizer.UNK:
                n_unk += 1
                had_unk = True
            if (t.startswith("[") and not t.endswith("]")) or (
                t.endswith("]") and not t.startswith("[") and t not in specials
            ):
                orphans.add(t)
        if had_unk:
            mol_with_unk += 1

        decoded = tokenizer.decode(
            tokenizer.encode(smi, add_special_tokens=False)
        )
        if decoded != smi and len(rt_failures) < n_examples:
            rt_failures.append((smi, decoded))

    return {
        "n_molecules": n_mol,
        "n_unk_tokens": n_unk,
        "molecules_with_unk": mol_with_unk,
        "orphan_bracket_tokens": sorted(orphans),
        "roundtrip_failures": rt_failures,
        "vocab_size": tokenizer.vocab_size,
        "tokens_used": len(token_counter),
        "tokens_unused": tokenizer.vocab_size - len(token_counter),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "chars_per_token": total_chars / max(total_tokens, 1),
        "top_30_tokens": token_counter.most_common(30),
    }
