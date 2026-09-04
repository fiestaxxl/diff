"""Atom-level SMILES tokenizer: the chemically safe baseline.

Every unit of the standard SMILES grammar becomes exactly one token: a bracket atom
`[nH]`, a two-letter element `Cl`, a single-letter element, a bond symbol, a
parenthesis, a ring digit. No merges, so no token can ever split an atom or hide a
ring digit inside a longer piece. The price is the longest sequences of all
candidates, which is exactly the trade-off the audit measures.

The result is a `SmilesTokenizer`, saved in the same tokenizers JSON format, so
everything downstream (encoding, padding, decoding, the audit) is unchanged.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

# The usual SMILES atom-level pattern: two-letter symbols before their first letter,
# bracket atoms as one unit, %NN ring closures before bare digits.
ATOM_PATTERN = (
    r"\[[^\]]+\]|Br|Cl|Si|Se|se|si|as|te|B|C|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"
)


def train_atomwise(
    smiles_iter: Iterable[str] | Callable[[], Iterable[str]],
    min_frequency: int = 1,
    show_progress: bool = True,
) -> SmilesTokenizer:
    """Build the vocabulary from the atom-level units observed in the corpus."""
    from tokenizers import pre_tokenizers as pt

    splitter = pt.Split(pattern=Regex(ATOM_PATTERN), behavior="isolated")
    counts: Counter[str] = Counter()
    n_molecules = 0
    stream = smiles_iter() if callable(smiles_iter) else smiles_iter
    for smi in stream:
        n_molecules += 1
        counts.update(piece for piece, _ in splitter.pre_tokenize_str(smi) if piece)

    kept = sorted(tok for tok, n in counts.items() if n >= min_frequency)
    if show_progress:
        print(f"[atomwise] {n_molecules} molecules, {len(counts)} distinct units, "
              f"{len(kept)} kept at min_frequency={min_frequency}")

    vocab = {tok: i for i, tok in enumerate(SmilesTokenizer.SPECIAL_TOKENS)}
    for tok in kept:
        vocab.setdefault(tok, len(vocab))

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=SmilesTokenizer.UNK))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(ATOM_PATTERN), behavior="isolated")
    tok.post_processor = processors.TemplateProcessing(
        single=f"{SmilesTokenizer.BOS} $A {SmilesTokenizer.EOS}",
        special_tokens=[
            (SmilesTokenizer.BOS, vocab[SmilesTokenizer.BOS]),
            (SmilesTokenizer.EOS, vocab[SmilesTokenizer.EOS]),
        ],
    )
    if show_progress:
        print(f"[atomwise] final vocab size: {tok.get_vocab_size()}")
    return SmilesTokenizer(tok)
