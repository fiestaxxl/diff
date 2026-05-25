"""SMILES tokenizer.

Design:
1. Every [...] bracket atom is preserved as a single, inviolable unit.
2. BPE merges run on everything outside brackets (carbon chains, common motifs,
   multi-char functional groups).
3. Final vocab = specials + base alphabet + all observed bracket atoms + BPE merges.

Why the previous attempts failed:
  - `pre_tokenizers.Split(pattern, behavior="isolated")` makes EVERY match a
    pre-token boundary. When the pattern matched too many things (Cl, Br,
    single C/c, @, @@, etc.), it shattered the input into single-character
    pre-tokens, leaving BPE nothing to merge.

This version's approach:
  - For TRAINING: replace every [...] with a space, then let BPE see the
    remaining text as whitespace-separated "words". This way runs like
    "CCCCCCCC" stay as one BPE word and merges accumulate naturally.
  - For INFERENCE: a Split pre-tokenizer that isolates ONLY [...] (so
    AddedTokens match them; everything else is one contiguous word for BPE).
  - Bracket atoms added as inviolable AddedTokens AFTER training.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    models,
    pre_tokenizers,
    processors,
    trainers,
)


BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")

# Initial alphabet: every single character that may appear in non-bracket SMILES.
# Guarantees BPE has full coverage of any non-bracket character.
SMILES_BASE_ALPHABET = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "()[]=#@%+-./\\:*"
)


def collect_bracket_atoms(
    corpus: Iterable[str], min_frequency: int = 1
) -> List[str]:
    """Return every [...] atom that appears at least `min_frequency` times."""
    counts: Counter[str] = Counter()
    for smi in corpus:
        counts.update(BRACKET_PATTERN.findall(smi))
    return sorted(t for t, c in counts.items() if c >= min_frequency)


class SmilesTokenizer:
    """SMILES tokenizer: brackets as inviolable units, BPE on everything else."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
    PAD, BOS, EOS, UNK, MASK = SPECIAL_TOKENS

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # -------- training --------
    @classmethod
    def train(
        cls,
        smiles_iter: Iterable[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
        bracket_min_frequency: int = 1,
        show_progress: bool = True,
    ) -> "SmilesTokenizer":
        """Train a SMILES tokenizer.

        Args:
            smiles_iter: iterable of SMILES strings
            vocab_size: target total vocab size
                (specials + base alphabet + bracket atoms + BPE merges)
            min_frequency: BPE merge minimum frequency
            bracket_min_frequency: minimum count to keep a bracket atom
            show_progress: print progress messages
        """
        # --- pass 1: materialize corpus + collect bracket atoms ---
        corpus = list(smiles_iter)
        if show_progress:
            print(f"[SmilesTokenizer] Corpus size: {len(corpus)} molecules")

        bracket_atoms = collect_bracket_atoms(
            corpus, min_frequency=bracket_min_frequency
        )
        if show_progress:
            print(
                f"[SmilesTokenizer] Found {len(bracket_atoms)} unique bracket "
                f"atoms (>= {bracket_min_frequency} occurrences)"
            )

        # --- pass 2: build BPE training corpus by stripping brackets ---
        # Replace each [...] with a single space. Whitespace pre-tokenizer
        # then turns each non-bracket run into one BPE "word".
        bracket_replace = re.compile(r"\[[^\]]+\]")
        bpe_corpus = [bracket_replace.sub(" ", smi) for smi in corpus]

        # --- BPE tokenizer ---
        tok = Tokenizer(models.BPE(unk_token=cls.UNK))

        # Whitespace pre-tokenizer for TRAINING:
        # each non-bracket run becomes a single BPE word.
        tok.pre_tokenizer = pre_tokenizers.Whitespace()

        # Compute BPE budget. Final vocab = specials + alphabet + brackets + BPE.
        n_specials = len(cls.SPECIAL_TOKENS)
        n_alphabet = len(SMILES_BASE_ALPHABET)
        n_brackets = len(bracket_atoms)
        bpe_budget = max(vocab_size - n_specials - n_alphabet - n_brackets, 64)
        bpe_target = n_specials + n_alphabet + bpe_budget

        if show_progress:
            print(
                f"[SmilesTokenizer] BPE budget: {bpe_budget} merges "
                f"(target during training: {bpe_target}); "
                f"will add {n_brackets} brackets after."
            )

        trainer = trainers.BpeTrainer(
            vocab_size=bpe_target,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=SMILES_BASE_ALPHABET,
            show_progress=show_progress,
        )
        tok.train_from_iterator(bpe_corpus, trainer=trainer)

        # --- swap pre-tokenizer for INFERENCE ---
        # Isolate [...] as separate pre-tokens so AddedTokens match them.
        # Non-bracket gaps stay as single contiguous pre-tokens for BPE.
        tok.pre_tokenizer = pre_tokenizers.Split(
            pattern=Regex(r"\[[^\]]+\]"),
            behavior="isolated",
        )

        # --- add bracket atoms as inviolable AddedTokens ---
        added = [
            AddedToken(
                b,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
            )
            for b in bracket_atoms
        ]
        tok.add_tokens(added)

        # --- post-processor: <bos> ... <eos> ---
        bos_id = tok.token_to_id(cls.BOS)
        eos_id = tok.token_to_id(cls.EOS)
        tok.post_processor = processors.TemplateProcessing(
            single=f"{cls.BOS} $A {cls.EOS}",
            special_tokens=[(cls.BOS, bos_id), (cls.EOS, eos_id)],
        )

        if show_progress:
            print(f"[SmilesTokenizer] Final vocab size: {tok.get_vocab_size()}")

        return cls(tok)

    # -------- encoding / decoding --------
    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        return self._tok.encode(
            smiles, add_special_tokens=add_special_tokens
        ).ids

    def encode_padded(
        self,
        smiles: str,
        max_length: int = 208,
        add_special_tokens: bool = True,
    ):
        encoded = self.encode(smiles, add_special_tokens=add_special_tokens)
        if max_length <= len(encoded):
            return None, None
        n_pad = max_length - len(encoded)
        attn_mask = [1] * len(encoded) + [0] * n_pad
        encoded = encoded + [self.pad_id] * n_pad
        return encoded, attn_mask

    def encode_batch(
        self, smiles: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        return [
            e.ids
            for e in self._tok.encode_batch(
                smiles, add_special_tokens=add_special_tokens
            )
        ]

    def encode_batch_padded(
        self,
        smiles: List[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ):
        encoded = self.encode_batch(
            smiles, add_special_tokens=add_special_tokens
        )
        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
            target_len = max_length
        else:
            target_len = max(len(ids) for ids in encoded)

        token_ids, attn_masks = [], []
        for ids in encoded:
            n_pad = target_len - len(ids)
            token_ids.append(ids + [self.pad_id] * n_pad)
            attn_masks.append([1] * len(ids) + [0] * n_pad)
        return token_ids, attn_masks

    def decode(
        self, ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        toks = [self._tok.id_to_token(i) for i in ids]
        if skip_special_tokens:
            specials = set(self.SPECIAL_TOKENS)
            toks = [t for t in toks if t is not None and t not in specials]
        return "".join(t for t in toks if t is not None)

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    def tokenize(self, smiles: str) -> List[str]:
        return self._tok.encode(smiles, add_special_tokens=False).tokens

    # -------- properties --------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(self.PAD)

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(self.BOS)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(self.EOS)

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id(self.UNK)

    @property
    def mask_id(self) -> int:
        return self._tok.token_to_id(self.MASK)

    def get_vocab(self) -> dict:
        return self._tok.get_vocab()

    # -------- save / load --------
    def save(self, path: Union[str, Path]) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SmilesTokenizer":
        return cls(Tokenizer.from_file(str(path)))


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


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_smiles = [
        "CC(=O)O[C@@H]1C[C@H]2C(C)(C)C(=O)C=C[C@]2(C)[C@H]2CC[C@]3(C)C(=CC[C@H]3c3ccoc3)[C@@]21C",
        "[125Te]",
        "CN(N=O)C(=N)O",
        "c1ccc2c(-c3nccs3)c[nH]c2c1",
        "NCC([O-])=NC1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
        "CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-]",
        "CCCCCCCCCCCCCCCC(=O)O",
        "CCCCCCCCCCCCCCCCCCCC(=O)O",
        "c1ccccc1",
        "c1ccc(C(=O)O)cc1",
    ] * 50

    tok = SmilesTokenizer.train(
        sample_smiles,
        vocab_size=300,
        min_frequency=2,
        bracket_min_frequency=1,
    )

    print(f"\nVocab size: {tok.vocab_size}")
    print("\nTokenization examples:")
    for s in sample_smiles[:6]:
        print(f"  {s}\n    -> {tok.tokenize(s)}")

    print("\nAudit:")
    audit = audit_tokenizer(tok, sample_smiles[:10])
    for k, v in audit.items():
        if k != "top_30_tokens":
            print(f"  {k}: {v}")
    print("  Top 30 tokens:")
    for t, c in audit["top_30_tokens"]:
        print(f"    {t!r:25s} {c}")