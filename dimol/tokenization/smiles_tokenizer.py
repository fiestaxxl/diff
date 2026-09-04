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
from typing import Iterable, List, Optional, Sequence, Union

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
        isolate_structure: bool = False,
        protect_elements: Sequence[str] = ("Cl", "Br"),
        restrict_alphabet: bool = True,
    ) -> "SmilesTokenizer":
        """Train a SMILES tokenizer.

        Args:
            smiles_iter: iterable of SMILES strings
            vocab_size: target total vocab size
                (specials + base alphabet + bracket atoms + BPE merges)
            min_frequency: BPE merge minimum frequency
            bracket_min_frequency: minimum count to keep a bracket atom
            show_progress: print progress messages
            isolate_structure: never let a merge cross a parenthesis or a ring digit,
                so those stay single tokens. Costs compression, removes the class of
                tokens that can open a branch or a ring without closing it.
            protect_elements: two-letter element symbols that must stay one token.
                Without this BPE happily cuts "CCCl" into "CCC" + "l", which is a
                chemical error: the chlorine disappears and a bare "l" appears.
            restrict_alphabet: build the initial alphabet from the characters the
                corpus actually contains. The default SMILES_BASE_ALPHABET carries the
                whole latin alphabet, so a corpus like ZINC leaves dozens of unreachable
                entries in the vocabulary that a generative model can still emit,
                producing strings that no chemistry can parse.

        ``smiles_iter`` may be a callable returning a fresh iterator; training then
        makes two passes over it instead of holding the corpus in memory.
        """
        def corpus_pass():
            return smiles_iter() if callable(smiles_iter) else smiles_iter

        # --- pass 1: collect bracket atoms and, optionally, the real alphabet ---
        observed_chars: set[str] = set()
        if restrict_alphabet:
            def counting_pass():
                for smi in corpus_pass():
                    observed_chars.update(smi)
                    yield smi
            bracket_atoms = collect_bracket_atoms(
                counting_pass(), min_frequency=bracket_min_frequency
            )
        else:
            bracket_atoms = collect_bracket_atoms(
                corpus_pass(), min_frequency=bracket_min_frequency
            )
        if show_progress:
            print(
                f"[SmilesTokenizer] Found {len(bracket_atoms)} unique bracket "
                f"atoms (>= {bracket_min_frequency} occurrences)"
            )

        # --- pass 2: build BPE training corpus by stripping brackets ---
        # Replace each [...] with a single space. Whitespace pre-tokenizer
        # then turns each non-bracket run into one BPE "word". With
        # isolate_structure, parentheses and ring digits are blanked out the same
        # way, so no merge can ever contain one.
        bracket_replace = re.compile(r"\[[^\]]+\]")
        structure_replace = re.compile(r"[()0-9]")
        elements = [e for e in (protect_elements or []) if e]
        element_replace = re.compile("|".join(sorted(elements, key=len, reverse=True))) if elements else None

        def bpe_corpus_pass():
            for smi in corpus_pass():
                text = bracket_replace.sub(" ", smi)
                if element_replace is not None:
                    text = element_replace.sub(" ", text)
                if isolate_structure:
                    text = structure_replace.sub(" ", text)
                yield text

        # --- BPE tokenizer ---
        tok = Tokenizer(models.BPE(unk_token=cls.UNK))

        # Whitespace pre-tokenizer for TRAINING:
        # each non-bracket run becomes a single BPE word.
        tok.pre_tokenizer = pre_tokenizers.Whitespace()

        # Compute BPE budget. Final vocab = specials + alphabet + brackets + BPE.
        alphabet = (
            sorted(c for c in observed_chars if not c.isspace())
            if restrict_alphabet
            else SMILES_BASE_ALPHABET
        )
        if show_progress and restrict_alphabet:
            print(f"[SmilesTokenizer] Alphabet from corpus: {len(alphabet)} characters "
                  f"(default list has {len(SMILES_BASE_ALPHABET)})")
        n_specials = len(cls.SPECIAL_TOKENS)
        n_alphabet = len(alphabet)
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
            initial_alphabet=alphabet,
            show_progress=show_progress,
        )
        tok.train_from_iterator(bpe_corpus_pass(), trainer=trainer)

        # --- swap pre-tokenizer for INFERENCE ---
        # Isolate [...] as separate pre-tokens so AddedTokens match them.
        # Non-bracket gaps stay as single contiguous pre-tokens for BPE.
        parts = [r"\[[^\]]+\]"]
        parts += sorted(elements, key=len, reverse=True)
        if isolate_structure:
            parts.append(r"[()0-9]")
        isolate_pattern = "|".join(parts)
        tok.pre_tokenizer = pre_tokenizers.Split(
            pattern=Regex(isolate_pattern),
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
            for b in list(bracket_atoms) + elements
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

    def special_decode(self, ids, skip_special_tokens=True):
        eos_id = self.eos_id
        toks = []
        saw_bos = False
        for i in ids:
            t = self._tok.id_to_token(i)
            if t is None: continue
            if i == self.bos_id:
                if saw_bos:           # second <bos> = stop
                    break
                saw_bos = True
                if skip_special_tokens: continue
            if i == eos_id:           # stop at first <eos>
                if not skip_special_tokens: toks.append(t)
                break
            if skip_special_tokens and t in set(self.SPECIAL_TOKENS):
                continue
            toks.append(t)
        return "".join(toks)

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
        special_decode: bool = False
    ) -> List[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ] if not special_decode else [
            self.special_decode(ids, skip_special_tokens=skip_special_tokens)
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
