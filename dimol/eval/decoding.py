"""Decoding the diffusion latents into SMILES.

The model produces one distribution per canvas position at once, and the plain decoder
takes the argmax of each independently. That is where most of the invalid molecules come
from: measured on ZINC-250k, 88% of the failures are unbalanced parentheses or an odd
number of ring-closure digits, i.e. violations of a grammar that is entirely known in
advance and has nothing to do with chemistry knowledge.

``GrammarConstrainedDecoder`` walks the positions left to right and repairs the string as
it goes: a token that would drive the parenthesis depth below zero is not written, and a
string that ends with something open is closed or trimmed.

What the repair must never do is invent chemistry. An earlier version, when the model
asked for an impossible ")", fell back to the next most likely token, and the next most
likely token is often an atom: strings came out with a phosphorus atom inserted where a
parenthesis had been, which rdkit accepts and a chemist would not. ``substitution="skip"``,
the default, drops the impossible token instead and keeps every other choice the model
made. ``substitution="next_best"`` is the old behaviour, kept only so the difference can
be measured.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch

BRACKET_RE = re.compile(r"\[[^\]]+\]")


def _token_grammar(token: str) -> tuple[int, int, np.ndarray]:
    """(paren delta, minimum running paren balance, ring-digit parity) of one token.

    Digits inside bracket atoms are isotopes, not ring closures, so brackets are stripped
    first. The running minimum matters because a token such as ")C(" has delta zero but
    dips below the current depth on the way.
    """
    text = BRACKET_RE.sub("", token)
    depth = 0
    lowest = 0
    parity = np.zeros(10, dtype=np.int8)
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            lowest = min(lowest, depth)
        elif ch.isdigit():
            parity[int(ch)] ^= 1
    return depth, lowest, parity


class GrammarConstrainedDecoder:
    """Greedy left-to-right decoding over a feasible token set."""

    def __init__(self, tokenizer, canvas: int, top_k: int = 16, repair: str = "close",
                 substitution: str = "skip"):
        vocab = tokenizer.get_vocab()
        self.canvas = int(canvas)
        self.top_k = int(top_k)
        if repair not in ("trim", "close", "mixed"):
            raise ValueError(f"repair={repair!r}; expected 'trim', 'close' or 'mixed'")
        self.repair = repair
        if substitution not in ("skip", "next_best"):
            raise ValueError(
                f"substitution={substitution!r}; expected 'skip' or 'next_best'"
            )
        self.substitution = substitution
        self.vocab_size = max(vocab.values()) + 1
        self.id_to_token: Dict[int, str] = {i: t for t, i in vocab.items()}

        specials = set(getattr(tokenizer, "SPECIAL_TOKENS", []))
        self.delta = np.zeros(self.vocab_size, dtype=np.int16)
        self.lowest = np.zeros(self.vocab_size, dtype=np.int16)
        self.parity = np.zeros((self.vocab_size, 10), dtype=np.int8)
        self.is_special = np.zeros(self.vocab_size, dtype=bool)
        for token, idx in vocab.items():
            if token in specials:
                self.is_special[idx] = True
                continue
            d, low, par = _token_grammar(token)
            self.delta[idx], self.lowest[idx], self.parity[idx] = d, low, par

        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id
        self.bos_id = tokenizer.bos_id
        self.unk_id = tokenizer.unk_id
        # <eos> and <pad> end the molecule; <bos>, <unk> and <mask> are simply skipped,
        # because the model is trained on "<bos> tokens <eos> <pad>..." and puts <bos>
        # on the first canvas position
        self.is_stop = np.zeros(self.vocab_size, dtype=bool)
        for idx in (self.eos_id, self.pad_id):
            if idx is not None and idx < self.vocab_size:
                self.is_stop[idx] = True

    def decode(self, logits: torch.Tensor, chunk: int = 2000) -> List[str]:
        """logits: (B, L, V) -> list of SMILES strings."""
        out: List[str] = []
        for start in range(0, logits.shape[0], chunk):
            out.extend(self._decode_chunk(logits[start : start + chunk].float().cpu().numpy()))
        return out

    def _decode_chunk(self, logits: np.ndarray) -> List[str]:
        """Model-driven decoding with grammar repair.

        The model decides the content and the length: its argmax is taken as the
        intended token per position, and the intended end of the molecule is the first
        position where it wants a special token. Inside that prefix the grammar only
        repairs: a token that would push the parenthesis depth below zero is dropped, and
        if the string still ends with something open it is closed or trimmed.
        """
        batch, length, vocab = logits.shape
        vocab = min(vocab, self.vocab_size)
        logits = logits[:, :, :vocab]
        order = np.argsort(-logits, axis=-1)[:, :, : self.top_k]  # candidates per position

        results: List[str] = []
        for i in range(batch):
            depth = 0
            parity = np.zeros(10, dtype=np.int8)
            pieces: List[str] = []
            balanced_at = 0  # longest prefix that leaves nothing open at all
            rings_clean_at = 0  # longest prefix with every ring digit paired

            for pos in range(length):
                intended = int(order[i, pos, 0])
                if self.is_stop[intended]:
                    break  # the model wants to end the molecule here
                if self.is_special[intended]:
                    continue  # <bos> and friends carry no content
                if depth + int(self.lowest[intended]) >= 0:
                    chosen = intended  # the model's choice is legal, take it
                elif self.substitution == "skip":
                    continue  # illegal here: drop it and keep the rest of the string
                else:
                    chosen = None
                    for cand in order[i, pos]:
                        cand = int(cand)
                        if self.is_special[cand]:
                            continue
                        if depth + int(self.lowest[cand]) < 0:
                            continue
                        chosen = cand
                        break
                    if chosen is None:
                        continue
                pieces.append(self.id_to_token[chosen])
                depth += int(self.delta[chosen])
                parity ^= self.parity[chosen]
                if depth == 0 and parity.sum() == 0:
                    balanced_at = len(pieces)
                if parity.sum() == 0:
                    rings_clean_at = len(pieces)

            if depth != 0 or parity.sum() != 0:
                if self.repair == "trim":
                    # cut back to the last point where nothing was left open
                    pieces = pieces[:balanced_at]
                elif self.repair == "mixed":
                    # keep the content up to the last paired ring, then close branches:
                    # appending a stray ring digit tends to break the valence instead
                    pieces = pieces[:rings_clean_at]
                    depth_left = 0
                    for token in pieces:
                        depth_left += _token_grammar(token)[0]
                    pieces.extend([")"] * max(depth_left, 0))
                else:
                    # close what is open instead of throwing content away: ring digits
                    # first, then the branches around them
                    for digit in np.nonzero(parity)[0]:
                        pieces.append(str(int(digit)))
                    pieces.extend([")"] * max(depth, 0))
            results.append("".join(pieces))
        return results


def build_decoder(tokenizer, canvas: int, mode: str = "argmax") -> Optional[GrammarConstrainedDecoder]:
    """Modes are `argmax`, `grammar_close|trim|mixed`, plus a `_subst` suffix.

    The suffix selects the old substituting behaviour, e.g. `grammar_close_subst`, which
    exists only to measure how much of the validity it was buying by inserting atoms the
    model never asked for.
    """
    if mode == "argmax":
        return None
    substitution = "skip"
    if mode.endswith("_subst"):
        substitution, mode = "next_best", mode[: -len("_subst")]
    repairs = {"grammar": "close", "grammar_close": "close",
               "grammar_trim": "trim", "grammar_mixed": "mixed"}
    if mode not in repairs:
        raise ValueError(
            f"generate.decode={mode!r}; expected 'argmax', 'grammar_close', "
            "'grammar_trim', 'grammar_mixed', optionally with a '_subst' suffix"
        )
    return GrammarConstrainedDecoder(tokenizer, canvas, repair=repairs[mode],
                                     substitution=substitution)
