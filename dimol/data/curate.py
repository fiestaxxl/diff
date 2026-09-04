"""Curation of a raw SMILES corpus: canonicalize, filter, deduplicate.

Everything streams: molecules are pulled from the source one at a time, canonicalized
in a process pool and handed to the caller, so nothing holds the corpus in memory. The
only state that grows with the corpus is the deduplication set, and it stores 64-bit
hashes rather than strings.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Two-character element symbols must be matched before their first letter, otherwise
# "Cl" would be read as "C" followed by "l".
ELEMENT_RE = re.compile(r"Br|Cl|Si|Se|se|as|te|[BCNOSPFIbcnosp]")

KEEP = "keep"
UNPARSABLE = "unparsable"
TOO_LONG = "too_long"
FORBIDDEN = "forbidden_elements"
DUPLICATE = "duplicate"


@dataclass
class CurationStats:
    total: int = 0
    unparsable: int = 0
    too_long: int = 0
    forbidden_elements: int = 0
    duplicates: int = 0
    kept: int = 0
    element_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "unparsable": self.unparsable,
            "too_long": self.too_long,
            "forbidden_elements": self.forbidden_elements,
            "duplicates": self.duplicates,
            "kept": self.kept,
            "element_counts": dict(sorted(self.element_counts.items(), key=lambda kv: -kv[1])),
        }


def canonicalize(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None when the molecule does not parse."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def elements_in(smiles: str) -> List[str]:
    """Element symbols in a SMILES string, aromatic ones folded to their upper case."""
    return [t[0].upper() + t[1:] if len(t) > 1 else t.upper() for t in ELEMENT_RE.findall(smiles)]


def stable_bucket(text: str, seed: int = 42, buckets: int = 1_000_000) -> int:
    """Deterministic bucket of a string, stable across processes and runs.

    Used to route molecules into splits without holding a split in memory: the bucket
    depends only on the molecule and the seed, not on the order of the corpus.
    """
    digest = hashlib.blake2b(text.encode(), digest_size=8, person=str(seed).encode()[:16])
    return int.from_bytes(digest.digest(), "big") % buckets


def fingerprint(text: str) -> int:
    """64-bit hash used for deduplication; 8 bytes per molecule instead of the string."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


# ----------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(canonicalize_smiles: bool, max_smiles_len: Optional[int],
                 allowed_elements: Optional[Sequence[str]]) -> None:
    _WORKER["canonicalize"] = canonicalize_smiles
    _WORKER["max_len"] = max_smiles_len
    _WORKER["allowed"] = set(allowed_elements) if allowed_elements else None


def _curate_one(raw: Optional[str]) -> Tuple[Optional[str], str, Tuple[str, ...]]:
    if not raw:
        return None, UNPARSABLE, ()
    smi = canonicalize(raw) if _WORKER["canonicalize"] else raw
    if smi is None:
        return None, UNPARSABLE, ()
    max_len = _WORKER["max_len"]
    if max_len is not None and len(smi) > max_len:
        return None, TOO_LONG, ()
    elements = tuple(sorted(set(elements_in(smi))))
    allowed = _WORKER["allowed"]
    if allowed is not None and not set(elements) <= allowed:
        return None, FORBIDDEN, ()
    return smi, KEEP, elements


class Curator:
    """Streams a raw SMILES source through curation, in parallel, keeping stats.

    ``nprocs > 1`` spreads canonicalization over a process pool with ``imap``, which
    preserves input order, so the output and the deduplication decisions do not depend
    on how many processes were used.
    """

    def __init__(
        self,
        *,
        canonicalize_smiles: bool = True,
        max_smiles_len: Optional[int] = None,
        allowed_elements: Optional[Sequence[str]] = None,
        deduplicate: bool = True,
        nprocs: Optional[int] = None,
        chunksize: int = 256,
        seen: Optional[Set[int]] = None,
    ):
        self.canonicalize_smiles = canonicalize_smiles
        self.max_smiles_len = max_smiles_len
        self.allowed_elements = allowed_elements
        self.deduplicate = deduplicate
        self.nprocs = nprocs if nprocs is not None else max(1, (os.cpu_count() or 2) // 2)
        self.chunksize = chunksize
        self.seen: Set[int] = seen if seen is not None else set()
        self.stats = CurationStats()

    def _account(self, smi: Optional[str], reason: str, elements: Tuple[str, ...]) -> Optional[str]:
        self.stats.total += 1
        if reason == UNPARSABLE:
            self.stats.unparsable += 1
            return None
        if reason == TOO_LONG:
            self.stats.too_long += 1
            return None
        if reason == FORBIDDEN:
            self.stats.forbidden_elements += 1
            return None
        assert smi is not None
        if self.deduplicate:
            key = fingerprint(smi)
            if key in self.seen:
                self.stats.duplicates += 1
                return None
            self.seen.add(key)
        for element in elements:
            self.stats.element_counts[element] = self.stats.element_counts.get(element, 0) + 1
        self.stats.kept += 1
        return smi

    def stream(self, smiles_iter: Iterable[str]) -> Iterator[str]:
        """Yield curated molecules one by one."""
        init_args = (self.canonicalize_smiles, self.max_smiles_len, self.allowed_elements)
        if self.nprocs <= 1:
            _init_worker(*init_args)
            for raw in smiles_iter:
                kept = self._account(*_curate_one(raw))
                if kept is not None:
                    yield kept
            return

        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(self.nprocs, initializer=_init_worker, initargs=init_args) as pool:
            for result in pool.imap(_curate_one, smiles_iter, chunksize=self.chunksize):
                kept = self._account(*result)
                if kept is not None:
                    yield kept
