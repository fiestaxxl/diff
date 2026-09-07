"""Incremental SMILES legality: what may be written next, given what is written already.

The bracket-and-ring-parity check that ``decoding.py`` started with accepts strings the
parser still rejects, and measurement says that is where most of the remaining failures
live. On the reference model, after bracket and ring repair, 5125 of 10000 attempts still
fail to parse, and 4438 of those are two rules:

* ``duplicated ring closure ... bonds atom to itself``, e.g. ``CC22``: the same ring digit
  opened and closed on one atom. Parity is even, so the old check saw nothing wrong;
* ``ring closure duplicates bond between atom N and atom M``: a ring closure between two
  atoms that a chain bond already joined.

Both need to know which atom is current and which atoms are already bonded, which means
tracking the molecule as it is written rather than counting characters. That is what this
module does: a token is turned into a short event list once, and the state replays those
events, refusing any that would make the string unparseable.

The state deliberately stops at connectivity. Valences and aromaticity are not checked
here: they need element and bond-order bookkeeping that only pays off once the connectivity
errors are gone, and they are a separate measurement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

ATOM = "atom"
RING = "ring"
OPEN = "open"
CLOSE = "close"
BOND = "bond"
DOT = "dot"

BRACKET_RE = re.compile(r"\[[^\]]+\]")
TWO_LETTER = ("Cl", "Br")
ORGANIC = set("BCNOSPFIbcnosp*")
BOND_CHARS = set("=#-:/\\$")


@dataclass(frozen=True)
class Event:
    kind: str
    digit: int = -1


@lru_cache(maxsize=8192)
def token_events(token: str) -> Tuple[Event, ...]:
    """One token of the vocabulary as a sequence of structural events.

    Multi-character tokens are normal here: a BPE vocabulary over SMILES contains
    fragments like ``CC(`` or ``c1cc``, and every character of them counts.
    """
    events: List[Event] = []
    i = 0
    while i < len(token):
        ch = token[i]
        if ch == "[":
            end = token.find("]", i)
            if end == -1:  # a truncated bracket cannot be placed at all
                return (Event("invalid"),)
            events.append(Event(ATOM))
            i = end + 1
            continue
        if token[i : i + 2] in TWO_LETTER:
            events.append(Event(ATOM))
            i += 2
            continue
        if ch == "(":
            events.append(Event(OPEN))
        elif ch == ")":
            events.append(Event(CLOSE))
        elif ch == "%":  # %NN is a two-digit ring number
            digits = token[i + 1 : i + 3]
            if len(digits) == 2 and digits.isdigit():
                events.append(Event(RING, int(digits)))
                i += 3
                continue
            return (Event("invalid"),)
        elif ch.isdigit():
            events.append(Event(RING, int(ch)))
        elif ch in BOND_CHARS:
            events.append(Event(BOND))
        elif ch == ".":
            events.append(Event(DOT))
        elif ch in ORGANIC or ch.isalpha():
            events.append(Event(ATOM))
        else:
            return (Event("invalid"),)
        i += 1
    return tuple(events)


@dataclass
class SmilesState:
    """Connectivity of the string written so far, and what it allows next."""

    atoms: int = 0
    current: int = -1  # index of the atom a new bond attaches to
    branch_stack: List[int] = field(default_factory=list)
    open_rings: Dict[int, int] = field(default_factory=dict)  # digit -> atom index
    bonds: Set[frozenset] = field(default_factory=set)
    last: str = "start"

    def copy(self) -> "SmilesState":
        return SmilesState(
            atoms=self.atoms, current=self.current,
            branch_stack=list(self.branch_stack),
            open_rings=dict(self.open_rings),
            bonds=set(self.bonds), last=self.last,
        )

    # ------------------------------------------------------------------
    def _apply(self, event: Event) -> bool:
        """Apply one event, returning False if it makes the string unparseable."""
        kind = event.kind
        if kind == "invalid":
            return False

        if kind == ATOM:
            index = self.atoms
            self.atoms += 1
            if self.current >= 0 and self.last != DOT:
                self.bonds.add(frozenset((self.current, index)))
            self.current = index
            self.last = ATOM
            return True

        if kind == OPEN:
            if self.current < 0 or self.last in (OPEN, BOND):
                return False  # a branch before any atom, or right after another opener
            self.branch_stack.append(self.current)
            self.last = OPEN
            return True

        if kind == CLOSE:
            if not self.branch_stack:
                return False  # closes a branch that was never opened
            if self.last in (OPEN, BOND):
                return False  # an empty branch, or one ending on a bond symbol
            self.current = self.branch_stack.pop()
            self.last = CLOSE
            return True

        if kind == RING:
            if self.current < 0 or self.last == OPEN:
                return False  # a ring digit before any atom or as a whole branch
            digit = event.digit
            if digit in self.open_rings:
                opener = self.open_rings[digit]
                if opener == self.current:
                    return False  # would bond an atom to itself
                pair = frozenset((opener, self.current))
                if pair in self.bonds:
                    return False  # that bond already exists
                self.bonds.add(pair)
                del self.open_rings[digit]
            else:
                # the same digit may legitimately appear twice on one atom, as in
                # c1ccccc11, where the first closes a ring and the second opens the next
                self.open_rings[digit] = self.current
            self.last = RING
            return True

        if kind == BOND:
            # a bond right after a branch opener is normal, as in C(=O)O
            if self.last in (BOND, "start", DOT):
                return False  # two bond symbols in a row, or one with nothing to bond
            self.last = BOND
            return True

        if kind == DOT:
            if self.last in (BOND, OPEN, "start"):
                return False
            self.current = -1  # a new disconnected fragment starts here
            self.last = DOT
            return True

        return False

    def can_place(self, events: Tuple[Event, ...]) -> bool:
        probe = self.copy()
        return all(probe._apply(event) for event in events)

    def place(self, events: Tuple[Event, ...]) -> bool:
        """Apply the events if all of them are legal; otherwise change nothing."""
        probe = self.copy()
        for event in events:
            if not probe._apply(event):
                return False
        self.__dict__.update(probe.__dict__)
        return True

    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        return len(self.branch_stack)

    @property
    def complete(self) -> bool:
        """Nothing left open and nothing dangling: the string can end here."""
        return (not self.branch_stack and not self.open_rings
                and self.last not in (BOND, "start", DOT, OPEN))

    def closing_tokens(self) -> Optional[List[str]]:
        """The shortest legal suffix that finishes the string, or None if there is none.

        Ring digits are closed first, then branches, which is the order the parser needs.
        A ring whose closure would duplicate a bond or point at its own atom cannot be
        closed at all, and the caller has to trim instead.
        """
        if self.last in (BOND, DOT, OPEN):
            return None
        probe = self.copy()
        suffix: List[str] = []
        for digit in sorted(probe.open_rings):
            token = str(digit) if digit < 10 else f"%{digit:02d}"
            if not probe.place(token_events(token)):
                return None
            suffix.append(token)
        while probe.branch_stack:
            if not probe.place(token_events(")")):
                return None
            suffix.append(")")
        return suffix if probe.complete else None
