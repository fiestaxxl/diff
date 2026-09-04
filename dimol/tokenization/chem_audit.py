"""Chemistry-aware evaluation of a SMILES tokenizer.

A SMILES tokenizer is not judged by compression alone. Two properties matter more:

* it must never split a chemical unit. Cutting ``Cl`` into ``C`` + ``l`` or ``[nH]``
  into ``[n`` + ``H]`` turns one atom into two symbols that mean something else;
* it should not make the structural bookkeeping harder. Parentheses and ring-closure
  digits are where an autoregressive or diffusion model usually fails, so the audit
  measures how many vocabulary entries can open a branch or a ring without closing it,
  and how far apart the matching symbols end up in token space.

The reference segmentation is the standard atom-level SMILES pattern; the tokenizer
under test is compared against it character by character.
"""
from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ATOM_PATTERN = (
    r"\[[^\]]+\]|Br|Cl|Si|Se|se|si|as|te|B|C|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9]"
)
ATOM_RE = re.compile(ATOM_PATTERN)
BRACKET_RE = re.compile(r"\[[^\]]+\]")
STRUCTURAL_CHARS = set("()0123456789=#-+/\\%:.~$")
RING_LABEL_RE = re.compile(r"^%?\d+$")


def atom_spans(smiles: str) -> Tuple[List[Tuple[int, int, str]], int]:
    """Reference segmentation: [(start, end, text)] plus the number of uncovered chars."""
    spans = [(m.start(), m.end(), m.group()) for m in ATOM_RE.finditer(smiles)]
    covered = sum(e - s for s, e, _ in spans)
    return spans, len(smiles) - covered


def token_bounds(tokens: Sequence[str]) -> List[int]:
    """Cumulative end offset of every token; tokens tile the string when lossless."""
    out, pos = [], 0
    for tok in tokens:
        pos += len(tok)
        out.append(pos)
    return out


def _token_index(bounds: List[int], char_pos: int) -> int:
    return bisect_right(bounds, char_pos)


def strip_brackets(token: str) -> str:
    """Token text without bracket atoms, so isotopes and charges are not read as rings."""
    return BRACKET_RE.sub("", token)


def is_balanced(token: str) -> bool:
    depth = 0
    for ch in strip_brackets(token):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def has_ring_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in strip_brackets(token))


def is_whole_atoms(token: str) -> bool:
    """True when the token is a concatenation of complete atom-level units."""
    spans, uncovered = atom_spans(token)
    return uncovered == 0 and bool(spans)


def ring_digit_counts(token: str) -> Counter:
    """Ring-closure digits in a token, ignoring digits inside bracket atoms."""
    return Counter(ch for ch in strip_brackets(token) if ch.isdigit())


def is_dangling(token: str) -> bool:
    """Token that opens a branch or a ring it does not close.

    This is the risky class: emitting it commits the model to closing something
    later. A token like "N3CCCC3" is the opposite - it carries a whole ring and can
    never be half-written, so it is counted as self-closing instead.
    """
    if not is_balanced(token):
        return True
    return any(n % 2 for n in ring_digit_counts(token).values())


def is_self_closing(token: str) -> bool:
    """Token that carries structure and closes all of it inside itself."""
    stripped = strip_brackets(token)
    has_structure = any(ch in "()" or ch.isdigit() for ch in stripped)
    return has_structure and not is_dangling(token)


def is_mixed_structure(token: str) -> bool:
    """Token that carries chemistry AND an unclosed parenthesis or a ring digit.

    A bare "(" is not a risk: it is the atomic unit and the model has to close it
    either way. The risky entries are the ones that force the model to commit to a
    branch or a ring while it emits atoms, e.g. "Cc1ccc(" or "N3CCCC3".
    """
    stripped = strip_brackets(token)
    has_atom = any(ch.isalpha() for ch in stripped) or "[" in token
    return has_atom and (not is_balanced(token) or has_ring_digit(token))


def is_structural(token: str) -> bool:
    stripped = strip_brackets(token)
    return bool(stripped) and all(ch in STRUCTURAL_CHARS for ch in stripped)


def audit(
    tokenizer,
    smiles_iter: Iterable[str],
    *,
    name: str = "",
    max_molecules: Optional[int] = None,
) -> Dict[str, object]:
    """Run the full audit over a corpus and return a flat-ish report."""
    import numpy as np

    counter: Counter[str] = Counter()
    n_mol = n_unk = n_roundtrip_fail = 0
    n_chars = n_tokens = 0
    uncovered_total = 0

    atoms_total = atoms_split = 0
    per_kind_total: Counter[str] = Counter()
    per_kind_split: Counter[str] = Counter()

    tokens_per_mol: List[int] = []
    ring_gaps: List[int] = []
    ring_unmatched = 0
    paren_gaps: List[int] = []
    paren_depth: List[int] = []
    structural_occurrences = 0

    unk_token = getattr(tokenizer, "UNK", "<unk>")
    examples: List[Tuple[str, List[str]]] = []

    for smi in smiles_iter:
        if max_molecules is not None and n_mol >= max_molecules:
            break
        n_mol += 1
        tokens = tokenizer.tokenize(smi)
        counter.update(tokens)
        n_tokens += len(tokens)
        n_chars += len(smi)
        tokens_per_mol.append(len(tokens))
        n_unk += sum(1 for t in tokens if t == unk_token)

        if "".join(tokens) != smi:
            n_roundtrip_fail += 1
            continue
        if len(examples) < 3:
            examples.append((smi, tokens))

        bounds = token_bounds(tokens)
        inner = set(bounds[:-1])  # boundaries strictly inside the string
        spans, uncovered = atom_spans(smi)
        uncovered_total += uncovered

        ring_open: Dict[str, List[int]] = {}
        paren_stack: List[int] = []
        for start, end, text in spans:
            kind = (
                "bracket" if text.startswith("[")
                else text if text in ("Cl", "Br")
                else "ring_digit" if RING_LABEL_RE.match(text)
                else "paren" if text in "()"
                else "other"
            )
            atoms_total += 1
            per_kind_total[kind] += 1
            if any(start < b < end for b in inner):
                atoms_split += 1
                per_kind_split[kind] += 1

            tok_idx = _token_index(bounds, start)
            if kind == "ring_digit":
                label = text.lstrip("%")
                stack = ring_open.setdefault(label, [])
                if stack:
                    ring_gaps.append(tok_idx - stack.pop())
                else:
                    stack.append(tok_idx)
            elif text == "(":
                paren_stack.append(tok_idx)
                paren_depth.append(len(paren_stack))
            elif text == ")":
                if paren_stack:
                    paren_gaps.append(tok_idx - paren_stack.pop())

        ring_unmatched += sum(len(v) for v in ring_open.values())
        structural_occurrences += sum(1 for t in tokens if is_structural(t))

    vocab = {t: i for t, i in tokenizer.get_vocab().items()}
    specials = set(getattr(tokenizer, "SPECIAL_TOKENS", []))
    content = [t for t in vocab if t not in specials]
    used = [t for t in content if counter.get(t, 0) > 0]

    unbalanced = [t for t in content if not is_balanced(t)]
    mixed = [t for t in content if is_mixed_structure(t)]
    dangling = [t for t in content if is_dangling(t)]
    self_closing = [t for t in content if is_self_closing(t)]
    with_digit = [t for t in content if has_ring_digit(t)]
    partial_atoms = [t for t in content if not is_whole_atoms(t)]

    def occurrence_share(tokens: Sequence[str]) -> float:
        return sum(counter.get(t, 0) for t in tokens) / max(n_tokens, 1)

    lengths = np.asarray(tokens_per_mol, dtype=np.int32)
    ordered = counter.most_common()
    cum, cover50 = 0, 0
    for i, (_, c) in enumerate(ordered, 1):
        cum += c
        if cum >= 0.5 * n_tokens:
            cover50 = i
            break
    probs = np.asarray([c for _, c in ordered], dtype=np.float64) / max(n_tokens, 1)
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0

    def pct(values: List[int], q: float) -> float:
        return float(np.percentile(values, q)) if values else float("nan")

    return {
        "name": name,
        "vocab_size": len(vocab),
        "molecules": n_mol,
        # --- hard requirements ---
        "roundtrip_failures": n_roundtrip_fail,
        "unk_tokens": n_unk,
        "uncovered_chars_in_reference": uncovered_total,
        # --- chemical integrity ---
        "atoms_split_pct": 100 * atoms_split / max(atoms_total, 1),
        "cl_br_split": per_kind_split["Cl"] + per_kind_split["Br"],
        "cl_br_total": per_kind_total["Cl"] + per_kind_total["Br"],
        "bracket_split": per_kind_split["bracket"],
        "bracket_total": per_kind_total["bracket"],
        "vocab_partial_atom_pct": 100 * len(partial_atoms) / max(len(content), 1),
        "occurrence_partial_atom_pct": 100 * occurrence_share(partial_atoms),
        # --- structural risk ---
        "vocab_unbalanced_pct": 100 * len(unbalanced) / max(len(content), 1),
        "occurrence_unbalanced_pct": 100 * occurrence_share(unbalanced),
        "vocab_mixed_pct": 100 * len(mixed) / max(len(content), 1),
        "occurrence_mixed_pct": 100 * occurrence_share(mixed),
        "vocab_dangling_pct": 100 * len(dangling) / max(len(content), 1),
        "occurrence_dangling_pct": 100 * occurrence_share(dangling),
        "occurrence_self_closing_pct": 100 * occurrence_share(self_closing),
        "ring_closed_in_token_pct": 100 * sum(1 for g in ring_gaps if g == 0) / max(len(ring_gaps), 1),
        "paren_closed_in_token_pct": 100 * sum(1 for g in paren_gaps if g == 0) / max(len(paren_gaps), 1),
        "vocab_with_ring_digit_pct": 100 * len(with_digit) / max(len(content), 1),
        "occurrence_with_ring_digit_pct": 100 * occurrence_share(with_digit),
        "structural_token_pct": 100 * structural_occurrences / max(n_tokens, 1),
        "ring_gap_p50": pct(ring_gaps, 50),
        "ring_gap_p95": pct(ring_gaps, 95),
        "ring_unmatched": ring_unmatched,
        "paren_gap_p50": pct(paren_gaps, 50),
        "paren_gap_p95": pct(paren_gaps, 95),
        "paren_depth_max": max(paren_depth) if paren_depth else 0,
        # --- efficiency ---
        "tokens_p50": int(np.percentile(lengths, 50)) if lengths.size else 0,
        "tokens_p95": int(np.percentile(lengths, 95)) if lengths.size else 0,
        "tokens_p99": int(np.percentile(lengths, 99)) if lengths.size else 0,
        "tokens_max": int(lengths.max()) if lengths.size else 0,
        "chars_per_token": n_chars / max(n_tokens, 1),
        "vocab_used": len(used),
        "vocab_dead": len(content) - len(used),
        "tokens_covering_half": cover50,
        "token_entropy_bits": entropy,
        "examples": [{"smiles": s, "tokens": t} for s, t in examples],
    }


COLUMNS = [
    ("name", "tokenizer", "{}"),
    ("vocab_size", "vocab", "{}"),
    ("roundtrip_failures", "rt fail", "{}"),
    ("unk_tokens", "unk", "{}"),
    ("atoms_split_pct", "atoms split %", "{:.3f}"),
    ("occurrence_partial_atom_pct", "partial-atom tok %", "{:.2f}"),
    ("cl_br_split", "Cl/Br split", "{}"),
    ("occurrence_dangling_pct", "dangling tok %", "{:.1f}"),
    ("occurrence_self_closing_pct", "self-closing tok %", "{:.1f}"),
    ("ring_closed_in_token_pct", "rings closed in tok %", "{:.1f}"),
    ("paren_closed_in_token_pct", "parens closed in tok %", "{:.1f}"),
    ("ring_gap_p50", "ring gap p50", "{:.0f}"),
    ("ring_gap_p95", "ring gap p95", "{:.0f}"),
    ("paren_gap_p95", "paren gap p95", "{:.0f}"),
    ("tokens_p50", "tok p50", "{}"),
    ("tokens_p99", "tok p99", "{}"),
    ("tokens_max", "tok max", "{}"),
    ("chars_per_token", "chars/tok", "{:.2f}"),
    ("vocab_dead", "dead vocab", "{}"),
    ("token_entropy_bits", "entropy bits", "{:.2f}"),
]


def format_table(reports: Sequence[Dict[str, object]]) -> str:
    header = [label for _, label, _ in COLUMNS]
    rows = []
    for rep in reports:
        row = []
        for key, _, fmt in COLUMNS:
            value = rep.get(key)
            row.append(fmt.format(value) if value is not None and not (
                isinstance(value, float) and math.isnan(value)) else "-")
        rows.append(row)
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
