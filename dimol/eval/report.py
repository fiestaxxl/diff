"""Quality report for generated SMILES.

Carried over VERBATIM from the old generate.py: the Wilson interval, the metric
panel (validity / uniqueness / novelty / diversity / failure breakdown) and the
report formatting. rdkit is imported lazily.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import List, Optional


def _canon(smi: Optional[str]) -> Optional[str]:
    from rdkit import Chem

    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval — correct for small proportions (validity ~3%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_smiles(
    generated: List[str],
    train_canon: Optional[set] = None,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    div_subsample: int = 1000,
) -> dict:
    """`generated` are already-truncated decoded SMILES strings."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    n = len(generated)
    canon = [_canon(s) for s in generated]
    valid = [c for c in canon if c is not None]
    nv = len(valid)
    uniq_valid = set(valid)

    m: dict = {"n": n, "n_valid": nv}

    m["validity"] = nv / n if n else 0.0
    m["validity_ci95"] = _wilson_ci(nv, n)

    m["uniqueness"] = len(uniq_valid) / nv if nv else 0.0          # among valid

    if train_canon is not None and uniq_valid:
        novel = sum(1 for c in uniq_valid if c not in train_canon)
        m["novelty"] = novel / len(uniq_valid)
    else:
        m["novelty"] = float("nan")

    # internal diversity = 1 - mean pairwise Tanimoto (Morgan), subsampled
    div = float("nan")
    pool = list(uniq_valid)[:div_subsample]
    if len(pool) > 1:
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), fp_radius, fp_bits)
            for s in pool
        ]
        sims: List[float] = []
        for i in range(len(fps)):
            sims += DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        if sims:
            div = 1.0 - sum(sims) / len(sims)
    m["diversity"] = div

    # failure breakdown — counted on the decoded STRING (BPE-agnostic)
    reasons: Counter = Counter()
    for s, c in zip(generated, canon):
        if c is None:
            s = s or ""
            if s.count("(") != s.count(")"):
                reasons["unbalanced_parens"] += 1
            elif any(s.count(str(d)) % 2 for d in range(1, 10)):
                reasons["odd_ring_digits"] += 1
            elif len(s) == 0:
                reasons["empty"] += 1
            else:
                reasons["other"] += 1
    m["failures"] = dict(reasons)
    m["mean_len"] = sum(len(s or "") for s in generated) / n if n else 0.0
    return m



def format_report(m: dict) -> str:
    lo, hi = m["validity_ci95"]
    return "\n".join([
        "================ QUALITY REPORT ================",
        f"samples:    {m['n']}",
        f"validity:   {m['validity']:.3%}   (95% CI {lo:.2%} - {hi:.2%})",
        f"uniqueness: {m['uniqueness']:.3%}   (among {m['n_valid']} valid)",
        f"novelty:    {m['novelty']:.3%}",
        f"diversity:  {m['diversity']:.3f}",
        f"mean_len:   {m['mean_len']:.1f} chars",
        f"failures:   {m['failures']}",
        "===============================================",
    ])


def load_train_canon(path: Optional[str]) -> Optional[set]:
    """Canonical training SMILES for novelty (file with one molecule per line)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[novelty] train smiles file not found: {p} -> skipping novelty")
        return None
    canon = set()
    with open(p) as f:
        for line in f:
            c = _canon(line.strip())
            if c is not None:
                canon.add(c)
    print(f"[novelty] loaded {len(canon)} canonical training SMILES")
    return canon
