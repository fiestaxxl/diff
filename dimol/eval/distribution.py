"""Is the validity real, or bought by generating simpler molecules?

Validity alone is easy to game: a model that emits ethanol forever scores 100%. The
metrics here answer the other question, whether the generated set looks like the corpus,
and they are the ones the molecular generation literature reports:

* per-descriptor distances between the generated and the reference distribution, in units
  of the reference standard deviation, so a value near 0 means "same distribution" and 1
  means "off by a whole standard deviation";
* Frechet ChemNet Distance, the standard aggregate distribution metric for this task;
* Bemis-Murcko scaffold counts, which catch a model that varies decorations on one core;
* the share of trivially small molecules, which catches the failure this module exists for.

Everything is computed from valid molecules only, since invalid strings have no
descriptors, and every function takes plain SMILES strings.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence

# The descriptors, in the order they are reported. Each is a name and a callable built
# lazily, because importing rdkit at module import time makes the config tools slow.
DESCRIPTOR_NAMES = (
    "heavy_atoms",
    "mol_weight",
    "rings",
    "aromatic_rings",
    "fraction_csp3",
    "rotatable_bonds",
    "hbd",
    "hba",
    "logp",
    "tpsa",
    "qed",
    "sa_score",
    "bertz",
)


def _sascorer():
    """The synthetic accessibility score lives in rdkit's contrib tree, not the API."""
    from rdkit.Chem import RDConfig

    path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if path not in sys.path:
        sys.path.append(path)
    import sascorer  # noqa: E402

    return sascorer


def descriptors(mol) -> Dict[str, float]:
    """All thirteen descriptors of one molecule.

    Raises whatever rdkit raises. Some molecules that MolFromSmiles accepts still fail
    later calls, QED in particular, because it re-kekulizes: callers must be ready for
    that and `describe` counts those molecules separately instead of dropping them
    silently.
    """
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, QED, rdMolDescriptors

    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mol_weight": float(Descriptors.MolWt(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": float(QED.qed(mol)),
        "sa_score": float(_sascorer().calculateScore(mol)),
        "bertz": float(GraphDescriptors.BertzCT(mol)),
    }


def describe(smiles: Sequence[str], scaffolds: bool = True) -> Dict[str, object]:
    """Descriptor columns plus scaffolds for the valid molecules in `smiles`."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    columns: Dict[str, List[float]] = {name: [] for name in DESCRIPTOR_NAMES}
    scaffold_set: set = set()
    canon: List[str] = []
    undescribed = 0
    for smi in smiles:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon.append(Chem.MolToSmiles(mol))
        try:
            values = descriptors(mol)
        except Exception:
            undescribed += 1  # parses, but at least one descriptor refuses it
            continue
        for name, value in values.items():
            columns[name].append(value)
        if scaffolds:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_set.add(Chem.MolToSmiles(core))
            except Exception:  # a molecule rdkit parses but cannot scaffold
                pass
    return {"columns": columns, "scaffolds": scaffold_set, "canonical": canon,
            "undescribed": undescribed}


def _quantiles(values: Sequence[float], qs=(0.05, 0.5, 0.95)) -> List[float]:
    if not values:
        return [float("nan")] * len(qs)
    ordered = sorted(values)
    out = []
    for q in qs:
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        out.append(ordered[idx])
    return out


def _mean_sd(values: Sequence[float]):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def compare_descriptors(sample: Dict[str, List[float]],
                        reference: Dict[str, List[float]]) -> Dict[str, dict]:
    """Per descriptor: both distributions, the gap in reference sigmas, and a KS statistic."""
    from scipy.stats import ks_2samp, wasserstein_distance

    out: Dict[str, dict] = {}
    for name in DESCRIPTOR_NAMES:
        s, r = sample.get(name, []), reference.get(name, [])
        s_mean, s_sd = _mean_sd(s)
        r_mean, r_sd = _mean_sd(r)
        row = {
            "sample_mean": s_mean, "sample_sd": s_sd,
            "reference_mean": r_mean, "reference_sd": r_sd,
            "sample_q": _quantiles(s), "reference_q": _quantiles(r),
        }
        if s and r:
            scale = r_sd if r_sd > 1e-12 else 1.0
            row["shift_sigma"] = (s_mean - r_mean) / scale
            row["wasserstein_sigma"] = wasserstein_distance(s, r) / scale
            row["ks"] = float(ks_2samp(s, r).statistic)
        else:
            row["shift_sigma"] = row["wasserstein_sigma"] = row["ks"] = float("nan")
        out[name] = row
    return out


def trivial_share(columns: Dict[str, List[float]],
                  min_heavy_atoms: int = 10, min_rings: int = 1) -> Dict[str, float]:
    """How much of the generated set is too small or too flat to count as a hit.

    The thresholds are deliberately generous against the model: ZINC-250k molecules have
    a 5th percentile of about 17 heavy atoms and every one of them has at least one ring,
    so anything under 10 heavy atoms or with no ring at all is not a molecule this corpus
    would contain.
    """
    heavy = columns.get("heavy_atoms", [])
    rings = columns.get("rings", [])
    if not heavy:
        return {"tiny": float("nan"), "acyclic": float("nan"), "trivial": float("nan")}
    tiny = sum(1 for h in heavy if h < min_heavy_atoms)
    acyclic = sum(1 for r in rings if r < min_rings)
    both = sum(1 for h, r in zip(heavy, rings) if h < min_heavy_atoms or r < min_rings)
    n = len(heavy)
    return {"tiny": tiny / n, "acyclic": acyclic / n, "trivial": both / n}


BRACKET_ATOM = None  # compiled on first use


def bracket_atoms(smiles: Sequence[str]) -> Dict[str, int]:
    """Every bracket atom token in a set of SMILES, with counts."""
    global BRACKET_ATOM
    if BRACKET_ATOM is None:
        import re

        BRACKET_ATOM = re.compile(r"\[[^\]]+\]")
    counts: Dict[str, int] = {}
    for smi in smiles:
        for token in BRACKET_ATOM.findall(smi or ""):
            counts[token] = counts.get(token, 0) + 1
    return counts


def bracket_report(sample_canonical: Sequence[str],
                   reference_canonical: Sequence[str]) -> Dict[str, object]:
    """Bracket atoms the corpus never contains, and how many molecules carry one.

    This is the cheapest test for chemistry the model made up. A corpus like ZINC-250k
    uses a small closed set of bracket atoms, charged nitrogens and stereocentres mostly,
    so a generated [CH] or a bare [P] is not a rare molecule, it is a wrong one. Counted
    on canonical strings, so rdkit has already normalised the spelling.
    """
    known = set(bracket_atoms(reference_canonical))
    counts = bracket_atoms(sample_canonical)
    unseen = {token: n for token, n in counts.items() if token not in known}
    carriers = 0
    for smi in sample_canonical:
        if any(token not in known for token in bracket_atoms([smi])):
            carriers += 1
    n = max(len(sample_canonical), 1)
    top = sorted(unseen.items(), key=lambda kv: -kv[1])[:8]
    return {
        "unseen_kinds": len(unseen),
        "unseen_total": sum(unseen.values()),
        "molecules_with_unseen": carriers / n,
        "top_unseen": top,
        "reference_kinds": len(known),
    }


def roundtrip_safe(smiles: Sequence[str]) -> List[str]:
    """Keep only strings rdkit can read back after writing them.

    rdkit can emit a canonical SMILES it then refuses to parse, usually an aromatic ring
    that will not kekulize on the way in. Anything downstream that re-parses, the ChemNet
    distance included, chokes on those, so they are filtered out explicitly rather than
    crashing a metric run.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return [s for s in smiles if s and Chem.MolFromSmiles(s) is not None]


def fcd(sample_canonical: Sequence[str], reference_canonical: Sequence[str],
        device: str = "cpu", n_jobs: int = 1) -> float:
    """Frechet ChemNet Distance. Lower is closer; identical sets give 0."""
    sample = roundtrip_safe(sample_canonical)
    reference = roundtrip_safe(reference_canonical)
    if len(sample) < 2 or len(reference) < 2:
        return float("nan")
    from fcd_torch import FCD

    return float(FCD(device=device, n_jobs=n_jobs)(sample, reference))


def scaffold_metrics(sample: Dict[str, object], reference: Dict[str, object]) -> Dict[str, float]:
    sample_scaffolds = sample["scaffolds"]
    reference_scaffolds = reference["scaffolds"]
    n_valid = max(len(sample["canonical"]), 1)
    shared = sample_scaffolds & reference_scaffolds
    return {
        "scaffolds": float(len(sample_scaffolds)),
        "scaffolds_per_valid": len(sample_scaffolds) / n_valid,
        "scaffold_novelty": 1.0 - (len(shared) / max(len(sample_scaffolds), 1)),
        "reference_scaffolds": float(len(reference_scaffolds)),
    }


def usable_molecules(sample_smiles: Sequence[str], reference_canonical: Sequence[str],
                     min_heavy_atoms: int = 10, min_rings: int = 1) -> Dict[str, object]:
    """Distinct molecules that a chemist would accept, as a share of all attempts.

    Validity on its own is gameable in three ways at once, and this study hit all three:
    a repair mode can trim molecules down to fragments, a loss change can make the model
    terminate early, and a decoder can insert atoms that the corpus never contains. A
    molecule counts here only if it parses, is distinct, has at least ten heavy atoms and
    one ring, and uses no bracket atom absent from the reference. The denominator is the
    number of attempts, so nothing is hidden by dropping empty outputs.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    known_brackets = set(bracket_atoms(reference_canonical))
    seen: set = set()
    rejected = {"invalid": 0, "duplicate": 0, "too_small": 0, "unseen_atom": 0}
    for smi in sample_smiles:
        if not smi:
            rejected["invalid"] += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected["invalid"] += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            rejected["duplicate"] += 1
            continue
        from rdkit.Chem import rdMolDescriptors

        if (mol.GetNumHeavyAtoms() < min_heavy_atoms
                or rdMolDescriptors.CalcNumRings(mol) < min_rings):
            rejected["too_small"] += 1
            continue
        if any(token not in known_brackets for token in bracket_atoms([canonical])):
            rejected["unseen_atom"] += 1
            continue
        seen.add(canonical)
    attempts = max(len(sample_smiles), 1)
    return {"count": len(seen), "attempts": attempts, "rate": len(seen) / attempts,
            "rejected": rejected}


def evaluate_distribution(sample_smiles: Sequence[str], reference_smiles: Sequence[str],
                          with_fcd: bool = True, device: str = "cpu") -> Dict[str, object]:
    sample = describe(sample_smiles)
    reference = describe(reference_smiles)
    report: Dict[str, object] = {
        "usable": usable_molecules(sample_smiles, reference["canonical"]),
        "n_valid": len(sample["canonical"]),
        "n_described": len(sample["columns"]["heavy_atoms"]),
        "n_undescribed": sample["undescribed"],
        "n_reference": len(reference["canonical"]),
        "descriptors": compare_descriptors(sample["columns"], reference["columns"]),
        "trivial": trivial_share(sample["columns"]),
        "scaffolds": scaffold_metrics(sample, reference),
        "brackets": bracket_report(sample["canonical"], reference["canonical"]),
    }
    if with_fcd:
        safe = roundtrip_safe(sample["canonical"])
        report["n_fcd"] = len(safe)
        report["fcd"] = fcd(sample["canonical"], reference["canonical"], device=device)
    return report


def format_distribution(report: Dict[str, object], name: str = "") -> str:
    lines = []
    head = f"=== {name}" if name else "==="
    lines.append(head)
    u = report["usable"]
    lines.append(f"usable molecules:          {u['count']} of {u['attempts']} attempts"
                 f"   ({u['rate'] * 100:.2f}%)")
    lines.append("  rejected: " + ", ".join(f"{k} {v}" for k, v in u["rejected"].items()))
    lines.append(f"valid molecules:           {report['n_valid']}")
    lines.append(f"of them fully described:   {report['n_described']}"
                 f"   ({report['n_undescribed']} rejected by a descriptor)")
    if "fcd" in report:
        lines.append(f"FCD vs reference:          {report['fcd']:.3f}"
                     f"   (0 = same distribution, on {report.get('n_fcd', 0)} molecules)")
    triv = report["trivial"]
    lines.append(f"too small (<10 heavy):     {triv['tiny'] * 100:.2f}%")
    lines.append(f"no ring at all:            {triv['acyclic'] * 100:.2f}%")
    lines.append(f"trivial by either rule:    {triv['trivial'] * 100:.2f}%")
    br = report["brackets"]
    lines.append(f"molecules with a bracket atom the corpus never uses: "
                 f"{br['molecules_with_unseen'] * 100:.2f}%"
                 f"  ({br['unseen_kinds']} kinds against {br['reference_kinds']} known)")
    if br["top_unseen"]:
        lines.append("  most common: " + ", ".join(f"{t} x{n}" for t, n in br["top_unseen"]))
    sc = report["scaffolds"]
    lines.append(f"distinct scaffolds:        {sc['scaffolds']:.0f}"
                 f"  ({sc['scaffolds_per_valid'] * 100:.1f}% of valid,"
                 f" {sc['scaffold_novelty'] * 100:.1f}% unseen in the reference)")
    lines.append("")
    lines.append(f"{'descriptor':<17}{'generated':>20}{'corpus':>20}{'shift':>8}{'W1':>7}{'KS':>7}")
    for key, row in report["descriptors"].items():
        gen = f"{row['sample_mean']:.2f} +- {row['sample_sd']:.2f}"
        ref = f"{row['reference_mean']:.2f} +- {row['reference_sd']:.2f}"
        lines.append(f"{key:<17}{gen:>20}{ref:>20}"
                     f"{row['shift_sigma']:>8.2f}{row['wasserstein_sigma']:>7.2f}{row['ks']:>7.2f}")
    lines.append("shift and W1 are in corpus standard deviations; KS is the two-sample statistic")
    return "\n".join(lines)
