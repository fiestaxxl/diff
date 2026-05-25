
import numpy as np
from rdkit import Chem, DataStructs, RDLogger 
from rdkit.Chem import AllChem, MACCSkeys
import itertools
import numpy as np   
RDLogger.DisableLog('rdApp.*') 

import math
from collections import Counter


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
from fcd import get_fcd



def extract_leftmost_between_special_tokens(
    text: str,
    start_token: str = "<s>",
    end_token: str = "</s>"
):
    """
    Extract the leftmost substring between <s> and </s>.
    Returns None if no valid leftmost span exists.
    """


    start_idx = text.find(start_token)
    if start_idx == -1:
        return None

    start_idx += len(start_token)

    end_idx = text.find(end_token, start_idx)
    if end_idx == -1:
        return None

    content = text[start_idx:end_idx].strip()
    return content if len(content) > 0 else None



def decode_ids_to_smiles(
    ids_batch,
    tokenizer,
    start_token="<s>",
    end_token="</s>"
):
    smiles = []
    raw = []

    for ids in ids_batch:
        text = tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

        smi = extract_leftmost_between_special_tokens(
            text,
            start_token=start_token,
            end_token=end_token
        )

        smiles.append(smi)
        raw.append(text)

    return smiles, raw


def is_valid_smiles(smiles: str) -> bool:
    if smiles is None:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_fps(
    smiles_list,
    radius=2,
    n_bits=2048
):
    fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits
        )
        fps.append(fp)

    return fps

def smiles_bleu(reference_smiles, generated_smiles):
    """
    Character-level BLEU
    """
    smoothie = SmoothingFunction().method4
    scores = []

    for ref, gen in zip(reference_smiles, generated_smiles):
        if ref is None or gen is None:
            continue
        ref_chars = list(ref)
        gen_chars = list(gen)
        score = sentence_bleu(
            [ref_chars],
            gen_chars,
            smoothing_function=smoothie
        )
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def exact_match_score(reference_smiles, generated_smiles):
    """
    Canonical SMILES exact match
    """
    matches = 0
    total = 0

    for ref, gen in zip(reference_smiles, generated_smiles):
        if ref is None or gen is None:
            continue

        ref_can = canonicalize_smiles(ref)
        gen_can = canonicalize_smiles(gen)

        if ref_can is None or gen_can is None:
            continue

        total += 1
        matches += int(ref_can == gen_can)

    return matches / total if total > 0 else 0.0


def average_levenshtein(reference_smiles, generated_smiles):
    distances = []

    for ref, gen in zip(reference_smiles, generated_smiles):
        if ref is None or gen is None:
            continue
        distances.append(Levenshtein.distance(ref, gen))

    return float(np.mean(distances)) if distances else 0.0



def ntel(smiles):
    if len(smiles) == 0:
        return 0.0

    counts = Counter(smiles)
    total = len(smiles)

    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs)

    max_entropy = math.log(len(counts))
    if max_entropy == 0:
        return 0.0

    entropy_norm = entropy / max_entropy
    length_term = math.log(1 + total)

    return entropy_norm * length_term


def average_ntel(smiles_list):
    return sum(ntel(s) for s in smiles_list) / max(len(smiles_list), 1)

def average_tanimoto(fp_pairs):
    sims = [
        DataStructs.TanimotoSimilarity(fp1, fp2)
        for fp1, fp2 in fp_pairs
        if fp1 is not None and fp2 is not None
    ]
    return float(np.mean(sims)) if sims else 0.0

def maccs_fps(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(MACCSkeys.GenMACCSKeys(mol) if mol else None)
    return fps

def rdk_fps(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(Chem.RDKFingerprint(mol) if mol else None)
    return fps


def morgan_fps(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            if mol else None
        )
    return fps

def fingerprint_similarity_metrics(reference_smiles, generated_smiles):
    ref_maccs = maccs_fps(reference_smiles)
    gen_maccs = maccs_fps(generated_smiles)

    ref_rdk = rdk_fps(reference_smiles)
    gen_rdk = rdk_fps(generated_smiles)

    ref_morgan = morgan_fps(reference_smiles)
    gen_morgan = morgan_fps(generated_smiles)

    return {
        "MACCS_FTS": average_tanimoto(zip(ref_maccs, gen_maccs)),
        "RDK_FTS": average_tanimoto(zip(ref_rdk, gen_rdk)),
        "Morgan_FTS": average_tanimoto(zip(ref_morgan, gen_morgan)),
    }


def calc_diversity(smiles_list, radius=2, n_bits=1024):
    """
    Compute molecular diversity as 1 - average pairwise Tanimoto similarity.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        radius (int): Morgan fingerprint radius.
        n_bits (int): Number of fingerprint bits.

    Returns:
        float: Diversity score.
    """
    # Convert SMILES to molecules, filter invalid ones
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]

    N = len(mols)
    if N < 2:
        return 0.0

    # Compute Morgan fingerprints
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        for m in mols
    ]

    # Compute pairwise Tanimoto similarities
    similarities = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i, j in itertools.combinations(range(N), 2)
    ]

    avg_similarity = sum(similarities) * 2 / (N * (N - 1))

    return 1.0 - avg_similarity


def compute_fcd(reference_smiles, generated_smiles):
    return get_fcd(reference_smiles, generated_smiles)

def compute_smiles_metrics(
    smiles_list,
    reference_list=None,
    train_smiles_set=None
):
    """
    smiles_list: List[str | None]
    train_smiles_set: set of canonical SMILES (optional)

    Returns dict of metrics
    """
    metrics = {}

    total = len(smiles_list)

    valid_smiles = [(idx, s) for idx, s in enumerate(smiles_list) if is_valid_smiles(s)]
    num_valid = len(valid_smiles)

    validity = num_valid / total if total > 0 else 0.0

    # canonicalize
    canonical = [canonicalize_smiles(s[1]) for s in valid_smiles]
    canonical = [s for s in canonical if s is not None]

    unique = set(canonical)
    uniqueness = len(unique) / len(canonical) if canonical else 0.0

    fps = smiles_to_fps(unique)
    diversity = calc_diversity(canonical)


    if reference_list is not None:
        metrics["BLEU"] = smiles_bleu(reference_list, canonical)
        metrics["Exact"] = exact_match_score(reference_list, canonical)
        metrics["Levenshtein"] = average_levenshtein(reference_list, canonical)
        metrics.update(
            fingerprint_similarity_metrics(reference_list, canonical)
        )


    # novelty = None
    # if train_smiles_set is not None:
    #     novelty = sum(s not in train_smiles_set for s in unique) / len(unique) if unique else 0.0

    metrics.update({
            "total": total,
            "valid": num_valid,
            "validity": validity,
            "unique": len(unique),
            "uniqueness": uniqueness,
            "diversity": diversity
        }
    )

    return metrics, unique, valid_smiles
