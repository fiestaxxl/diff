"""Sanity-check a trained SMILES BPE tokenizer."""
import argparse
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from smiles_tokenizer import SmilesTokenizer


def round_trip(tok: SmilesTokenizer, smiles_list):
    """Every SMILES must encode/decode to itself."""
    fails = []
    for s in smiles_list:
        ids = tok.encode(s)
        d = tok.decode(ids)
        if d != s:
            fails.append((s, d))
    return fails


def unk_rate(tok: SmilesTokenizer, smiles_list):
    unk = tok.unk_id
    total, n_unk, with_unk = 0, 0, 0
    for s in smiles_list:
        ids = tok.encode(s, add_special_tokens=False)
        total += len(ids)
        c = sum(1 for i in ids if i == unk)
        n_unk += c
        if c:
            with_unk += 1
    return {
        "tokens_total": total,
        "tokens_unk": n_unk,
        "rate": n_unk / max(total, 1),
        "molecules_with_unk": with_unk,
        "molecules_total": len(smiles_list),
    }


def length_stats(tok: SmilesTokenizer, smiles_list):
    lens = [len(tok.encode(s, add_special_tokens=False)) for s in smiles_list]
    chars = [len(s) for s in smiles_list]
    compression = [c / l for c, l in zip(chars, lens) if l]
    return {
        "tokens_per_mol_mean":   statistics.mean(lens),
        "tokens_per_mol_median": statistics.median(lens),
        "tokens_per_mol_p95":    sorted(lens)[int(len(lens) * 0.95)],
        "tokens_per_mol_p99":    sorted(lens)[int(len(lens) * 0.99)],
        "tokens_per_mol_max":    max(lens),
        "chars_per_token_mean":  statistics.mean(compression),
    }


def rdkit_validity(tok: SmilesTokenizer, smiles_list, sample=2000):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    smiles_list = smiles_list[:sample]
    n_orig_valid = 0
    n_decoded_valid = 0
    for s in smiles_list:
        if Chem.MolFromSmiles(s) is not None:
            n_orig_valid += 1
        d = tok.decode(tok.encode(s))
        if Chem.MolFromSmiles(d) is not None:
            n_decoded_valid += 1
    return {
        "checked": len(smiles_list),
        "original_valid": n_orig_valid,
        "decoded_valid": n_decoded_valid,
    }


def show_examples(tok: SmilesTokenizer, smiles_list, n=8):
    print("\n--- Tokenization examples ---")
    for s in smiles_list[:n]:
        toks = tok.tokenize(s)
        print(f"\n  SMILES ({len(s)} chars): {s}")
        print(f"  Tokens ({len(toks)}): {toks}")


def vocab_inspection(tok: SmilesTokenizer, smiles_list, top=20):
    print("\n--- Most common tokens (excluding specials) ---")
    counter = Counter()
    specials = set(SmilesTokenizer.SPECIAL_TOKENS)
    for s in smiles_list:
        for t in tok.tokenize(s):
            if t not in specials:
                counter[t] += 1
    for t, c in counter.most_common(top):
        print(f"  {t!r:30s} {c}")

    vocab = tok.get_vocab()
    used = set(counter)
    unused = [t for t in vocab if t not in used and t not in specials]
    print(f"\n  vocab size: {len(vocab)}")
    print(f"  tokens used in test set: {len(used)}")
    print(f"  unused tokens: {len(unused)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/smiles_bpe.json"))
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    tok = SmilesTokenizer.load(args.tokenizer)
    ds = load_dataset("liupf/ChEBI-20-MM")
    smiles = [ex["SMILES"] for ex in ds[args.split] if ex["SMILES"]]

    print(f"Loaded tokenizer: vocab_size={tok.vocab_size}")
    print(f"Evaluating on {args.split} ({len(smiles)} molecules)\n")

    print("--- Round-trip ---")
    fails = round_trip(tok, smiles)
    print(f"  {len(smiles) - len(fails)} / {len(smiles)} round-trip OK")
    for orig, dec in fails[:5]:
        print(f"    ORIG: {orig}\n    DEC : {dec}\n")

    print("\n--- <unk> rate ---")
    for k, v in unk_rate(tok, smiles).items():
        print(f"  {k:25s} {v}")

    print("\n--- Length stats ---")
    for k, v in length_stats(tok, smiles).items():
        print(f"  {k:25s} {v:.2f}" if isinstance(v, float) else f"  {k:25s} {v}")

    print("\n--- RDKit validity ---")
    rd = rdkit_validity(tok, smiles)
    if rd is None:
        print("  RDKit not installed, skipping.")
    else:
        for k, v in rd.items():
            print(f"  {k:25s} {v}")

    show_examples(tok, smiles, n=8)
    vocab_inspection(tok, smiles, top=300)

    # sm = 'CCCCC'
    # enc = tok.encode_padded(sm, add_special_tokens=True, max_length=128)
    # print(enc)
    # dec = tok.decode(enc[0], skip_special_tokens=False)
    # print(dec)

    # batch_sm = [enc[0], enc[0], enc[0]]
    # print(tok.decode_batch(batch_sm))



if __name__ == "__main__":
    main()