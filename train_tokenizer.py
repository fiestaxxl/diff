"""Train a SMILES BPE tokenizer on ChEBI-20-MM."""
import argparse
from pathlib import Path
from datasets import load_dataset
from smiles_tokenizer import SmilesTokenizer

from datasets import load_dataset
import os

import torch
from collections import Counter
from typing import Iterable, Set

def compute_class_weights(
    smiles_iter: Iterable[int],
    tok: SmilesTokenizer,
    vocab_size: int,
    pad_idx: int,
    special_ids: Set[int],
    strategy: str = "sqrt_inverse",
    smoothing: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute per-token class weights from a token-id stream.

    Args:
        token_id_iter: iterable yielding token IDs from training data.
            Could be a flat iterator over all training tokens, or a list
            of per-molecule token-id lists (will be flattened).
        vocab_size: total vocab size
        pad_idx: pad token ID — gets weight 0 (also covered by ignore_index)
        special_ids: set of special token IDs (pad, bos, eos, unk, mask)
        strategy: "sqrt_inverse" (recommended), "linear_inverse", or "none"
        smoothing: added to counts before inverting (avoids div-by-zero,
            softens weight on very rare tokens)
        min_weight, max_weight: clamp range for final weights to prevent
            extreme values

    Returns:
        Tensor of shape (vocab_size,) with weights normalized so that
        the mean weight (over non-zero entries) equals 1.0.
    """
    counts = torch.zeros(vocab_size, dtype=torch.float32)
    for sm in smiles_iter:
        item = tok.encode(sm, add_special_tokens=True)
        if isinstance(item, int):
            counts[item] += 1
        else:
            for tok_id in item:
                counts[tok_id] += 1

    weights = torch.zeros(vocab_size, dtype=torch.float32)

    if strategy == "none":
        weights[:] = 1.0
    elif strategy == "sqrt_inverse":
        used = counts > 0
        weights[used] = 1.0 / (counts[used] + smoothing).sqrt()
    elif strategy == "linear_inverse":
        used = counts > 0
        weights[used] = 1.0 / (counts[used] + smoothing)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Zero out specials (they'll be ignored via ignore_index, but explicit is good)
    for sid in special_ids:
        if 0 <= sid < vocab_size:
            weights[sid] = 0.0

    # Clamp to prevent extreme values from rare tokens
    weights = weights.clamp(min=min_weight, max=max_weight)

    # Normalize so mean weight (over non-zero) equals 1.0
    nonzero = weights > 0
    if nonzero.any():
        weights = weights * (nonzero.sum().float() / weights.sum())

    return weights

def iter_train_token_ids():
    for sample in train_dataset:
        # adapt to your dataset structure - get the token_ids
        yield sample['token_ids'].tolist() if hasattr(sample['token_ids'], 'tolist') \
              else list(sample['token_ids'])

def _canon(smi: str) -> str | None:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--min_frequency", type=int, default=10)
    ap.add_argument("--bracket_min_frequency", type=int, default=2)
    ap.add_argument("--canonicalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=Path, default=Path("data/smiles_bpe.json"))
    ap.add_argument("--weights", type=Path, default=Path("data/class_weights.pt"))
    ap.add_argument("--include_validation", action="store_true",
                    help="Train on train+validation (use only if not tuning).")
    args = ap.parse_args()

    ds = load_dataset("liupf/ChEBI-20-MM")
    print('Loaded dataset: \n', ds)


    def smiles_iter():
        for ex in ds["train"]:
            s = _canon(ex["SMILES"]) if args.canonicalize else ex["SMILES"]
            if s:
                yield s
        if args.include_validation:
            for ex in ds["validation"]:
                s = _canon(ex["SMILES"]) if args.canonicalize else ex["SMILES"]
                if s:
                    yield s

    n_train = len(ds["train"]) + (len(ds["validation"]) if args.include_validation else 0)
    print(f"Training BPE on {n_train} SMILES, vocab_size={args.vocab_size}")

    tokenizer = SmilesTokenizer.train(
        smiles_iter=smiles_iter(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        bracket_min_frequency=args.bracket_min_frequency,
        show_progress=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"Saved to {args.out}")
    print(f"Final vocab size: {tokenizer.vocab_size}")
    print(f"Special token IDs: pad={tokenizer.pad_id} bos={tokenizer.bos_id} eos={tokenizer.eos_id} "
          f"unk={tokenizer.unk_id} mask={tokenizer.mask_id}")

    special_ids = {
        tokenizer.pad_id,
        tokenizer.mask_id,
    }

    class_weights = compute_class_weights(
        smiles_iter(),
        tokenizer,
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_id,
        special_ids=special_ids,
        strategy="sqrt_inverse",
        smoothing=1.0,
        max_weight=5.0,  # cap rare-token weight to prevent instability
    )
    # Quick sanity print
    print(f"Class weights stats:")
    print(f"  num tokens with weight > 0: {(class_weights > 0).sum().item()}")
    print(f"  mean weight: {class_weights[class_weights > 0].mean().item():.3f}")
    print(f"  min weight: {class_weights[class_weights > 0].min().item():.3f}")
    print(f"  max weight: {class_weights.max().item():.3f}")

    # Show a few examples
    vocab = tokenizer.get_vocab()
    id_to_tok = {v: k for k, v in vocab.items()}
    print("\nWeights for top tokens:")
    for tok in vocab:
        tid = vocab[tok]
        print(f"  {tok!r:15s} weight={class_weights[tid].item():.3f}")

    torch.save(class_weights, args.weights)

if __name__ == "__main__":
    main()