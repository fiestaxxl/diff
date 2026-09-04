#!/usr/bin/env python3
"""Train the SMILES tokenizer (plus class weights and audit) from config.

    python scripts/train_tokenizer.py configs/tokenizer_chebi.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Set

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.data.sources import iter_smiles  # noqa: E402
from dimol.tokenization.audit import audit_tokenizer  # noqa: E402
from dimol.tokenization.smiles_tokenizer import SmilesTokenizer  # noqa: E402


def compute_class_weights(
    smiles_iter: Iterable[str],
    tok: SmilesTokenizer,
    vocab_size: int,
    pad_idx: int,
    special_ids: Set[int],
    strategy: str = "sqrt_inverse",
    smoothing: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Per-token class weights from token frequencies. From the old train_tokenizer.py."""
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

    for sid in special_ids:
        if 0 <= sid < vocab_size:
            weights[sid] = 0.0

    weights = weights.clamp(min=min_weight, max=max_weight)

    nonzero = weights > 0
    if nonzero.any():
        weights = weights * (nonzero.sum().float() / weights.sum())

    return weights


def main(cfg: DictConfig) -> None:
    log_config(cfg)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True) or {}
    tok_cfg = dict(prep.get("tokenizer") or {})  # already a plain dict
    source = dict(prep.get("source") or {})
    canonicalize = bool(prep.get("canonicalize", True))

    train_split = source.get("splits", {}).get("train", "train")
    extra_splits = list(tok_cfg.get("include_splits") or [])

    def smiles_iter():
        yield from iter_smiles(source, train_split, canonicalize=canonicalize)
        for split in extra_splits:
            yield from iter_smiles(source, split, canonicalize=canonicalize)

    vocab_size = int(tok_cfg.get("vocab_size", 512))
    print(f"Training BPE, vocab_size={vocab_size}")
    tokenizer = SmilesTokenizer.train(
        smiles_iter=smiles_iter(),
        vocab_size=vocab_size,
        min_frequency=int(tok_cfg.get("min_frequency", 10)),
        bracket_min_frequency=int(tok_cfg.get("bracket_min_frequency", 2)),
        show_progress=True,
    )

    out = Path(tok_cfg.get("out") or cfg.tokenizer.get("path", None) or "data/smiles_bpe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    print(f"Saved to {out}")
    print(f"Final vocab size: {tokenizer.vocab_size}")
    print(
        f"Special token IDs: pad={tokenizer.pad_id} bos={tokenizer.bos_id} "
        f"eos={tokenizer.eos_id} unk={tokenizer.unk_id} mask={tokenizer.mask_id}"
    )

    audit_out: Optional[str] = tok_cfg.get("audit_out")
    if audit_out:
        n_audit = int(tok_cfg.get("audit_molecules", 20000))
        sample = []
        for i, smi in enumerate(smiles_iter()):
            if i >= n_audit:
                break
            sample.append(smi)
        report = audit_tokenizer(tokenizer, sample)
        Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(
            f"[audit] molecules={report['n_molecules']} unk={report['n_unk_tokens']} "
            f"chars_per_token={report['chars_per_token']:.2f} -> {audit_out}"
        )

    weights_out = tok_cfg.get("class_weights_out")
    if weights_out:
        class_weights = compute_class_weights(
            smiles_iter(),
            tokenizer,
            vocab_size=tokenizer.vocab_size,
            pad_idx=tokenizer.pad_id,
            special_ids={tokenizer.pad_id, tokenizer.mask_id},
            strategy=str(tok_cfg.get("class_weights_strategy", "sqrt_inverse")),
            smoothing=float(tok_cfg.get("class_weights_smoothing", 1.0)),
            max_weight=float(tok_cfg.get("class_weights_max", 5.0)),
        )
        nz = class_weights > 0
        print(
            f"[class weights] nonzero={int(nz.sum())} mean={class_weights[nz].mean():.3f} "
            f"min={class_weights[nz].min():.3f} max={class_weights.max():.3f}"
        )
        Path(weights_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(class_weights, weights_out)
        print(f"Saved class weights to {weights_out}")


if __name__ == "__main__":
    main(load_from_argv())
