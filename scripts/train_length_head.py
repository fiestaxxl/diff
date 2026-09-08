#!/usr/bin/env python3
"""Train the caption-to-length head. Minutes on one GPU, seconds of data.

    python scripts/train_length_head.py --config configs/chebi20_finetune.yaml \
        --out data/chebi20/length_head.pt

Reports the mean absolute error against two references: predicting the corpus mean, which
is what ignoring the caption gets you, and the ridge regression on mean-pooled states,
which is what a linear model on a worse summary gets you. If the head cannot beat those it
is not worth carrying.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.models.length_head import LengthHead, length_loss  # noqa: E402


def load_split(data_dir: Path, split: str):
    text = np.load(data_dir / f"{split}_text.npy", mmap_mode="r")
    mask = np.load(data_dir / f"{split}_text_mask.npy", mmap_mode="r")
    lengths = np.load(data_dir / f"{split}_attn_mask_00000.npy", mmap_mode="r").sum(1)
    return text, mask, np.asarray(lengths, dtype=np.int64)


def batches(text, mask, lengths, size, device, shuffle=True, generator=None):
    order = torch.randperm(len(lengths), generator=generator) if shuffle \
        else torch.arange(len(lengths))
    for start in range(0, len(order), size):
        idx = order[start : start + size].numpy()
        idx.sort()
        yield (torch.from_numpy(np.asarray(text[idx], dtype=np.float32)).to(device),
               torch.from_numpy(np.asarray(mask[idx]).astype(bool)).to(device),
               torch.from_numpy(lengths[idx]).to(device))


def evaluate(head, text, mask, lengths, device, batch_size):
    errors, exact = [], 0
    for t, m, y in batches(text, mask, lengths, batch_size, device, shuffle=False):
        pred = head.predict(t, m)
        errors.append((pred - y).abs().float().cpu())
        exact += int((pred == y).sum())
    err = torch.cat(errors)
    return err.mean().item(), (err <= 3).float().mean().item() * 100, exact / len(lengths) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_dir = Path(cfg.variables.data_dir)
    canvas = int(cfg.variables.seq_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train = load_split(data_dir, "train")
    val = load_split(data_dir, "val")
    text_dim = train[0].shape[2]
    print(f"train {len(train[2])}, val {len(val[2])}, caption width {text_dim}, "
          f"canvas {canvas}, on {device}")
    print(f"lengths: mean {train[2].mean():.1f}, sd {train[2].std():.1f}")
    baseline = np.abs(train[2].mean() - val[2]).mean()
    print(f"predicting the corpus mean gives MAE {baseline:.2f} tokens")

    head = LengthHead(text_dim=text_dim, canvas=canvas).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    best = (float("inf"), None)
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for t, m, y in batches(*train, args.batch_size, device, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            loss = length_loss(head(t, m), y.clamp(0, canvas))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        head.eval()
        mae, within3, exact = evaluate(head, *val, device, args.batch_size)
        if mae < best[0]:
            best = (mae, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}: loss {np.mean(losses):.3f}, val MAE {mae:5.2f}, "
                  f"within 3 {within3:4.1f}%, exact {exact:4.1f}%")

    head.load_state_dict(best[1])
    head.save(args.out)
    mae, within3, exact = evaluate(head, *val, device, args.batch_size)
    print(f"best: MAE {mae:.2f} tokens against {baseline:.2f} for the corpus mean "
          f"and 9.03 for ridge on a mean pool; within 3 tokens {within3:.1f}%, "
          f"exact {exact:.1f}%")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
