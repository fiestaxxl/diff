#!/usr/bin/env python3
"""Did the content channel learn anything, or is the decision rule at fault?

    python scripts/analyze_content_channel.py --config configs/chebi20_finetune.yaml \
        --checkpoint runs/cc_both/ep1616-ba160000 --split test --limit 500

The content channel was supposed to let the model decide where its molecule ends. Measured
against an external length predictor it did much worse: with the channel alone, validity
0.159 and MACCS 0.141; with the channel plus a predicted length, 0.564 and 0.525, against
0.763 and 0.755 for the predictor by itself.

Two very different causes produce that, and they call for opposite responses:

* **The channel carries no signal.** Then the architecture is wrong and the idea is dead.
* **The channel carries signal but the readback is brittle.** The sampler truncates at the
  FIRST position where the channel turns negative, so a single noisy position early in the
  string amputates the molecule. Then the idea survives and the rule needs work.

This separates them. It reports where the channel crosses zero against the true length,
how sharp the crossing is, and what several alternative decision rules would have given —
first crossing, last positive, first run of k consecutive negatives, and a swept threshold.
If a better rule recovers the length, the channel learned something.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_path, build_tokenizer  # noqa: E402
from dimol.eval.sampling import SamplingParams, sample_smiles  # noqa: E402
from dimol.models.diffusion_transformer import DiffusionTransformer  # noqa: E402


def first_negative(channel: np.ndarray, canvas: int) -> np.ndarray:
    """The rule the sampler uses: truncate at the first position below zero."""
    negative = channel < 0
    return np.where(negative.any(1), negative.argmax(1), canvas)


def first_run(channel: np.ndarray, canvas: int, k: int) -> np.ndarray:
    """First start of k consecutive negative positions — robust to isolated noise."""
    negative = channel < 0
    n, length = negative.shape
    out = np.full(n, canvas)
    if k > length:
        return out
    # A run of k starts at i when negative[i:i+k] are all true.
    windows = np.lib.stride_tricks.sliding_window_view(negative, k, axis=1).all(-1)
    has = windows.any(1)
    out[has] = windows.argmax(1)[has]
    return out


def last_positive(channel: np.ndarray, canvas: int) -> np.ndarray:
    """One past the last position the channel calls content."""
    positive = channel > 0
    out = np.zeros(len(channel), dtype=int)
    has = positive.any(1)
    out[has] = (channel.shape[1] - 1) - positive[:, ::-1].argmax(1)[has] + 1
    return np.minimum(out, canvas)


def report(name: str, predicted: np.ndarray, true: np.ndarray) -> dict:
    error = predicted - true
    return {
        "rule": name,
        "exact": float((error == 0).mean()),
        "within_3": float((np.abs(error) <= 3).mean()),
        "mae": float(np.abs(error).mean()),
        "median_error": float(np.median(error)),
        "too_early": float((error < 0).mean()),
        "too_late": float((error > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--length-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer(cfg)
    model = DiffusionTransformer.from_pretrained(load_dir=str(args.checkpoint),
                                                 map_location=device)
    model.eval()
    if not getattr(model, "content_channel", False):
        raise SystemExit("this checkpoint has no content channel")

    canvas = int(cfg.variables.seq_len)
    width = getattr(model, "latent_dim", model.config.emb_dim)
    path = build_path(cfg, seq_len=canvas, emb_dim=width, device=device)

    data_dir = Path(cfg.variables.data_dir)
    text = np.load(data_dir / f"{args.split}_text.npy", mmap_mode="r")
    text_mask = np.load(data_dir / f"{args.split}_text_mask.npy", mmap_mode="r")
    masks = np.load(data_dir / f"{args.split}_attn_mask_00000.npy", mmap_mode="r")
    n = min(args.limit, len(text))
    true_lengths = np.asarray(masks[:n]).sum(1).astype(np.int64)

    lengths = None
    if args.length_file is not None:
        lengths = np.asarray(np.load(args.length_file)[:n], dtype=np.int64)

    params = SamplingParams(
        num_samples=n, batch_size=args.batch_size, num_timesteps=args.steps,
        decode="grammar_close", strict=True, seed=42,
        text=np.asarray(text[:n], dtype=np.float32),
        text_mask=np.asarray(text_mask[:n]).astype(bool),
        length_prior=lengths, length_exact=lengths is not None,
    )
    sink: list = []
    sample_smiles(model, path, tokenizer, params, device, latent_sink=sink)

    # sample_smiles strips the channel off before the readout, so recover it from the raw
    # latent it stored. The sink holds (latent, logits) and the latent is already narrowed;
    # re-run the split here would be wrong, so take the channel from the stored width.
    latents = torch.cat([x for x, _ in sink]).float().numpy()
    emb_dim = int(model.config.emb_dim)
    if latents.shape[-1] <= emb_dim:
        raise SystemExit(
            "the stored latent has no content channel; sample_smiles narrowed it before "
            "the sink. Store the latent before the split to use this script."
        )
    channel = latents[..., emb_dim]

    rules = [
        report("first negative (what the sampler uses)",
               first_negative(channel, canvas), true_lengths),
        report("first run of 2 negatives", first_run(channel, canvas, 2), true_lengths),
        report("first run of 3 negatives", first_run(channel, canvas, 3), true_lengths),
        report("first run of 5 negatives", first_run(channel, canvas, 5), true_lengths),
        report("one past the last positive", last_positive(channel, canvas), true_lengths),
    ]

    print(f"{args.checkpoint}")
    print(f"  {n} molecules, canvas {canvas}, "
          f"length prior: {args.length_file or 'none'}\n")
    print("  Channel value by position relative to the true end of the molecule:")
    for offset in (-5, -3, -1, 0, 1, 3, 5, 10):
        take = true_lengths + offset
        ok = (take >= 0) & (take < canvas)
        if not ok.any():
            continue
        values = channel[np.arange(len(channel))[ok], take[ok]]
        print(f"    {offset:+3d}: mean {values.mean():+8.4f}   "
              f"negative for {float((values < 0).mean()) * 100:5.1f}%")
    print(f"\n  Channel scale: mean |value| {np.abs(channel).mean():.4f}, "
          f"sd {channel.std():.4f}")
    print(f"  Share of all positions called content: {(channel > 0).mean() * 100:.1f}% "
          f"(true share {float((np.arange(canvas)[None, :] < true_lengths[:, None]).mean()) * 100:.1f}%)\n")
    print("  What each decision rule would have given:")
    print("    %-38s%8s%10s%8s%11s" % ("rule", "exact", "within 3", "MAE", "too early"))
    for r in rules:
        print("    %-38s%7.1f%%%9.1f%%%8.2f%10.1f%%"
              % (r["rule"], r["exact"] * 100, r["within_3"] * 100, r["mae"],
                 r["too_early"] * 100))

    if args.out:
        args.out.write_text(json.dumps(
            {"checkpoint": str(args.checkpoint), "rules": rules,
             "content_share": float((channel > 0).mean()),
             "channel_abs_mean": float(np.abs(channel).mean())}, indent=2))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
