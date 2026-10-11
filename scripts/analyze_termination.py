#!/usr/bin/env python3
"""Anatomy of the termination failure: what is in the latent past the end of the molecule?

    python scripts/analyze_termination.py --config configs/chebi20_finetune.yaml \
        --checkpoint runs/cepad_s43/ep2020-ba200000 --split test --limit 500

Padding supervision taught the readout to name padding perfectly — cross-entropy on the
padding class fell to 0.0004 — and the model still cannot stop on its own: given no length,
MACCS falls 0.756 to 0.599 and Levenshtein rises 21.6 to 50.3. So the failure is in the
latent the sampler produces, not in the classifier that reads it. This measures what is
actually there.

For every canvas position it asks three questions:

* **Is the latent near the padding embedding?** Distance to `emb[pad]` against the distance
  to the nearest real token. If the tail latents sit close to padding but lose the argmax,
  the problem is a decision boundary. If they sit far from it, the model genuinely does not
  produce padding and no readout could recover it.
* **Where does the model imply the molecule ends?** The first position whose argmax is the
  padding token, against the true length. A model that stops too early, too late, or not at
  all fails in three different ways that call for three different fixes.
* **Is the transition sharp?** How the padding margin behaves across the true boundary. A
  smeared transition means the model has a soft, position-local notion of ending rather
  than a global one — which is the case a content channel would address.

Run it with and without a length prior: the prior pins the tail to padding by construction,
so the difference between the two is exactly what the model contributes on its own.
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


def percentiles(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    q = np.percentile(values, [10, 50, 90])
    return f"p10 {q[0]:+.2f}  p50 {q[1]:+.2f}  p90 {q[2]:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--length-file", type=Path, default=None,
                        help="pin the tail to padding with these lengths. Omit to see "
                             "what the model does unaided, which is the point")
    parser.add_argument("--no-pin", action="store_true",
                        help="give the model the length as an input but leave the canvas "
                             "alone. With pinning on, the tail is padding by construction "
                             "and every tail statistic below is a tautology")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer(cfg)
    model = DiffusionTransformer.from_pretrained(load_dir=str(args.checkpoint),
                                                 map_location=device)
    model.eval()
    canvas_len = int(cfg.variables.seq_len)
    path = build_path(cfg, seq_len=canvas_len, emb_dim=getattr(model, "latent_dim", model.config.emb_dim),
                      device=device)

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
        pin_canvas=not args.no_pin,
    )
    sink: list = []
    sample_smiles(model, path, tokenizer, params, device, latent_sink=sink)

    latents = torch.cat([x for x, _ in sink]).float()          # (N, T, emb)
    logits = torch.cat([g for _, g in sink]).float()           # (N, T, V)
    table = model.token_embedding.weight.detach().cpu().float()  # (V, emb)
    pad_idx = int(getattr(model.config, "pad_idx", 0))

    # Distance to the padding embedding, and to the nearest embedding that is not padding.
    distances = torch.cdist(latents.reshape(-1, latents.shape[-1]), table)  # (N*T, V)
    to_pad = distances[:, pad_idx]
    other = distances.clone()
    other[:, pad_idx] = float("inf")
    to_nearest_real, nearest_real_id = other.min(dim=-1)
    argmax_id = logits.argmax(-1).reshape(-1)

    shape = latents.shape[:2]
    margin = (to_nearest_real - to_pad).reshape(shape).numpy()   # >0 means padding is closer
    says_pad = (argmax_id == pad_idx).reshape(shape).numpy()

    canvas = shape[1]
    positions = np.arange(canvas)[None, :]
    is_tail = positions >= true_lengths[:, None]                 # should be padding
    is_body = ~is_tail

    # Where does the model itself say the molecule ends?
    implied = np.where(says_pad.any(1), says_pad.argmax(1), canvas)
    error = implied - true_lengths

    report = {
        "checkpoint": str(args.checkpoint),
        "length_prior": str(args.length_file) if args.length_file else None,
        "pin_canvas": not args.no_pin,
        "molecules": int(shape[0]),
        "canvas": int(canvas),
        "tail_positions_where_padding_is_nearest": float(says_pad[is_tail].mean()),
        "body_positions_wrongly_called_padding": float(says_pad[is_body].mean()),
        "tail_margin_mean": float(margin[is_tail].mean()),
        "body_margin_mean": float(margin[is_body].mean()),
        "implied_end_never_reached": float((implied == canvas).mean()),
        "implied_end_error_mean_abs": float(np.abs(error).mean()),
        "implied_end_error_median": float(np.median(error)),
        "stops_too_early": float((error < 0).mean()),
        "stops_too_late": float((error > 0).mean()),
        "stops_exactly": float((error == 0).mean()),
    }

    print(f"{args.checkpoint}")
    print(f"  length prior: {report['length_prior'] or 'drawn from the corpus'}"
          f"   canvas pinning: {'on' if report['pin_canvas'] else 'OFF'}")
    print(f"  {report['molecules']} molecules, canvas {canvas}\n")
    print("  Does the latent reach padding where it should?")
    print(f"    tail positions whose argmax is padding      "
          f"{report['tail_positions_where_padding_is_nearest'] * 100:6.1f}%")
    print(f"    body positions wrongly called padding       "
          f"{report['body_positions_wrongly_called_padding'] * 100:6.1f}%")
    print(f"    margin (dist to nearest real - dist to pad), positive means pad is closer")
    print(f"      tail: {percentiles(margin[is_tail])}")
    print(f"      body: {percentiles(margin[is_body])}\n")
    print("  Where does the model imply the molecule ends?")
    print(f"    never emits padding at all                  "
          f"{report['implied_end_never_reached'] * 100:6.1f}%")
    print(f"    exactly right                               "
          f"{report['stops_exactly'] * 100:6.1f}%")
    print(f"    too early / too late                        "
          f"{report['stops_too_early'] * 100:6.1f}% / "
          f"{report['stops_too_late'] * 100:6.1f}%")
    print(f"    mean |error|                                "
          f"{report['implied_end_error_mean_abs']:6.2f} tokens\n")
    print("  Margin profile around the true boundary (offset from the last content token):")
    for offset in (-3, -2, -1, 0, 1, 2, 3, 5, 10):
        take = true_lengths + offset
        valid = (take >= 0) & (take < canvas)
        if not valid.any():
            continue
        rows = np.arange(shape[0])[valid]
        values = margin[rows, take[valid]]
        print(f"    {offset:+3d}: margin {values.mean():+7.3f}   "
              f"padding is nearest for {float((values > 0).mean()) * 100:5.1f}%")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
