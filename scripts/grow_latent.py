#!/usr/bin/env python3
"""Grow a checkpoint's latent so it can be fine-tuned with model.content_channel on.

    python scripts/grow_latent.py --checkpoint runs/c48m_grown --out runs/c48m_content

The content channel widens the latent by one, which widens the only tensor that reads it:
``up_proj.weight``. Everything else is untouched — the embedding table, the readout, the
vocabulary and the noise head all still work in embedding space, because the channel is
carried beside the embedding rather than inside it.

The insertion is not an append. With self-conditioning the input is laid out as

    [ x_t (emb_dim) | x0_self (emb_dim) ]

and becomes

    [ x_t (emb_dim) | content (1) | x0_self (emb_dim) | content_self (1) ]

so a zero column goes in at ``emb_dim`` and again at ``2 * emb_dim + 1``. Zero, so the
grown checkpoint ignores the channel on its first step and behaves exactly as it did
before; gradients reach the new columns immediately, so it does not stay ignored.

``content_head`` is absent from the source checkpoint and is left to the model's own
initialisation, which zeroes it for the same reason.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.models.diffusion_transformer import DiffusionTransformer  # noqa: E402

CONFIG_NAME = DiffusionTransformer.CONFIG_NAME
WEIGHTS_NAME = DiffusionTransformer.WEIGHTS_NAME


def load_state(directory: Path) -> dict:
    weights = directory / WEIGHTS_NAME
    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weights), device="cpu")
    return torch.load(weights, map_location="cpu")


def save_state(state: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHTS_NAME
    if target.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(target))
    else:
        torch.save(state, target)


def grow(state: dict, emb_dim: int, self_conditioning: bool) -> dict:
    key = "up_proj.weight"
    if key not in state:
        raise KeyError(f"{key} is not in the checkpoint; is this a diffusion transformer?")
    weight = state[key]
    expected = emb_dim * (2 if self_conditioning else 1)
    if weight.shape[1] == expected + (2 if self_conditioning else 1):
        print("  already grown, leaving it alone")
        return state
    if weight.shape[1] != expected:
        raise ValueError(
            f"{key} has {weight.shape[1]} input columns; expected {expected} for "
            f"emb_dim={emb_dim} and self_conditioning={self_conditioning}"
        )

    zeros = torch.zeros(weight.shape[0], 1, dtype=weight.dtype)
    if self_conditioning:
        head, tail = weight[:, :emb_dim], weight[:, emb_dim:]
        grown = torch.cat((head, zeros, tail, zeros), dim=1)
    else:
        grown = torch.cat((weight, zeros), dim=1)
    state[key] = grown
    print(f"  {key}: {tuple(weight.shape)} -> {tuple(grown.shape)}, "
          f"{'2 zero columns' if self_conditioning else '1 zero column'} inserted")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.checkpoint / CONFIG_NAME
    config = json.loads(config_path.read_text())
    emb_dim = int(config["emb_dim"])
    self_conditioning = bool(config.get("self_conditioning", False))
    print(f"{args.checkpoint}: emb_dim {emb_dim}, "
          f"self_conditioning {self_conditioning}")

    state = grow(load_state(args.checkpoint), emb_dim, self_conditioning)
    args.out.mkdir(parents=True, exist_ok=True)
    save_state(state, args.out)

    # The config travels with the weights, with the flag set so the shape is explicable
    # from the checkpoint alone rather than only from the command that trained it.
    config["content_channel"] = True
    (args.out / CONFIG_NAME).write_text(json.dumps(config, indent=2))
    for extra in ("meta.json",):
        source = args.checkpoint / extra
        if source.exists():
            shutil.copy2(source, args.out / extra)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
