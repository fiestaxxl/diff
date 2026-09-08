#!/usr/bin/env python3
"""Grow a checkpoint's vocabulary and canvas so it can be fine-tuned on a second corpus.

    python scripts/grow_vocab.py --checkpoint runs/c48m_s43/ep0-ba34375 \
        --vocab-size 640 --max-pos 128 --out runs/c48m_s43_grown

The tokenizer extension keeps every existing id, so rows 0..447 of the embedding table
and of the readout still mean exactly what they meant. This appends rows for the new
tokens and leaves everything else untouched, which is what makes a warm start possible
instead of pretraining again from scratch.

New rows are drawn at the scale of the trained ones, not at initialisation scale: the
embedding table grows about fiftyfold over a run, so a fresh row at std 0.02 would sit
near the origin and be unreachable. The readout bias for new tokens starts at the minimum
of the trained biases, because a token never seen should begin unlikely rather than
average.

RoPE needs no surgery: it registers its cos and sin as non-persistent buffers, rebuilt
from max_pos when the model is constructed. The length-conditioning embedding does need
it, because it has one row per possible length. Its new rows start as copies of the
longest length the model was trained on, so a molecule of 130 tokens is treated like one
of 63 until fine-tuning says otherwise, which is closer to right than noise.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

GROW_ROWS = ("token_embedding.weight", "out_proj.weight")
GROW_BIAS = ("out_proj.bias",)


def load_state(directory: Path) -> dict:
    safetensors = directory / "model.safetensors"
    if safetensors.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors))
    return torch.load(directory / "model.pt", map_location="cpu", weights_only=True)


def save_state(state: dict, directory: Path) -> None:
    try:
        from safetensors.torch import save_file

        save_file({k: v.contiguous() for k, v in state.items()},
                  str(directory / "model.safetensors"))
    except ImportError:
        torch.save(state, directory / "model.pt")


def grow_length_table(state: dict, max_pos: int) -> dict:
    """One row per length: extend it by repeating the longest trained row."""
    key = "length_embedding.weight"
    if key not in state or max_pos is None:
        return state
    out = dict(state)
    old = out[key]
    want = max_pos + 1
    if old.shape[0] >= want:
        return out
    extra = want - old.shape[0]
    out[key] = torch.cat([old, old[-1:].expand(extra, -1).clone()], dim=0)
    print(f"  {key}: {tuple(old.shape)} -> {tuple(out[key].shape)}, "
          f"{extra} rows copied from length {old.shape[0] - 1}")
    return out


def grow(state: dict, vocab_size: int) -> tuple[dict, int]:
    out = dict(state)
    added = 0
    for key in GROW_ROWS:
        if key not in out:
            continue
        old = out[key]
        if old.shape[0] >= vocab_size:
            continue
        extra = vocab_size - old.shape[0]
        std = old.float().std().item()
        new = torch.randn(extra, old.shape[1], dtype=old.dtype) * std
        out[key] = torch.cat([old, new], dim=0)
        added = extra
        print(f"  {key}: {tuple(old.shape)} -> {tuple(out[key].shape)}, "
              f"{extra} rows at std {std:.4f}")
    for key in GROW_BIAS:
        if key not in out:
            continue
        old = out[key]
        if old.shape[0] >= vocab_size:
            continue
        extra = vocab_size - old.shape[0]
        floor = old.float().min().item()
        new = torch.full((extra,), floor, dtype=old.dtype)
        out[key] = torch.cat([old, new], dim=0)
        print(f"  {key}: {tuple(old.shape)} -> {tuple(out[key].shape)}, "
              f"new biases at {floor:.4f}, the minimum of the trained ones")
    return out, added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--max-pos", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = json.loads((args.checkpoint / "config.json").read_text())
    print(f"{args.checkpoint}: vocab {config['vocab_size']}, max_pos {config['max_pos']}")

    state = load_state(args.checkpoint)
    grown, added = grow(state, args.vocab_size)
    grown = grow_length_table(grown, args.max_pos)
    if not added:
        print("nothing to grow; the checkpoint is already at least this wide")

    # the rows that were already trained must come through untouched
    for key in GROW_ROWS + GROW_BIAS + ("length_embedding.weight",):
        if key in state:
            before = state[key]
            assert torch.equal(grown[key][: before.shape[0]], before), key
    print("every trained row is unchanged")

    config["vocab_size"] = args.vocab_size
    if args.max_pos is not None:
        config["max_pos"] = args.max_pos
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    save_state(grown, args.out)
    for extra in ("meta.json",):
        source = args.checkpoint / extra
        if source.exists():
            shutil.copy(source, args.out / extra)
    print(f"written to {args.out}: vocab {config['vocab_size']}, "
          f"max_pos {config['max_pos']}")


if __name__ == "__main__":
    main()
