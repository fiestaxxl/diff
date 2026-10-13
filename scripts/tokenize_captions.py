#!/usr/bin/env python3
"""Store the caption token ids beside the caption states they produced.

    python scripts/tokenize_captions.py --config configs/chebi20.yaml

``scripts/prepare_paired.py --stage text`` tokenizes every caption, runs SciBERT over it
and keeps only the output states. Putting the encoder inside the training loop needs the
ids themselves, so this writes them next to the states: ``<split>_caption_ids.npy`` and
``<split>_caption_attn.npy``, about 12 MB for all three splits against 5 GB for the train
states alone. Nothing is re-encoded and ``<split>_text.npy`` is never opened.

The guarantee that these ids are the *same* tokenization that produced the stored states
is an assertion, not a convention. Per split the attention mask this tokenizer produces
must be bit-identical to the stored ``<split>_text_mask.npy``, and the row count must
match the molecule arrays. A tokenizer, a caption file or a ``text_length`` that moved
since the states were written therefore fails here, in seconds, rather than training for
six hours against a caption the model's states never saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_alignment(out_dir: Path, split: str, attn: np.ndarray) -> None:
    """Refuse to write ids that disagree with the stored states' own mask.

    The mask is the only part of the stored encoding that survives in a form this script
    can reproduce, which makes it the whole alignment proof: two tokenizations agreeing on
    where every caption ends, token for token and row for row, are the same tokenization.
    """
    stored_path = out_dir / f"{split}_text_mask.npy"
    if not stored_path.exists():
        raise FileNotFoundError(
            f"{stored_path} is missing; run scripts/prepare_paired.py --stage text first, "
            "so there is a stored encoding to check these ids against"
        )
    stored = np.load(stored_path)
    if stored.shape != attn.shape:
        raise ValueError(
            f"split {split!r}: the stored text mask is {stored.shape} but this "
            f"tokenization is {attn.shape}; prepare.text_length or the caption file has "
            "moved since the states were written"
        )
    if not np.array_equal(stored.astype(np.uint8), attn):
        differing = int((stored.astype(np.uint8) != attn).any(1).sum())
        raise ValueError(
            f"split {split!r}: {differing} captions tokenize differently than the stored "
            f"{stored_path.name}; these ids are not the tokenization that produced the "
            "states, and training on them would condition on a different caption"
        )
    tokens_path = out_dir / f"{split}_tokens_00000.npy"
    rows = len(np.load(tokens_path, mmap_mode="r"))
    if rows != len(attn):
        raise ValueError(
            f"split {split!r}: {rows} molecules in {tokens_path.name} but {len(attn)} "
            "captions; the arrays in this directory are no longer row-aligned"
        )


def write_caption_ids(tokenizer, out_dir: Path, splits, text_len: int,
                      batch: int = 512) -> dict:
    """Tokenize ``<split>_captions.txt`` and write the ids and the mask.

    ``tokenizer`` is called exactly as ``prepare_paired.stage_text`` calls it, and only
    the container differs: numpy rather than torch, so preparing the ids needs no GPU and
    no torch import at all.
    """
    report = {}
    for split in splits:
        captions = (out_dir / f"{split}_captions.txt").read_text().splitlines()
        ids = np.zeros((len(captions), text_len), dtype=np.uint16)
        attn = np.zeros((len(captions), text_len), dtype=np.uint8)
        for start in range(0, len(captions), batch):
            chunk = captions[start : start + batch]
            encoded = tokenizer(chunk, padding="max_length", truncation=True,
                                max_length=text_len, return_tensors="np")
            block = np.asarray(encoded["input_ids"])
            largest = int(block.max()) if block.size else 0
            if largest > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"token id {largest} does not fit in uint16; this storage assumes an "
                    "encoder vocabulary below 65,536 (SciBERT has 31,090)"
                )
            ids[start : start + len(chunk)] = block.astype(np.uint16)
            attn[start : start + len(chunk)] = np.asarray(
                encoded["attention_mask"]).astype(np.uint8)

        check_alignment(out_dir, split, attn)
        np.save(out_dir / f"{split}_caption_ids.npy", ids)
        np.save(out_dir / f"{split}_caption_attn.npy", attn)
        report[split] = {"captions": len(captions), "caption_tokens": text_len,
                         "caption_tokens_mean": round(float(attn.sum(1).mean()), 2)
                         if len(captions) else 0.0}
        print(f"  {split}: {len(captions)} captions -> {ids.shape} uint16, "
              f"{attn.sum(1).mean() if len(captions) else 0:.1f} tokens each")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    cfg = OmegaConf.load(args.config)
    prep = OmegaConf.to_container(cfg.prepare, resolve=True)
    out_dir = Path(prep["tokenized_dir"])
    encoder_path = prep["text_encoder"]
    text_len = int(prep["text_length"])

    meta_path = out_dir / "meta.json"
    report = json.loads(meta_path.read_text())
    splits = list(report["splits"])
    print(f"{encoder_path}: caption length {text_len}, splits {splits}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    written = write_caption_ids(tokenizer, out_dir, splits, text_len)
    for split, stats in written.items():
        report["splits"][split].update(stats)
    report["caption_ids_encoder"] = encoder_path
    meta_path.write_text(json.dumps(report, indent=2))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
