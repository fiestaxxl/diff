#!/usr/bin/env python3
"""Re-encode a prepared corpus's captions with a different encoder.

    # from a plain huggingface encoder directory
    python scripts/export_caption_states.py --encoder data/chebi20/length_encoder/encoder \
        --corpus data/chebi20/tokenized --out data/chebi20/tokenized_lenc

    # from the encoder a trained checkpoint carries
    python scripts/export_caption_states.py --checkpoint checkpoints/enc_win/ep12-ba6000 \
        --corpus data/chebi20/tokenized --out data/chebi20/tokenized_ft

The output is a drop-in corpus directory: the same molecule arrays, the same captions and
the same caption ids, with only ``<split>_text.npy`` replaced. That is what makes the
downstream comparison controlled — ``eval_text.py``, ``score_with_ar.py``,
``select_candidates.py``, ``oracle_ceiling.py`` and the analysis scripts all read
``cfg.variables.data_dir`` and need no change at all, and a fine-tune launched against
this directory differs from the measured baseline in the bytes of one array and nothing
else.

Two details are deliberate. The states are written as float16, the dtype
``prepare_paired.py`` already uses, so no new quantisation error enters the comparison.
And the encoder runs over the STORED ``<split>_caption_ids.npy`` rather than
re-tokenizing, which removes the whole class of silent mismatch that
``scripts/tokenize_captions.py`` asserts against.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Before torch: the thread pool size is read when the library is imported, and sr008 is a
# shared 176-core node where torch otherwise takes every core for its CPU side. A cap
# already in the environment wins, because it is the one the job was launched under.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Everything the new corpus inherits unchanged. The two arrays this script writes are
# excluded, and so is meta.json, which is rewritten with the provenance of the states.
REGENERATED = ("_text.npy", "_text_mask.npy")


def load_encoder_from_checkpoint(checkpoint: Path, device: str):
    """The encoder a trained denoiser carries, or a readable refusal."""
    from dimol.models.diffusion_transformer import DiffusionTransformer

    model = DiffusionTransformer.from_pretrained(load_dir=str(checkpoint),
                                                 map_location=device)
    encoder = getattr(model, "text_encoder", None)
    if encoder is None:
        raise ValueError(
            f"{checkpoint} holds no caption encoder (model.text_encoder is unset in its "
            "config.json), so there is nothing here to export; the states it was trained "
            "against are the ones already in the corpus"
        )
    return encoder


def copy_corpus_files(corpus: Path, out: Path) -> list[str]:
    """Everything the new corpus inherits: molecules, captions, caption ids, indices.

    Copies rather than symlinks. A symlink into another directory on NFS is one `rm -rf`
    of a scratch corpus away from a dangling reference, and the failure would surface as a
    training run that cannot read its own molecules.
    """
    copied = []
    for path in sorted(corpus.iterdir()):
        if not path.is_file() or path.name == "meta.json":
            continue
        if any(path.name.endswith(suffix) for suffix in REGENERATED):
            continue
        shutil.copy2(path, out / path.name)
        copied.append(path.name)
    return copied


def export_states(encoder, corpus: Path, out: Path, splits, device: str = "cpu",
                  batch: int = 64, provenance: dict | None = None) -> None:
    """Write ``<split>_text.npy`` from the stored caption ids, then fill the corpus out."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    encoder = encoder.to(device).eval()
    hidden = int(encoder.config.hidden_size)

    for split in splits:
        ids = np.load(corpus / f"{split}_caption_ids.npy")
        attn = np.load(corpus / f"{split}_caption_attn.npy")
        if ids.shape != attn.shape:
            raise ValueError(
                f"split {split!r}: caption ids are {ids.shape} but the mask is "
                f"{attn.shape}; re-run scripts/tokenize_captions.py"
            )
        states = np.zeros((len(ids), ids.shape[1], hidden), dtype=np.float16)
        with torch.no_grad():
            for start in range(0, len(ids), batch):
                stop = min(start + batch, len(ids))
                chunk = torch.from_numpy(ids[start:stop].astype(np.int64)).to(device)
                mask = torch.from_numpy(attn[start:stop].astype(np.int64)).to(device)
                out_states = encoder(input_ids=chunk,
                                     attention_mask=mask).last_hidden_state
                states[start:stop] = out_states.float().cpu().numpy().astype(np.float16)
        np.save(out / f"{split}_text.npy", states)
        np.save(out / f"{split}_text_mask.npy", attn.astype(np.uint8))
        print(f"  {split}: {len(ids)} captions -> {states.shape}")

    copied = copy_corpus_files(corpus, out)
    print(f"  carried over unchanged: {len(copied)} files")

    meta_path = corpus / "meta.json"
    report = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    report.update(provenance or {})
    report["text_hidden"] = hidden
    report["exported_from"] = str(corpus)
    (out / "meta.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="a prepared corpus with <split>_caption_ids.npy in it")
    parser.add_argument("--out", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder", type=Path,
                        help="a huggingface encoder directory, e.g. the one "
                             "train_length_encoder.py writes")
    source.add_argument("--checkpoint", type=Path,
                        help="a run directory whose model carries a caption encoder")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="default: every split in the corpus's meta.json")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = args.splits
    if splits is None:
        meta = json.loads((args.corpus / "meta.json").read_text())
        splits = list(meta["splits"])

    if args.encoder is not None:
        from transformers import AutoModel

        # The same call prepare_paired.stage_text makes, so pointing this at the corpus's
        # own encoder reproduces the stored states rather than approximating them.
        encoder = AutoModel.from_pretrained(str(args.encoder))
        provenance = {"text_encoder": str(args.encoder), "text_encoder_step": None}
    else:
        encoder = load_encoder_from_checkpoint(args.checkpoint, device)
        step = args.checkpoint.name.split("-ba")[-1] if "-ba" in args.checkpoint.name \
            else None
        provenance = {"text_encoder": str(args.checkpoint),
                      "text_encoder_step": int(step) if step and step.isdigit() else None}

    print(f"{provenance['text_encoder']}: {args.corpus} -> {args.out}, on {device}")
    export_states(encoder, args.corpus, args.out, splits, device=device,
                  batch=args.batch_size, provenance=provenance)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
