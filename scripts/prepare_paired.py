#!/usr/bin/env python3
"""Prepare a molecule-and-caption corpus, keeping every pair aligned by row.

Two stages, because the work is split across two machines: canonicalising SMILES needs
rdkit, which only the login node has, and encoding captions wants a GPU, which only the
job has.

    # where rdkit is
    python scripts/prepare_paired.py --stage molecules --config configs/chebi20.yaml
    # where the GPU is
    python scripts/prepare_paired.py --stage text --config configs/chebi20.yaml

Unlike scripts/prepare_data.py this does not deduplicate or reshuffle: a benchmark's
splits are part of the benchmark, and a caption is meaningless once separated from its
molecule. A pair whose SMILES will not parse or will not fit the canvas is dropped whole,
and the count is recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pairs(source: dict, split: str) -> tuple[list[str], list[str]]:
    """(smiles, captions) for one split, in the order the dataset gives them."""
    from datasets import load_dataset

    data = load_dataset(source["hf_repo"])
    upstream = source.get("splits", {}).get(split, split)
    if upstream not in data:
        raise KeyError(f"{source['hf_repo']} has no split {upstream!r}; has {list(data)}")
    rows = data[upstream]
    smiles_col = source.get("smiles_column", "SMILES")
    text_col = source.get("text_column", "description")
    return [r[smiles_col] for r in rows], [r[text_col] for r in rows]


def stage_molecules(cfg) -> None:
    from rdkit import Chem, RDLogger

    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer

    RDLogger.DisableLog("rdApp.*")
    prep = OmegaConf.to_container(cfg.prepare, resolve=True)
    out_dir = Path(prep["tokenized_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = int(prep["max_length"])
    tokenizer = SmilesTokenizer.load(cfg.tokenizer["path"])
    print(f"tokenizer {cfg.tokenizer['path']} with {len(tokenizer.get_vocab())} tokens, "
          f"canvas {canvas}")

    report = {"tokenizer": cfg.tokenizer["path"], "max_length": canvas, "splits": {}}
    for split in prep["source"]["splits"]:
        smiles, captions = load_pairs(prep["source"], split)
        tokens = np.zeros((len(smiles), canvas), dtype=np.uint16)
        masks = np.zeros((len(smiles), canvas), dtype=np.uint8)
        kept_captions: list[str] = []
        kept = dropped_unparsable = dropped_long = 0
        for text, caption in zip(smiles, captions):
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                dropped_unparsable += 1
                continue
            ids = tokenizer.encode(Chem.MolToSmiles(mol))
            if len(ids) > canvas:
                dropped_long += 1
                continue
            tokens[kept, : len(ids)] = np.asarray(ids, dtype=np.uint16)
            masks[kept, : len(ids)] = 1
            kept_captions.append(caption)
            kept += 1
        np.save(out_dir / f"{split}_tokens_00000.npy", tokens[:kept])
        np.save(out_dir / f"{split}_attn_mask_00000.npy", masks[:kept])
        (out_dir / f"{split}_captions.txt").write_text(
            "\n".join(c.replace("\n", " ") for c in kept_captions) + "\n")
        lengths = masks[:kept].sum(1)
        report["splits"][split] = {
            "pairs": len(smiles), "kept": kept,
            "dropped_unparsable": dropped_unparsable, "dropped_too_long": dropped_long,
            "tokens_per_molecule": {"mean": round(float(lengths.mean()), 2),
                                    "p95": int(np.percentile(lengths, 95)),
                                    "max": int(lengths.max())},
        }
        print(f"  {split}: {len(smiles)} pairs -> {kept} kept "
              f"({dropped_unparsable} unparsable, {dropped_long} too long), "
              f"{lengths.mean():.1f} tokens each")
    (out_dir / "meta.json").write_text(json.dumps(report, indent=2))
    print(f"written to {out_dir}")


def stage_text(cfg) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    prep = OmegaConf.to_container(cfg.prepare, resolve=True)
    out_dir = Path(prep["tokenized_dir"])
    encoder_path = prep["text_encoder"]
    text_len = int(prep["text_length"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = AutoModel.from_pretrained(encoder_path).to(device).eval()
    hidden = model.config.hidden_size
    print(f"{encoder_path}: hidden {hidden}, on {device}, caption length {text_len}")

    meta_path = out_dir / "meta.json"
    report = json.loads(meta_path.read_text())
    for split in report["splits"]:
        captions = (out_dir / f"{split}_captions.txt").read_text().splitlines()
        embeddings = np.zeros((len(captions), text_len, hidden), dtype=np.float16)
        masks = np.zeros((len(captions), text_len), dtype=np.uint8)
        batch = 64
        with torch.no_grad():
            for start in range(0, len(captions), batch):
                chunk = captions[start : start + batch]
                encoded = tokenizer(chunk, padding="max_length", truncation=True,
                                    max_length=text_len, return_tensors="pt").to(device)
                out = model(**encoded).last_hidden_state
                embeddings[start : start + len(chunk)] = out.cpu().numpy().astype(np.float16)
                masks[start : start + len(chunk)] = encoded["attention_mask"].cpu().numpy()
        np.save(out_dir / f"{split}_text.npy", embeddings)
        np.save(out_dir / f"{split}_text_mask.npy", masks)
        report["splits"][split]["text_tokens_mean"] = round(float(masks.sum(1).mean()), 2)
        print(f"  {split}: {len(captions)} captions -> {embeddings.shape}, "
              f"{masks.sum(1).mean():.1f} tokens each")
    report["text_encoder"] = encoder_path
    report["text_length"] = text_len
    report["text_hidden"] = hidden
    meta_path.write_text(json.dumps(report, indent=2))
    print(f"written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("molecules", "text"))
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    (stage_molecules if args.stage == "molecules" else stage_text)(cfg)


if __name__ == "__main__":
    main()
