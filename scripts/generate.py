#!/usr/bin/env python3
"""Generate from a checkpoint and report quality metrics.

    python scripts/generate.py configs/generate_chebi.yaml
    python scripts/generate.py configs/generate_chebi.yaml generate.checkpoint=checkpoints/v12/ep499-ba12000

The sampling scheme and the metrics are the same as in the old generate.py; the
parameters now come from yaml instead of environment variables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_path, build_tokenizer  # noqa: E402
from dimol.config import load_from_argv, log_config  # noqa: E402
from dimol.eval.report import evaluate_smiles, format_report, load_train_canon  # noqa: E402
from dimol.eval.sampling import SamplingParams, sample_smiles  # noqa: E402
from dimol.models.diffusion_transformer import DiffusionTransformer  # noqa: E402
from dimol.training.distributed import seed_all  # noqa: E402


def _allowed_brackets(gen: dict):
    """The bracket atoms the training corpus uses, when the config asks for the restriction.

    Off by default: it changes what the decoder may emit, so it has to be requested.
    """
    if not gen.get("restrict_atoms"):
        return None
    from dimol.eval.decoding import corpus_brackets

    source = gen.get("atom_vocab_path") or gen.get("train_smiles_path")
    if not source:
        raise KeyError("generate.restrict_atoms needs generate.train_smiles_path")
    found = frozenset(corpus_brackets(source, limit=gen.get("atom_vocab_limit")))
    print(f"restricting bracket atoms to the {len(found)} the corpus uses")
    return found


def main(cfg: DictConfig) -> None:
    gen = OmegaConf.to_container(cfg.generate, resolve=True) or {}
    checkpoint = gen.get("checkpoint")
    if not checkpoint:
        raise KeyError("generate.checkpoint is not set")

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(gen.get("seed", 42))
    seed_all(seed, tf32=bool(cfg.tf32))
    log_config(cfg)

    model = DiffusionTransformer.from_pretrained(load_dir=checkpoint, map_location=device)
    model.eval()
    print(f"loaded model from {checkpoint}")

    tokenizer = build_tokenizer(cfg)

    seq_len = int(gen.get("seq_len") or cfg.variables.get("seq_len", None))
    path = build_path(cfg, seq_len=seq_len, emb_dim=model.config.emb_dim, device=device)

    params = SamplingParams(
        num_samples=int(gen.get("num_samples") or 2000),
        num_timesteps=int(gen.get("num_timesteps") or 300),
        variance=float(gen.get("variance", 1.0)),
        t_start=float(gen.get("t_start", 1e-3)),
        t_end=float(gen.get("t_end", 0.999)),
        seed=seed,
        batch_size=int(gen.get("batch_size") or 500),
        regime=str(gen.get("regime") or cfg.diffusion.get("regime", "epsilon")),
        progress=bool(gen.get("progress", True)),
        clamp_strength=float(gen.get("clamp_strength", 0.0)),
        clamp_from_alpha=float(gen.get("clamp_from_alpha", 0.5)),
        decode=str(gen.get("decode", "argmax")),
        allowed_brackets=_allowed_brackets(gen),
        on_disallowed=str(gen.get("on_disallowed", "next_best")),
        strict=bool(gen.get("strict_decode", False)),
        time_grid=str(gen.get("time_grid", "uniform")),
        time_grid_power=float(gen.get("time_grid_power", 2.0)),
    )

    print("generating...")
    # as in the old generate(): the seed is reset BEFORE integrating the SDE, otherwise
    # the per-step noise depends on how much RNG the model loading consumed
    seed_all(seed)
    smiles = sample_smiles(model, path, tokenizer, params, device)

    out_dir = Path(gen.get("output_dir") or checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.txt").write_text("\n".join(smiles) + "\n")
    print(f"{len(smiles)} samples written to {out_dir / 'samples.txt'}")

    if not bool(gen.get("report", True)):
        # Sampling needs the GPU, scoring needs rdkit; when they live in different
        # places, generate here and run scripts/evaluate_samples.py there.
        print("generate.report=false: skipping the quality report")
        return

    train_canon = load_train_canon(gen.get("train_smiles_path"))
    metrics = evaluate_smiles(smiles, train_canon=train_canon)

    report = format_report(metrics)
    print("\n" + report + "\n")
    print(f"First {min(20, len(smiles))} generated (truncated at <eos>):")
    for s in smiles[:20]:
        print(f"  {s!r}")

    (out_dir / "report.txt").write_text(
        report + "\n\n" + "\n".join(f"  {s!r}" for s in smiles[:20]) + "\n"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"report and samples written to {out_dir}")


if __name__ == "__main__":
    main(load_from_argv())
