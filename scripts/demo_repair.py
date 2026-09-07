#!/usr/bin/env python3
"""Show what grammar repair actually does, case by case.

    python scripts/demo_repair.py configs/generate_zinc250k.yaml \
        generate.checkpoint=runs/r_tl0_ce3/ep80-ba17600 generate.num_samples=512

Sampling needs a GPU and rdkit is not installed next to one here, so the script runs in
two phases around a file of decodings: `generate.decoded_tsv=<path>` writes it where the
GPU is if it does not exist yet, and reads it and does the chemistry where rdkit is.

One batch is sampled once and the same logits are decoded four ways, so the only
difference between the columns is the decoding rule. For the cases where the plain argmax
string is invalid and a repaired one is valid, the script prints both strings, the reason
argmax failed, and what the repair changed. It ends with a table of accepted molecules and
their descriptors, so the output can be read as evidence rather than a claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_path, build_tokenizer  # noqa: E402
from dimol.config import load_from_argv  # noqa: E402
from dimol.diffusion.diff_eqs import LearnedScoreSDE  # noqa: E402
from dimol.diffusion.simulators import EulerMaruyamaSimulator  # noqa: E402
from dimol.eval.decoding import build_decoder, _token_grammar  # noqa: E402
from dimol.eval.sampling import SamplingParams, _time_grid  # noqa: E402
from dimol.models.denoiser import DenoiserModel  # noqa: E402
from dimol.models.diffusion_transformer import DiffusionTransformer  # noqa: E402
from dimol.training.distributed import seed_all  # noqa: E402

MODES = ("argmax", "grammar_close", "grammar_trim", "grammar_mixed")


def grammar_state(smiles: str):
    """Running parenthesis depth and ring-digit parity of a finished string."""
    depth, lowest = 0, 0
    parity = [0] * 10
    inside_bracket = False
    for ch in smiles:
        if ch == "[":
            inside_bracket = True
        elif ch == "]":
            inside_bracket = False
        elif inside_bracket:
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            lowest = min(lowest, depth)
        elif ch.isdigit():
            parity[int(ch)] ^= 1
    return depth, lowest, [d for d, p in enumerate(parity) if p]


def why_invalid(smiles: str) -> str:
    from rdkit import Chem

    if not smiles:
        return "empty string"
    depth, lowest, odd_rings = grammar_state(smiles)
    reasons = []
    if lowest < 0:
        reasons.append("closes a branch that was never opened")
    if depth > 0:
        reasons.append(f"{depth} branch{'es' if depth > 1 else ''} left open")
    if odd_rings:
        reasons.append("unpaired ring digit " + ",".join(str(d) for d in odd_rings))
    if reasons:
        return "; ".join(reasons)
    if Chem.MolFromSmiles(smiles) is None:
        return "grammar is fine, chemistry is not (valence or aromaticity)"
    return "valid"


def describe_change(before: str, after: str) -> str:
    if after.startswith(before):
        return f"appended {after[len(before):]!r}"
    if before.startswith(after):
        return f"trimmed {before[len(after):]!r} from the end"
    common = 0
    for a, b in zip(before, after):
        if a != b:
            break
        common += 1
    return f"kept {common} chars, then {before[common:]!r} -> {after[common:]!r}"


def main(cfg: DictConfig) -> None:
    gen = OmegaConf.to_container(cfg.generate, resolve=True) or {}
    checkpoint = gen.get("checkpoint")
    if not checkpoint:
        raise KeyError("generate.checkpoint is not set")
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(gen.get("seed", 42))
    seed_all(seed, tf32=bool(cfg.tf32))

    tsv = gen.get("decoded_tsv")
    tsv_path = Path(tsv) if tsv else None
    if tsv_path is not None and tsv_path.exists():
        rows = [line.rstrip("\n").split("\t") for line in tsv_path.read_text().splitlines()]
        decoded = {mode: [row[i] if i < len(row) else "" for row in rows]
                   for i, mode in enumerate(MODES)}
        print(f"read {len(rows)} decoded samples from {tsv_path}\n")
        analyse(decoded, len(rows), int(gen.get("show", 8) or 8),
                int(gen.get("show_valid", 15) or 15))
        return

    model = DiffusionTransformer.from_pretrained(load_dir=checkpoint, map_location=device)
    model.eval()
    tokenizer = build_tokenizer(cfg)
    seq_len = int(gen.get("seq_len") or cfg.variables.get("seq_len", None))
    path = build_path(cfg, seq_len=seq_len, emb_dim=model.config.emb_dim, device=device)

    params = SamplingParams(
        num_samples=int(gen.get("num_samples") or 512),
        num_timesteps=int(gen.get("num_timesteps") or 100),
        variance=float(gen.get("variance", 1.0)),
        t_start=float(gen.get("t_start", 1e-3)),
        t_end=float(gen.get("t_end", 0.999)),
        seed=seed,
        regime=str(gen.get("regime") or cfg.diffusion.get("regime", "epsilon")),
    )

    print(f"model {checkpoint}, {params.num_samples} samples, "
          f"{params.num_timesteps} solver steps, seed {seed}\n")

    with torch.no_grad():
        x = path.p_simple.sample(params.num_samples, seed=params.seed)
        ts = (_time_grid(params).view(1, params.num_timesteps, 1, 1)
              .expand(params.num_samples, -1, -1, -1).to(device))
        sde = LearnedScoreSDE(path, DenoiserModel(model, path, regime=params.regime),
                              params.variance)
        xt = EulerMaruyamaSimulator(sde).simulate(x, ts, use_bar=False)
        logits = model.out_proj(xt).detach()

    decoded = {}
    for mode in MODES:
        decoder = build_decoder(tokenizer, canvas=path.p_simple.shape[0], mode=mode)
        if decoder is None:
            ids = logits.softmax(-1).argmax(-1).cpu().tolist()
            decoded[mode] = tokenizer.decode_batch(ids, special_decode=True)
        else:
            decoded[mode] = decoder.decode(logits)

    if tsv_path is not None:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text("\n".join(
            "\t".join(decoded[mode][i].replace("\t", " ") for mode in MODES)
            for i in range(params.num_samples)
        ) + "\n")
        print(f"wrote {params.num_samples} decoded samples to {tsv_path}")
        print("run the same command where rdkit is installed to see the analysis")
        return

    analyse(decoded, params.num_samples, int(gen.get("show", 8) or 8),
            int(gen.get("show_valid", 15) or 15))


def analyse(decoded, n: int, show: int, show_valid: int) -> None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    ok = {m: [Chem.MolFromSmiles(s) is not None if s else False for s in decoded[m]]
          for m in MODES}

    print(f"{'decoding':<16}{'valid':>8}{'rate':>9}")
    for mode in MODES:
        k = sum(ok[mode])
        print(f"{mode:<16}{k:>8}{k / n * 100:>8.1f}%")

    print("\n--- cases the repair rescued (argmax invalid, closing valid)\n")
    shown = 0
    for i in range(n):
        if ok["argmax"][i] or not ok["grammar_close"][i]:
            continue
        before, after = decoded["argmax"][i], decoded["grammar_close"][i]
        print(f"[{shown + 1}] argmax   {before}")
        print(f"    why      {why_invalid(before)}")
        print(f"    closed   {after}")
        print(f"    change   {describe_change(before, after)}")
        print(f"    trimmed  {decoded['grammar_trim'][i]}"
              f"{'' if ok['grammar_trim'][i] else '   (still invalid)'}")
        print(f"    mixed    {decoded['grammar_mixed'][i]}"
              f"{'' if ok['grammar_mixed'][i] else '   (still invalid)'}")
        print()
        shown += 1
        if shown >= show:
            break

    print("--- accepted molecules, closing decoder\n")
    from dimol.eval.distribution import descriptors

    print(f"{'smiles':<52}{'atoms':>6}{'rings':>6}{'weight':>8}{'qed':>6}{'sa':>6}")
    kept = 0
    for i in range(n):
        if not ok["grammar_close"][i]:
            continue
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(decoded["grammar_close"][i]))
        d = descriptors(Chem.MolFromSmiles(smi))
        print(f"{smi[:50]:<52}{d['heavy_atoms']:>6.0f}{d['rings']:>6.0f}"
              f"{d['mol_weight']:>8.0f}{d['qed']:>6.2f}{d['sa_score']:>6.2f}")
        kept += 1
        if kept >= show_valid:
            break


if __name__ == "__main__":
    main(load_from_argv())
