#!/usr/bin/env python3
"""Text-to-molecule evaluation: generate one molecule per caption, then score it.

    # where the GPU is
    python scripts/eval_text.py --stage generate --config configs/chebi20_finetune.yaml \
        --checkpoint runs/ft_lr3.0e-4/ep60-ba6000 --split val --limit 1000 \
        --guidance 0 1 2 4 --out scratch/texteval
    # where rdkit is
    python scripts/eval_text.py --stage score --dir scratch/texteval

The generation stage reports only what can be measured without rdkit, because the job
container has none: whether the string is grammatical, by the same connectivity checker
the decoder uses, and how much of the reference's token multiset it recovers. The scoring
stage adds the chemistry the ChEBI-20 literature reports: rdkit validity, exact match
after canonicalisation, and MACCS, RDK and Morgan fingerprint similarity to the reference
molecule for the same caption.

Exact match needs canonicalisation on both sides, which is why it lives in the second
stage rather than the first.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def grammatical(smiles: str) -> bool:
    """Is the string parseable at all, by the decoder's own legality checker?"""
    from dimol.eval.smiles_state import SmilesState, token_events

    state = SmilesState()
    index = 0
    while index < len(smiles):
        if smiles[index] == "[":
            end = smiles.find("]", index)
            if end == -1:
                return False
            token, index = smiles[index : end + 1], end + 1
        elif smiles[index : index + 2] in ("Cl", "Br"):
            token, index = smiles[index : index + 2], index + 2
        elif smiles[index] == "%":
            token, index = smiles[index : index + 3], index + 3
        else:
            token, index = smiles[index], index + 1
        if not state.place(token_events(token)):
            return False
    return state.complete


def token_f1(generated: str, reference: str, tokenizer) -> float:
    """Multiset overlap of tokens, as a cheap in-job progress signal.

    It is computed over the tokenizer's segmentation, so the same atoms in a different
    order can score low: a byte-pair vocabulary splits "CCO" and "OCC" differently. Use
    it to watch a run move, not to compare chemistry. The fingerprint similarities in the
    scoring stage are the chemical measure.
    """
    a = Counter(tokenizer.encode(generated, add_special_tokens=False))
    b = Counter(tokenizer.encode(reference, add_special_tokens=False))
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision = overlap / max(sum(a.values()), 1)
    recall = overlap / max(sum(b.values()), 1)
    return 2 * precision * recall / (precision + recall)


def stage_generate(args) -> None:
    import torch
    from omegaconf import OmegaConf

    from dimol.builders import build_path
    from dimol.eval.sampling import SamplingParams, sample_smiles
    from dimol.models.diffusion_transformer import DiffusionTransformer
    from dimol.tokenization.smiles_tokenizer import SmilesTokenizer
    from dimol.training.distributed import seed_all

    cfg = OmegaConf.load(args.config)
    data_dir = Path(cfg.variables.data_dir)
    tokenizer = SmilesTokenizer.load(cfg.tokenizer["path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(args.seed)

    model = DiffusionTransformer.from_pretrained(load_dir=str(args.checkpoint),
                                                 map_location=device)
    model.eval()
    canvas = int(cfg.variables.seq_len)
    path = build_path(cfg, seq_len=canvas, emb_dim=getattr(model, "latent_dim", model.config.emb_dim), device=device)

    text = np.load(data_dir / f"{args.split}_text.npy", mmap_mode="r")
    text_mask = np.load(data_dir / f"{args.split}_text_mask.npy", mmap_mode="r")
    tokens = np.load(data_dir / f"{args.split}_tokens_00000.npy", mmap_mode="r")
    masks = np.load(data_dir / f"{args.split}_attn_mask_00000.npy", mmap_mode="r")
    n = min(args.limit, len(text)) if args.limit else len(text)
    print(f"{args.checkpoint}: {n} captions from {args.split}")

    references = [
        tokenizer.decode([int(t) for t, m in zip(tokens[i], masks[i]) if m],
                         skip_special_tokens=True)
        for i in range(n)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.txt").write_text("\n".join(references) + "\n")

    ar_guide = None
    if args.ar_guide is not None:
        from dimol.models.gpt import SmilesAR

        ar_guide = SmilesAR.from_pretrained(load_dir=str(args.ar_guide),
                                            map_location=device).to(device).eval()
        print(f"  AR guidance from {args.ar_guide} "
              f"(scale {args.ar_scale}, from alpha {args.ar_start_alpha})")

    predicted_lengths = None
    if args.length_file is not None:
        # Lengths from any external predictor, one row per caption in split order. This
        # keeps the sampler ignorant of how a length was arrived at, so a fine-tuned
        # encoder needs no changes here.
        stored = np.load(args.length_file)
        if len(stored) < n:
            raise ValueError(
                f"{args.length_file} holds {len(stored)} lengths for {n} captions"
            )
        predicted_lengths = np.asarray(stored[:n], dtype=np.int64)
        true_lengths = np.asarray(masks[:n]).sum(1)
        print(f"  length file: MAE "
              f"{np.abs(predicted_lengths - true_lengths).mean():.2f} tokens")
    elif args.length_head is not None:
        from dimol.models.length_head import LengthHead

        head = LengthHead.load(args.length_head, map_location=device).to(device)
        chunks = []
        with torch.no_grad():
            for start in range(0, n, 256):
                stop = min(start + 256, n)  # the arrays hold the whole split, not just n
                t = torch.from_numpy(np.asarray(text[start:stop],
                                                dtype=np.float32)).to(device)
                m = torch.from_numpy(np.asarray(text_mask[start:stop])
                                     .astype(bool)).to(device)
                chunks.append(head.predict(t, m, args.length_temperature).cpu())
        predicted_lengths = torch.cat(chunks).numpy()
        true_lengths = np.asarray(masks[:n]).sum(1)
        print(f"  length head: MAE {np.abs(predicted_lengths - true_lengths).mean():.2f} "
              f"tokens on these captions")

    length_cdf = None
    if args.length_posterior is not None:
        # P(length <= position), built from the predictor's own top-k distribution over
        # lengths. The canvas is then pinned by how sure the predictor is that the
        # molecule has ended, instead of being cut at its argmax.
        cand = np.load(args.length_posterior)[:n]
        probs = np.load(args.length_posterior.with_name(
            args.length_posterior.name.replace("lengths", "probs")))[:n]
        probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-8, None)
        canvas = int(OmegaConf.load(args.config).variables.seq_len)
        positions = np.arange(canvas).reshape(1, canvas, 1)
        below = (cand.reshape(len(cand), 1, -1) <= positions)
        length_cdf = (below * probs.reshape(len(probs), 1, -1)).sum(-1).astype(np.float32)
        length_cdf = np.clip(length_cdf, 0.0, 1.0)
        sharp = float(args.length_posterior_power)
        if sharp != 1.0:
            length_cdf = length_cdf ** sharp
        band = ((length_cdf > 1e-3) & (length_cdf < 1 - 1e-3)).sum(1)
        print(f"  length posterior: {band.mean():.1f} canvas positions per caption are "
              f"neither certainly molecule nor certainly padding")

    particles = max(int(args.smc_particles), 1)
    smc_alphas = tuple(float(a) for a in args.smc_alphas.split(",") if a.strip()) \
        if args.smc_alphas else ()
    length_matrix = None
    if particles > 1:
        if args.length_matrix is not None:
            # One length per particle, so the particles span the length candidates and
            # the filter chooses among plans rather than among noise draws alone.
            length_matrix = np.load(args.length_matrix)[:n]
            if length_matrix.shape[1] < particles:
                raise ValueError(
                    f"{args.length_matrix} holds {length_matrix.shape[1]} candidate "
                    f"lengths but {particles} particles were asked for")
            length_matrix = length_matrix[:, :particles].astype(np.int64)
        print(f"  {particles} particles per caption"
              + (f", resampling at alpha {smc_alphas}" if smc_alphas
                 else ", selected only at the end")
              + (" , lengths from the candidate matrix" if length_matrix is not None
                 else ""))

    caption_index = np.arange(n)
    if args.shuffle_captions:
        caption_index = np.roll(caption_index, n // 2)
        print("  captions shuffled: every molecule is paired with another one's caption")

    for scale in args.guidance:
        text_rows = np.asarray(text[:n], dtype=np.float32)[caption_index]
        mask_rows = np.asarray(text_mask[:n]).astype(bool)[caption_index]
        length_rows = (predicted_lengths if predicted_lengths is not None
                       else np.asarray(masks[:n]).sum(1))
        batch_size = args.batch_size
        if particles > 1:
            text_rows = np.repeat(text_rows, particles, axis=0)
            mask_rows = np.repeat(mask_rows, particles, axis=0)
            length_rows = (length_matrix.reshape(-1) if length_matrix is not None
                           else np.repeat(np.asarray(length_rows), particles))
            batch_size = max(particles, (batch_size // particles) * particles)

        params = SamplingParams(
            num_samples=n * particles, batch_size=batch_size,
            num_timesteps=args.steps,
            decode="grammar_close", strict=True, seed=args.seed, solver=args.solver,
            content_rule=args.content_rule,
            ar_model=ar_guide, ar_scale=args.ar_scale,
            ar_clamp_weight=args.ar_clamp,
            ar_clamp_strength=args.ar_clamp_strength,
            ar_repaint_rounds=args.ar_repaint,
            ar_repaint_t=args.ar_repaint_t,
            ar_repaint_steps=args.ar_repaint_steps,
            ar_repaint_min_prob=args.ar_repaint_min_prob,
            ar_min_prob=args.ar_min_prob,
            draft_max_prob=args.draft_max_prob,
            ar_start_alpha=args.ar_start_alpha,
            init_tokens=(np.load(args.init_tokens)[:n]
                         if args.init_tokens is not None else None),
            init_t=args.init_t,
            text=text_rows,
            text_mask=mask_rows,
            guidance=float(scale),
            length_prior=length_rows,
            length_exact=(bool(args.oracle_length)
                          or predicted_lengths is not None
                          or length_matrix is not None),
            smc_group=particles, smc_alphas=smc_alphas,
            smc_lambda=args.smc_lambda, smc_ess=args.smc_ess,
            length_cdf=(np.repeat(length_cdf, particles, axis=0)
                        if length_cdf is not None and particles > 1 else length_cdf),
        )
        generated = sample_smiles(model, path, tokenizer, params, device)
        name = (f"guidance_{scale}"
                + ("_heun" if args.solver == "heun" else "")
                + ("_shuffled" if args.shuffle_captions else "")
                + ("_oraclelen" if args.oracle_length else "")
                + (f"_rp{args.ar_repaint}" if args.ar_repaint else "")
                + (f"_arc{args.ar_clamp}" if args.ar_clamp else "")
                + (f"_arg{args.ar_scale}"
                   if args.ar_guide is not None and not args.ar_clamp else "")
                + (f"_init{args.init_t}" if args.init_tokens is not None else "")
                + ("_filelen" if args.length_file is not None else "")
                + ("_headlen" if args.length_head is not None
                   and args.length_file is None else "")
                + (f"_smc{particles}" if particles > 1 else "")
                + (f"_post{args.length_posterior_power}"
                   if length_cdf is not None else ""))
        if particles > 1:
            # One directory per particle slot, each a drop-in for the candidate pipeline:
            # validity needs a chemistry toolkit, so the choice among survivors is made
            # downstream rather than here.
            for j in range(particles):
                sub = args.out / f"p{j}"
                sub.mkdir(parents=True, exist_ok=True)
                (sub / "reference.txt").write_text("\n".join(references) + "\n")
                (sub / f"{name}.txt").write_text(
                    "\n".join(generated[j::particles]) + "\n")
            generated = generated[0::particles]
        (args.out / f"{name}.txt").write_text("\n".join(generated) + "\n")
        ok = sum(grammatical(s) for s in generated)
        f1 = float(np.mean([token_f1(g, r, tokenizer)
                            for g, r in zip(generated, references)]))
        print(f"  guidance {scale}: grammatical {ok / n * 100:.1f}%, token F1 {f1:.3f}")
    print(f"written to {args.out}")


def stage_score(args) -> None:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint

    RDLogger.DisableLog("rdApp.*")
    directory = args.dir
    references = (directory / "reference.txt").read_text().splitlines()
    reference_mols = [Chem.MolFromSmiles(s) for s in references]

    rows = {}
    for path in sorted(directory.glob("guidance_*.txt")):
        generated = path.read_text().splitlines()
        valid = exact = 0
        maccs, rdk, morgan = [], [], []
        for gen, ref, ref_mol in zip(generated, references, reference_mols):
            mol = Chem.MolFromSmiles(gen) if gen else None
            if mol is None:
                continue
            valid += 1
            if ref_mol is None:
                continue
            if Chem.MolToSmiles(mol) == Chem.MolToSmiles(ref_mol):
                exact += 1
            maccs.append(DataStructs.TanimotoSimilarity(
                MACCSkeys.GenMACCSKeys(mol), MACCSkeys.GenMACCSKeys(ref_mol)))
            rdk.append(DataStructs.TanimotoSimilarity(
                RDKFingerprint(mol), RDKFingerprint(ref_mol)))
            morgan.append(DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
                AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048)))
        n = max(len(generated), 1)
        rows[path.stem] = {
            "n": len(generated),
            "validity": valid / n,
            "exact_match": exact / n,
            "maccs": float(np.mean(maccs)) if maccs else 0.0,
            "rdk": float(np.mean(rdk)) if rdk else 0.0,
            "morgan": float(np.mean(morgan)) if morgan else 0.0,
        }

    print("%-16s%6s%10s%12s%9s%8s%9s" % ("setting", "n", "validity", "exact match",
                                         "MACCS", "RDK", "Morgan"))
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["maccs"]):
        print("%-16s%6d%9.1f%%%11.2f%%%9.3f%8.3f%9.3f"
              % (name, r["n"], r["validity"] * 100, r["exact_match"] * 100,
                 r["maccs"], r["rdk"], r["morgan"]))
    (directory / "text_metrics.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("generate", "score"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--guidance", type=float, nargs="+", default=[0.0])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--length-file", type=Path, default=None,
                        help="npy array of predicted lengths, one per caption in split "
                             "order, from scripts/train_length_encoder.py. Takes "
                             "precedence over --length-head")
    parser.add_argument("--ar-guide", type=Path, default=None,
                        help="checkpoint of a caption-conditioned autoregressive model "
                             "whose log-likelihood gradient steers the reverse SDE. "
                             "Ranking finished samples by this model took exact match "
                             "from 0.052 to 0.121; this consults it during the trajectory "
                             "instead of after it")
    parser.add_argument("--ar-scale", type=float, default=1.0)
    parser.add_argument("--ar-clamp", type=float, default=0.0,
                        help="weight, in logit units, on the autoregressive model's "
                             "opinion when the x0 estimate is snapped to tokens at each "
                             "step. 1.0 weighs it equally with the diffusion readout. "
                             "This revises a DISCRETE draft, which is what the AR model "
                             "was trained on, unlike the gradient route")
    parser.add_argument("--ar-clamp-strength", type=float, default=1.0)
    parser.add_argument("--ar-repaint", type=int, default=0,
                        help="rounds of selective repaint: re-noise only the positions "
                             "the autoregressive verifier confidently disputes and let "
                             "the diffusion redo them, holding the rest of the molecule "
                             "fixed. Forcing the verifier's token destroys the sample; "
                             "this forces nothing")
    parser.add_argument("--ar-repaint-t", type=float, default=0.75)
    parser.add_argument("--ar-repaint-steps", type=int, default=40)
    parser.add_argument("--ar-repaint-min-prob", type=float, default=0.9)
    parser.add_argument("--ar-min-prob", type=float, default=0.0,
                        help="revise only where the AR model is this confident")
    parser.add_argument("--draft-max-prob", type=float, default=1.0,
                        help="and only where the diffusion readout is not")
    parser.add_argument("--length-posterior", type=Path, default=None,
                        help="n x k array of candidate lengths whose sibling *_probs.npy "
                             "holds their probabilities. The canvas is then pinned by "
                             "P(length <= position) instead of cut at the argmax, so one "
                             "trajectory covers the posterior and the readout chooses the "
                             "stop inside its support")
    parser.add_argument("--length-posterior-power", type=float, default=1.0,
                        help="raise the posterior to this power before pinning; above 1 "
                             "narrows the uncertain band toward a hard length")
    parser.add_argument("--smc-particles", type=int, default=1,
                        help="carry this many trajectories per caption and let the "
                             "verifier reallocate compute among them during the solve. "
                             "The verifier cannot edit a position -- forcing destroys "
                             "the sample, its gradient moves nothing, and re-noising a "
                             "disputed position returns the same token -- but it ranks "
                             "whole molecules well, so it is consulted on whole states")
    parser.add_argument("--smc-alphas", default="",
                        help="comma separated alpha values at which to resample; empty "
                             "keeps the particles independent, which is selection at the "
                             "end and the control this has to beat")
    parser.add_argument("--smc-lambda", type=float, default=1.0)
    parser.add_argument("--smc-ess", type=float, default=0.5)
    parser.add_argument("--length-matrix", type=Path, default=None,
                        help="n x k array of candidate lengths, one column per particle")
    parser.add_argument("--ar-start-alpha", type=float, default=0.5,
                        help="leave guidance off until the x0 estimate carries signal")
    parser.add_argument("--init-tokens", type=Path, default=None,
                        help="npy of (N, T) token ids to start the reverse SDE from, "
                             "instead of pure noise: a retrieved molecule, noised to "
                             "--init-t. Retrieval already reaches the fingerprint metrics "
                             "on this benchmark, so this hands the diffusion the "
                             "neighbourhood and asks it only for the details")
    parser.add_argument("--init-t", type=float, default=0.5,
                        help="how far back to noise the starting molecule; lower keeps "
                             "more of it")
    parser.add_argument("--content-rule", default="last_positive",
                        choices=("first_negative", "last_positive", "run_2", "run_3",
                                 "run_5"),
                        help="how a content channel becomes a stop position. Only used by "
                             "models trained with model.content_channel")
    parser.add_argument("--solver", default="euler_maruyama",
                        choices=("euler_maruyama", "heun"),
                        help="heun re-averages the drift at the end of each step for two "
                             "model evaluations, so --steps N costs what 2N euler steps "
                             "cost; compare at matched compute, not matched steps")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-head", type=Path, default=None,
                        help="predict each molecule's length from its caption with this "
                             "head, instead of drawing a length from the corpus. This is "
                             "the honest version of --oracle-length")
    parser.add_argument("--length-temperature", type=float, default=0.0,
                        help="0 takes the head's most likely length, above 0 samples")
    parser.add_argument("--oracle-length", action="store_true",
                        help="give each sample the reference molecule's own length. An "
                             "upper bound, not a result: a real system would have to "
                             "predict the length from the caption")
    parser.add_argument("--shuffle-captions", action="store_true",
                        help="pair every molecule with somebody else's caption. The "
                             "control that separates weak conditioning from none: if the "
                             "metrics do not fall, the text path is decorative")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.stage == "generate":
        for needed in ("config", "checkpoint", "out"):
            if getattr(args, needed) is None:
                raise SystemExit(f"--{needed} is required for --stage generate")
        stage_generate(args)
    else:
        if args.dir is None:
            raise SystemExit("--dir is required for --stage score")
        stage_score(args)


if __name__ == "__main__":
    main()
