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
    path = build_path(cfg, seq_len=canvas, emb_dim=model.config.emb_dim, device=device)

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

    caption_index = np.arange(n)
    if args.shuffle_captions:
        caption_index = np.roll(caption_index, n // 2)
        print("  captions shuffled: every molecule is paired with another one's caption")

    for scale in args.guidance:
        params = SamplingParams(
            num_samples=n, batch_size=args.batch_size, num_timesteps=args.steps,
            decode="grammar_close", strict=True, seed=args.seed,
            text=np.asarray(text[:n], dtype=np.float32)[caption_index],
            text_mask=np.asarray(text_mask[:n]).astype(bool)[caption_index],
            guidance=float(scale),
            length_prior=np.asarray(masks[:n]).sum(1),
            length_exact=bool(args.oracle_length),
        )
        generated = sample_smiles(model, path, tokenizer, params, device)
        name = (f"guidance_{scale}"
                + ("_shuffled" if args.shuffle_captions else "")
                + ("_oraclelen" if args.oracle_length else ""))
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
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
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
