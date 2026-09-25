#!/usr/bin/env python3
"""Predict molecule length from the caption with a fine-tuned encoder.

    python scripts/train_length_encoder.py --config configs/chebi20_finetune.yaml \
        --corpus data/chebi20/tokenized --encoder <path to scibert> \
        --out data/chebi20/length_encoder

Why this exists. Giving the sampler the true length takes exact match from 3.1% to 14.2%
and MACCS from 0.723 to 0.777 on the ChEBI-20 test split, so length is the largest
measured lever left. The head that reads frozen SciBERT states got mean absolute error
from 7.5 tokens down to 6.4 and then stopped: 6.48, 6.43, 6.31, 6.39 over the last four
evaluations of an 80-epoch run. A head that has stopped improving while its loss keeps
falling is limited by its inputs, not its size, so the encoder itself has to move.

The task is small enough that this is cheap: 25,574 captions, one BERT-base fine-tune.
Length is stated compositionally in a large share of ChEBI captions ("the acyl group has
27 carbons and 0 double bonds"), and a frozen pooled summary is exactly where that kind
of detail goes missing.

The output is a directory holding the fine-tuned encoder, the head, and one predicted
length per caption for every split. The predictions are what the sampler consumes, via
``eval_text.py --length-file``, so nothing about the diffusion model or its sampler needs
to know how a length was arrived at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LengthRegressor(nn.Module):
    """Encoder plus a two-headed readout: a distribution over lengths and a scalar.

    The classification head is what the frozen-state head already did, and it is the one
    that gives a mode to sample from. The regression head is added because length is
    ordinal: cross-entropy alone treats a prediction of 12 for a target of 40 as exactly
    as wrong as 39, which throws away the only structure the label has. Training both and
    reading the classifier's argmax keeps the calibrated distribution while letting the
    ordinal signal shape the encoder.
    """

    def __init__(self, encoder, hidden: int, max_length: int):
        super().__init__()
        self.encoder = encoder
        self.max_length = int(max_length)
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, self.max_length + 1)
        self.regressor = nn.Linear(hidden, 1)

    def pool(self, states, mask):
        """Attention pooling over caption tokens, padding excluded."""
        weights = self.attention(states).squeeze(-1)
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = weights.softmax(-1).unsqueeze(-1)
        return (states * weights).sum(1)

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        pooled = self.pool(states, attention_mask.bool())
        return self.classifier(pooled), self.regressor(pooled).squeeze(-1)

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        logits, _ = self(input_ids, attention_mask)
        return logits.argmax(-1)


def smoothed_loss(logits, target, sigma: float = 1.5):
    """Cross-entropy against a narrow Gaussian on the length axis.

    A one-hot target says a molecule of 31 tokens is as wrong as one of 3 when the answer
    is 30. Spreading the target over neighbouring lengths encodes that being close is
    better, which is what the metric actually rewards: the sampler pins padding past the
    predicted length, so an error of one token costs one token.
    """
    positions = torch.arange(logits.shape[-1], device=logits.device).float()
    centred = positions[None, :] - target[:, None].float()
    weights = torch.exp(-0.5 * (centred / sigma) ** 2)
    weights = weights / weights.sum(-1, keepdim=True)
    return -(weights * logits.log_softmax(-1)).sum(-1).mean()


def load_split(corpus: Path, split: str):
    captions = (corpus / f"{split}_captions.txt").read_text().splitlines()
    lengths = np.load(corpus / f"{split}_attn_mask_00000.npy").sum(1)
    if len(captions) != len(lengths):
        raise ValueError(
            f"{split}: {len(captions)} captions against {len(lengths)} molecules; "
            "the pair files are misaligned and the labels would be wrong"
        )
    return captions, np.asarray(lengths, dtype=np.int64)


def encode(tokenizer, captions, max_tokens, device):
    batch = tokenizer(captions, padding="max_length", truncation=True,
                      max_length=max_tokens, return_tensors="pt")
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.no_grad()
def evaluate(model, tokenizer, captions, lengths, max_tokens, device, batch_size):
    model.eval()
    errors = []
    for start in range(0, len(captions), batch_size):
        chunk = captions[start : start + batch_size]
        ids, mask = encode(tokenizer, chunk, max_tokens, device)
        pred = model.predict(ids, mask).cpu().numpy()
        errors.append(np.abs(pred - lengths[start : start + batch_size]))
    err = np.concatenate(errors)
    model.train()
    return err.mean(), float((err <= 3).mean() * 100), float((err == 0).mean() * 100)


@torch.no_grad()
def predict_all(model, tokenizer, captions, max_tokens, device, batch_size):
    model.eval()
    out = []
    for start in range(0, len(captions), batch_size):
        ids, mask = encode(tokenizer, captions[start : start + batch_size],
                           max_tokens, device)
        out.append(model.predict(ids, mask).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="directory with <split>_captions.txt and the token arrays")
    parser.add_argument("--encoder", type=Path, default=None,
                        help="defaults to prepare.source.text_encoder in the config")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-5,
                        help="encoder learning rate; the heads get 10x this")
    parser.add_argument("--lambda-regression", type=float, default=0.1)
    parser.add_argument("--max-caption-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    corpus = args.corpus or Path(cfg.variables.root) / "tokenized"
    encoder_path = args.encoder or Path(cfg.prepare.source.text_encoder)
    max_length = int(cfg.model.max_pos)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path))
    encoder = AutoModel.from_pretrained(str(encoder_path))
    hidden = encoder.config.hidden_size
    model = LengthRegressor(encoder, hidden, max_length).to(device)

    train_captions, train_lengths = load_split(corpus, "train")
    val_captions, val_lengths = load_split(corpus, "val")
    print(f"{encoder_path}: hidden {hidden}, {len(train_captions)} train captions, "
          f"lengths up to {max_length}, on {device}")
    print(f"  predicting the corpus mean would give MAE "
          f"{np.abs(val_lengths - train_lengths.mean()).mean():.2f} tokens")

    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": args.lr},
         {"params": heads, "lr": args.lr * 10}], weight_decay=0.01)
    steps = args.epochs * ((len(train_captions) + args.batch_size - 1) // args.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * 10], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = (float("inf"), -1)
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(train_captions))
        running = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            ids, mask = encode(tokenizer, [train_captions[i] for i in idx],
                               args.max_caption_tokens, device)
            target = torch.from_numpy(train_lengths[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, scalar = model(ids, mask)
                loss = (smoothed_loss(logits, target)
                        + args.lambda_regression * F.l1_loss(scalar, target.float()))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += float(loss) * len(idx)
            seen += len(idx)

        mae, within3, exact = evaluate(model, tokenizer, val_captions, val_lengths,
                                       args.max_caption_tokens, device, 128)
        flag = ""
        if mae < best[0]:
            best = (mae, epoch)
            model.encoder.save_pretrained(args.out / "encoder")
            tokenizer.save_pretrained(args.out / "encoder")
            torch.save({k: v for k, v in model.state_dict().items()
                        if not k.startswith("encoder.")}, args.out / "heads.pt")
            flag = "  <- saved"
        print(f"  epoch {epoch:3}: loss {running / seen:.3f}, val MAE {mae:5.2f}, "
              f"within 3 {within3:4.1f}%, exact {exact:4.1f}%{flag}")

    print(f"best MAE {best[0]:.2f} at epoch {best[1]}")

    # Reload the best encoder before writing predictions, so the files match the
    # checkpoint that was kept rather than whatever the last epoch happened to be.
    model.encoder = AutoModel.from_pretrained(str(args.out / "encoder")).to(device)
    model.load_state_dict(torch.load(args.out / "heads.pt", map_location=device),
                          strict=False)
    for split in ("train", "val", "test"):
        captions, lengths = load_split(corpus, split)
        pred = predict_all(model, tokenizer, captions, args.max_caption_tokens,
                           device, 128)
        np.save(args.out / f"{split}_pred_lengths.npy", pred)
        print(f"  {split}: MAE {np.abs(pred - lengths).mean():.2f} tokens, "
              f"written to {args.out / f'{split}_pred_lengths.npy'}")

    (args.out / "meta.json").write_text(json.dumps(
        {"encoder": str(encoder_path), "epochs": args.epochs, "lr": args.lr,
         "best_val_mae": best[0], "best_epoch": best[1],
         "max_length": max_length}, indent=2))


if __name__ == "__main__":
    main()
