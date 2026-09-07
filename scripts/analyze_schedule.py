#!/usr/bin/env python3
"""Where in the diffusion schedule does the model actually carry information?

    python scripts/analyze_schedule.py configs/diffusion_zinc250k.yaml \
        analyze.checkpoint=runs/r_base80/ep80-ba17600 analyze.points=41

For a grid of timesteps, real data is noised to that level and the denoiser is run once:

* eps error, the training objective at that t;
* x0 error relative to the norm of x0, which is what the sampler cares about;
* token accuracy of the readout applied to the reconstruction, i.e. whether the tokens
  are still recoverable from the model's estimate;
* the effective signal-to-noise ratio, alpha^2 |x0|^2 / (beta^2 d), which is the quantity
  the schedule is supposed to control.

The last column is the reason this script exists. In embedding-space diffusion the latent
is learned, so its scale drifts during training and the schedule drifts with it: the same
t means a different noise level at step 1000 and at step 17600. A model can lower its loss
by growing the embedding table instead of denoising better, and that is invisible in the
loss curve. Comparing checkpoints, or models trained with different timestep
distributions, on this grid shows where the capacity actually went.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dimol.builders import build_path, build_tokenizer  # noqa: E402
from dimol.config import load_from_argv  # noqa: E402
from dimol.data.loaders import build_dataloader  # noqa: E402
from dimol.training.distributed import init_distributed  # noqa: E402
from dimol.models.diffusion_transformer import DiffusionTransformer  # noqa: E402
from dimol.training.distributed import seed_all  # noqa: E402


@torch.no_grad()
def schedule_curve(model, path, batch, points: int, device, regime: str = "epsilon"):
    token_ids = batch["token_ids"].to(device)
    embed = model.token_embedding
    x0 = embed(token_ids)
    dim = x0.shape[-1]
    x0_norm = x0.pow(2).mean().sqrt().item()

    rows = []
    grid = torch.linspace(1e-3, 0.999, points)
    for t_value in grid:
        t = torch.full((x0.shape[0], 1, 1), float(t_value), device=device)
        alpha, beta = path.alpha(t), path.beta(t)
        noise = torch.randn_like(x0)
        x_t = alpha * x0 + beta * noise

        time = t.squeeze(-1)
        eps_theta = model(input_embeddings=x_t, time=time, attention_mask=None)
        target = noise if regime == "epsilon" else x0
        eps_error = (eps_theta - target).pow(2).mean().item()

        alpha_c = alpha.clamp(min=1e-3)
        x0_hat = (x_t - beta * eps_theta) / alpha_c if regime == "epsilon" else eps_theta
        x0_error = ((x0_hat - x0).pow(2).mean() / x0.pow(2).mean()).item()

        logits = model.out_proj(x0_hat)
        accuracy = (logits.argmax(-1) == token_ids).float().mean().item()
        logits_direct = model.out_proj(x_t)
        accuracy_direct = (logits_direct.argmax(-1) == token_ids).float().mean().item()

        a, b = float(alpha.flatten()[0]), float(beta.flatten()[0])
        snr = (a**2 * x0_norm**2) / max(b**2, 1e-12)
        rows.append({
            "t": float(t_value), "alpha": a, "beta": b, "snr": snr,
            "eps_error": eps_error, "x0_error": x0_error,
            "token_acc": accuracy, "token_acc_direct": accuracy_direct,
        })
    return rows, x0_norm


def main(cfg: DictConfig) -> None:
    node = OmegaConf.to_container(cfg.get("analyze") or {}, resolve=True) or {}
    checkpoint = node.get("checkpoint")
    if not checkpoint:
        raise KeyError("set analyze.checkpoint")
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(int(node.get("seed", 42)), tf32=bool(cfg.tf32))

    model = DiffusionTransformer.from_pretrained(load_dir=checkpoint, map_location=device)
    model.eval()
    tokenizer = build_tokenizer(cfg)
    seq_len = int(cfg.variables.get("seq_len"))
    path = build_path(cfg, seq_len=seq_len, emb_dim=model.config.emb_dim, device=device)

    env = init_distributed(device=device)
    loader, _ = build_dataloader(cfg.eval_loader, int(node.get("batch_size", 512)),
                                 env, is_train=False)
    batch = next(iter(loader))

    rows, x0_norm = schedule_curve(model, path, batch, int(node.get("points", 41)),
                                   device, regime=str(cfg.diffusion.get("regime", "epsilon")))

    print(f"checkpoint {checkpoint}")
    print(f"embedding rms norm {x0_norm:.3f}   (the schedule is stated in these units)\n")
    print(f"{'t':>6}{'alpha':>7}{'snr':>10}{'eps err':>9}{'x0 err':>9}"
          f"{'acc(x0_hat)':>13}{'acc(x_t)':>10}")
    for row in rows:
        print(f"{row['t']:>6.3f}{row['alpha']:>7.3f}{row['snr']:>10.1f}"
              f"{row['eps_error']:>9.4f}{row['x0_error']:>9.4f}"
              f"{row['token_acc']:>13.4f}{row['token_acc_direct']:>10.4f}")

    out = node.get("out")
    if out:
        Path(out).write_text(json.dumps({"checkpoint": str(checkpoint),
                                         "embedding_norm": x0_norm, "rows": rows}, indent=2))
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main(load_from_argv())
