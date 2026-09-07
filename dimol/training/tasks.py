"""Training tasks: what exactly the loss is computed on.

IMPORTANT: the loss math is carried over VERBATIM from
``dimol/diffusion/trainers.py`` (classes ``ConditionalGaussianDenoiserTrainerLite``
and ``ARTrainer``). Only the following changed:

* constants that used to be hardcoded (the time eps 5e-5, the alpha thresholds
  0.80 and 0.5) became config fields with THE SAME defaults;
* the grammar-loss tables (paren_delta / ring_count) are built here instead of
  being stored in the config;
* the set of metric keys is now the same on every step (previously the
  ``ce_*``/``mse_*`` per-noise-bucket keys appeared only when a bucket was empty,
  so all_reduce could receive different key sets on different ranks).

The formulas themselves are unchanged: see docs/architecture.md, section
"what was deliberately NOT changed".
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dimol import registry
from dimol.training.distributed import unwrap_model

NOISE_BUCKETS = ((0.0, 0.3, "high_noise"), (0.3, 0.7, "mid_noise"), (0.7, 1.0, "low_noise"))


def build_grammar_tables(tokenizer, vocab_size: int, device: str | torch.device):
    """paren_delta / ring_count, carried over from the old scripts/train.py::main."""
    paren_delta = torch.zeros(vocab_size, device=device)  # "(" minus ")"
    ring_count = torch.zeros(vocab_size, 10, device=device)  # per-digit char counts
    for tok, idx in tokenizer.get_vocab().items():
        if tok in tokenizer.SPECIAL_TOKENS:
            continue
        if idx >= vocab_size:
            continue
        paren_delta[idx] = tok.count("(") - tok.count(")")
        for d in range(10):
            ring_count[idx, d] = tok.count(str(d))
    return paren_delta, ring_count


class Task:
    """The task interface expected by the trainer."""

    #: tokens per sample (used for throughput)
    seq_len: int = 0

    def compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError


@registry.tasks.register("diffusion")
class DiffusionTask(Task):
    def __init__(
        self,
        path,
        *,
        pad_idx: int = 0,
        regime: str = "epsilon",
        x0_noise_std: float = 0.25,
        x0_noise_kind: str = "gaussian",
        embedding_norm: str = "none",
        time_sampler: str = "uniform",
        time_logit_mean: float = 0.0,
        time_logit_std: float = 1.0,
        t_eps: float = 5e-5,
        lambda_mse: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_grammar: float = 0.0,
        alpha_eps: float = 1e-3,
        ce_alpha_threshold: float = 0.80,
        mse_t0_alpha_threshold: float = 0.80,
        grammar_alpha_threshold: float = 0.5,
        label_smoothing: float = 0.0,
        ce_input: str = "x0",
        mask_padding: str = "none",
        pad_weight: float = 1.0,
        self_cond_prob: float = 0.5,
        ce_include_pad: bool = False,
        min_snr_gamma: Optional[float] = None,
        decoder_pretrain_steps: int = 0,
        grammar_enabled: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        paren_delta: Optional[torch.Tensor] = None,
        ring_count: Optional[torch.Tensor] = None,
        autocast: Callable[[], Any] = nullcontext,
        autocast_scope: str = "loss",
        seq_len: int = 0,
    ):
        if regime not in ("epsilon", "x"):
            raise ValueError(
                f'Incorrect training regime: {regime}. Expected to be "epsilon" or "x"'
            )
        self.path = path
        self.pad_idx = pad_idx
        self.regime = regime
        self.x0_noise_std = x0_noise_std
        if x0_noise_kind not in ("gaussian", "laplace", "sphere", "token_mixup"):
            raise ValueError(f"diffusion.x0_noise_kind={x0_noise_kind!r} is unknown")
        self.x0_noise_kind = x0_noise_kind
        if embedding_norm not in ("none", "rms"):
            raise ValueError(f"diffusion.embedding_norm={embedding_norm!r} is unknown")
        self.embedding_norm = embedding_norm
        if time_sampler not in ("uniform", "logit_normal"):
            raise ValueError(f"diffusion.time_sampler={time_sampler!r} is unknown")
        self.time_sampler = time_sampler
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.t_eps = t_eps
        self.lambda_mse = lambda_mse
        self.lambda_ce = lambda_ce
        self.lambda_grammar = lambda_grammar
        self.alpha_eps = alpha_eps
        self.ce_alpha_threshold = ce_alpha_threshold
        self.mse_t0_alpha_threshold = mse_t0_alpha_threshold
        self.grammar_alpha_threshold = grammar_alpha_threshold
        self.label_smoothing = label_smoothing
        if ce_input not in ("x0", "x0_hat"):
            raise ValueError(f"loss.ce_input={ce_input!r}; expected 'x0' or 'x0_hat'")
        self.ce_input = ce_input
        if isinstance(mask_padding, bool):
            mask_padding = "both" if mask_padding else "none"
        if mask_padding not in ("none", "attention", "loss", "both"):
            raise ValueError(f"loss.mask_padding={mask_padding!r} is unknown")
        self.mask_padding = mask_padding
        self.mask_attention = mask_padding in ("attention", "both")
        self.mask_loss = mask_padding in ("loss", "both")
        # pad_weight is the continuous version of mask_padding="loss": 1.0 keeps the
        # original objective, 0.0 is the mask, and anything between keeps some
        # supervision on where the molecule stops while freeing capacity for the atoms.
        if not 0.0 <= float(pad_weight) <= 1.0:
            raise ValueError(f"loss.pad_weight={pad_weight!r} must be in [0, 1]")
        self.pad_weight = float(pad_weight)
        # Self-conditioning, when the model has the input for it: on this share of steps
        # the model first estimates x0 with the second input zeroed, then denoises again
        # with that estimate fed back. The first pass costs one extra forward and no
        # backward, and the gradient never flows through it.
        self.self_cond_prob = float(self_cond_prob)
        # The original cross-entropy ignores padding, so the readout is never told to
        # emit it. On a fixed canvas padding is the only way a molecule can end, and the
        # model's main failure is not ending; including it gives that decision explicit
        # supervision at the cost of a very frequent, very easy class.
        self.ce_include_pad = bool(ce_include_pad)
        if min_snr_gamma is not None and float(min_snr_gamma) <= 0.0:
            raise ValueError(f"loss.min_snr_gamma={min_snr_gamma!r} must be > 0 or null")
        self.min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)
        self.decoder_pretrain_steps = decoder_pretrain_steps
        self.grammar_enabled = grammar_enabled and lambda_grammar != 0.0
        self.class_weights = class_weights
        self.paren_delta = paren_delta
        self.ring_count = ring_count
        self.autocast = autocast
        if autocast_scope not in ("loss", "forward"):
            raise ValueError(
                f"loss.autocast_scope={autocast_scope!r}; expected 'loss' or 'forward'"
            )
        self.autocast_scope = autocast_scope
        self.seq_len = seq_len

        if self.grammar_enabled and (paren_delta is None or ring_count is None):
            raise ValueError("grammar loss is enabled but paren_delta/ring_count tables are missing")

    # ------------------------------------------------------------------
    def _sample_x0_noise(self, reference, token_ids, embed) -> torch.Tensor:
        """Corruption applied to the data latents before diffusion.

        "gaussian" is the original isotropic noise. "laplace" and "sphere" keep the same
        per-dimension variance but change the shape of the perturbation. "token_mixup"
        moves the latent towards the embedding of another token instead of a random
        direction, so the corruption stays on the data manifold and the denoiser is asked
        which token it was rather than how to remove isotropic noise.
        """
        kind = self.x0_noise_kind
        if kind == "gaussian":
            return torch.randn_like(reference)
        if kind == "laplace":
            u = torch.rand_like(reference) - 0.5
            return -(2 ** -0.5) * torch.sign(u) * torch.log1p(-2 * u.abs())
        if kind == "sphere":
            n = torch.randn_like(reference)
            scale = reference.shape[-1] ** 0.5
            return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6) * scale
        other = torch.randint_like(token_ids, low=1, high=int(embed.num_embeddings))
        return embed(other) - reference

    def _sample_time(self, batch_size: int, device) -> torch.Tensor:
        eps = self.t_eps
        if self.time_sampler == "uniform":
            return torch.rand(batch_size, 1, 1, device=device) * (1 - 2 * eps) + eps
        logits = torch.randn(batch_size, 1, 1, device=device)
        t = torch.sigmoid(self.time_logit_mean + self.time_logit_std * logits)
        return t.clamp(eps, 1 - eps)

    def compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:
        # autocast_scope="loss" (default) wraps the whole loss, not just the denoiser
        # forward: the readout matmul and the (B, L, V) logits then live in bf16/fp16
        # instead of fp32, while autocast keeps the reductions (mse_loss,
        # cross_entropy, softmax) in fp32 by its own op policy. The formulas are
        # unchanged. autocast_scope="forward" restores the original placement, where
        # only the denoiser call ran under autocast; useful for A/B measurements.
        if self.autocast_scope == "loss":
            with self.autocast():
                return self._compute_loss(model, batch, step)
        return self._compute_loss(model, batch, step)

    def _compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:
        token_ids = batch["token_ids"]  # B L
        # mask_padding=False (default) reproduces the original behaviour: the padding
        # mask arrives in the batch but is passed neither to attention nor to the loss.
        # mask_padding=True switches on the masked variants that the old trainer carried
        # as commented-out code; it is what makes length bucketing safe, because the
        # objective then stops depending on how many pad positions a batch contains.
        pad_mask = batch.get("attention_mask")
        if (self.mask_padding != "none" or self.pad_weight != 1.0) and pad_mask is None:
            raise ValueError(
                "loss.mask_padding / loss.pad_weight need 'attention_mask' in the batch"
            )

        raw = unwrap_model(model)
        embed = raw.token_embedding
        get_logits = raw.out_proj

        token_embeddings = embed(token_ids)
        if self.embedding_norm == "rms":
            # fix the latent scale: the raw table grows by ~50x over a run, which silently
            # changes the signal-to-noise ratio of the whole schedule
            token_embeddings = token_embeddings / token_embeddings.pow(2).mean(
                -1, keepdim=True
            ).add(1e-8).sqrt()

        x0 = token_embeddings + self._sample_x0_noise(
            token_embeddings, token_ids, embed
        ) * self.x0_noise_std

        batch_size, seq_len, emb_dim = x0.shape

        t = self._sample_time(batch_size, token_ids.device)

        noise = torch.randn_like(x0)

        alpha, beta = self.path.alpha(t), self.path.beta(t)
        x = alpha * x0 + beta * noise

        time = t
        if time.dim() == 3:
            time = time.squeeze(-1)  # (B, 1, 1) -> (B,1)

        target = noise if self.regime == "epsilon" else x0

        # (B, 1, 1, L) boolean mask; True marks positions that take part in attention
        attn_mask = pad_mask[:, None, None, :].bool() if self.mask_attention else None
        forward_ctx = nullcontext if self.autocast_scope == "loss" else self.autocast
        x0_self = None
        raw_model = unwrap_model(model)
        if getattr(raw_model, "self_conditioning", False) and self.self_cond_prob > 0:
            use = torch.rand((), device=x.device) < self.self_cond_prob
            if bool(use):
                with torch.no_grad(), forward_ctx():
                    first = model(input_embeddings=x, time=time, attention_mask=attn_mask)
                alpha_first = alpha.clamp(min=self.alpha_eps)
                if self.regime == "epsilon":
                    x0_self = ((x - beta * first) / alpha_first).detach()
                else:
                    x0_self = first.detach()

        forward_kwargs = {"input_embeddings": x, "time": time, "attention_mask": attn_mask}
        if getattr(raw_model, "self_conditioning", False):
            forward_kwargs["x0_self"] = x0_self  # None means "zeros", handled by the model
        with forward_ctx():
            eps_theta = model(**forward_kwargs)

        # Per-sample weight from the signal-to-noise ratio of the drawn timestep. The
        # plain mean treats every t alike, which lets the easy high-alpha steps, where
        # the target is nearly free to predict, dominate the gradient. min-SNR caps the
        # weight of those steps at gamma. Off by default, so the default objective is
        # bit-for-bit the original one.
        snr_weight = None
        if self.min_snr_gamma is not None:
            snr = (alpha**2 / beta.clamp(min=1e-8) ** 2).squeeze(-1).squeeze(-1)  # (B,)
            capped = snr.clamp(max=self.min_snr_gamma)
            snr_weight = capped / snr.clamp(min=1e-8) if self.regime == "epsilon" else capped

        if self.mask_loss or self.pad_weight != 1.0:
            loss_mask = pad_mask.float()  # (B, L)
            pos_w = loss_mask if self.mask_loss else loss_mask + (1.0 - loss_mask) * self.pad_weight
            mse_per_pos = ((eps_theta - target) ** 2).mean(dim=-1)  # (B, L)
            mse_per_sample = (mse_per_pos * pos_w).sum(-1) / pos_w.sum(-1).clamp(min=1e-8)  # (B,)
            mse_loss = (
                mse_per_sample.mean()
                if snr_weight is None
                else (mse_per_sample * snr_weight).sum() / snr_weight.sum().clamp(min=1e-8)
            )
        elif snr_weight is None:
            mse_loss = ((eps_theta - target) ** 2).mean()
        else:
            mse_per_sample = ((eps_theta - target) ** 2).mean(dim=(-1, -2))  # (B,)
            mse_loss = (mse_per_sample * snr_weight).sum() / snr_weight.sum().clamp(min=1e-8)

        # ----- Denoise to predicted x0 -----
        alpha = alpha.clamp(min=self.alpha_eps)  # (B, 1, 1)

        if self.regime == "epsilon":
            x0_hat = (x - beta * eps_theta) / alpha
        else:
            x0_hat = eps_theta  # (B, L, C)

        # ----- Reconstruction MSE (only at low-noise / high-alpha steps) -----
        mse_t0_sample_mask = (alpha > self.mse_t0_alpha_threshold).squeeze(-1).squeeze(-1)  # (B,)
        if self.mask_loss:
            mse_t0_loss_mask = loss_mask * mse_t0_sample_mask[:, None].float()  # (B, L)
        else:
            mse_t0_loss_mask = mse_t0_sample_mask[:, None].float()  # (B, 1)

        if mse_t0_loss_mask.sum() > 0:
            sq = ((x0_hat - token_embeddings) ** 2).mean(-1)  # (B, L) avg over C
            mse_loss_t0 = (sq * mse_t0_loss_mask).sum() / mse_t0_loss_mask.sum().clamp(min=1)
        else:
            mse_loss_t0 = torch.tensor(0.0, device=x.device)

        # ----- Cross-entropy (gate by alpha) -----
        # ce_input="x0" is the original behaviour: the readout is trained on the data
        # latents, so it only has to invert the embedding table it owns, and at sampling
        # time it is fed the SDE output instead. ce_input="x0_hat" trains it on what it
        # will actually see and sends the gradient through the denoiser.
        logits = get_logits(x0 if self.ce_input == "x0" else x0_hat)  # (B, L, V)

        ce_sample_mask = (alpha > self.ce_alpha_threshold).squeeze(-1).squeeze(-1)  # (B)

        if ce_sample_mask.any():
            ce_loss = F.cross_entropy(
                logits[ce_sample_mask].reshape(-1, logits.size(-1)),
                token_ids[ce_sample_mask].reshape(-1),
                ignore_index=-100 if self.ce_include_pad else self.pad_idx,
                label_smoothing=self.label_smoothing,
                reduction="mean",
                weight=self.class_weights,
            )
        else:
            ce_loss = logits.sum() * 0.0

        alpha_flat = alpha.squeeze(-1).squeeze(-1)  # (B,)

        if self.grammar_enabled:
            probs_mask = alpha_flat > self.grammar_alpha_threshold
            if probs_mask.any():
                probs = logits[probs_mask].softmax(-1)  # (B, L, V)
                # paren balance: cumulative net parens must stay >= 0 and end at 0
                delta = (probs * self.paren_delta).sum(-1)  # (B, L)
                balance = delta.cumsum(dim=1)
                paren_loss = F.relu(-balance).mean() + balance[:, -1].abs().mean()

                # ring parity: each digit's expected count should be even
                ring_exp = torch.einsum("blv,vd->bd", probs, self.ring_count)  # (B, 10)
                ring_loss = (ring_exp - ring_exp.round()).pow(2).mean()  # ~0 when even

                grammar_loss = paren_loss + ring_loss
            else:
                grammar_loss = logits.sum() * 0.0
        else:
            grammar_loss = torch.zeros((), device=x.device)

        metrics: Dict[str, torch.Tensor] = {}

        # ----- Accuracy -----
        with torch.no_grad():
            pred_ids = logits.argmax(-1)  # (B, L)
            correct = (pred_ids == token_ids) & (token_ids != self.pad_idx)  # (B, L)
            valid = token_ids != self.pad_idx
            metrics["token_acc"] = correct.sum() / valid.sum().clamp(min=1)

            for lo, hi, name in NOISE_BUCKETS:
                bucket = (alpha_flat >= lo) & (alpha_flat < hi)
                if bucket.any():
                    sel_correct = correct[bucket]
                    sel_valid = valid[bucket]
                    metrics[f"token_acc_{name}"] = sel_correct.sum() / sel_valid.sum().clamp(min=1)
                else:
                    metrics[f"token_acc_{name}"] = torch.zeros((), device=x.device)

            emb = raw.token_embedding.weight
            emb_norms = emb.norm(dim=-1)
            metrics["emb_norm_mean"] = emb_norms.mean()
            metrics["emb_norm_std"] = emb_norms.std()
            metrics["eps_theta_norm"] = eps_theta.detach().norm(dim=-1).mean()

            # Treat tokens with norm < 1e-3 as dead (never updated / WD-collapsed)
            # and exclude them from the ratio; report the count separately.
            alive = emb_norms > 1e-3
            metrics["emb_n_dead"] = (~alive).sum().float()
            if alive.any():
                alive_norms = emb_norms[alive]
                metrics["emb_norm_ratio"] = alive_norms.max() / alive_norms.min().clamp(min=1e-6)
            else:
                metrics["emb_norm_ratio"] = torch.zeros((), device=emb.device)

        # ----- Total -----
        # NOTE: the old code read `self.lambda_x or 1.0` here, so a 0 in the config
        # silently became 1.0. The defaults are the same (1.0 / 1.0 / 0.001), but a
        # zero now means zero, otherwise loss-weight ablations are impossible.
        lambda_ce = self.lambda_ce
        lambda_mse = self.lambda_mse
        lambda_grammar = self.lambda_grammar

        if step < self.decoder_pretrain_steps:
            lambda_mse = 0.0

        loss = (
            lambda_mse * mse_loss
            + lambda_ce * ce_loss
            + lambda_mse * mse_loss_t0
            + lambda_grammar * grammar_loss
        )

        metrics.update(
            {
                "loss": loss,
                "mse_loss": mse_loss.detach(),
                "ce_loss": ce_loss.detach(),
                "mse_loss_t0": mse_loss_t0.detach(),
                "grammar_loss": grammar_loss.detach(),
            }
        )
        return metrics


@registry.tasks.register("ar")
class ARTask(Task):
    """Autoregressive baseline. The loss is carried over from ARTrainer.get_loss."""

    def __init__(
        self,
        *,
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        autocast: Callable[[], Any] = nullcontext,
        autocast_scope: str = "loss",
        seq_len: int = 0,
        **kwargs: Any,
    ):
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.autocast = autocast
        self.autocast_scope = autocast_scope
        self.seq_len = seq_len

    def compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:
        token_ids = batch["token_ids"]  # (B, L)
        inp = token_ids[:, :-1]  # (B, L-1)
        tgt = token_ids[:, 1:]  # (B, L-1)

        loss_ctx = self.autocast if self.autocast_scope == "loss" else nullcontext
        forward_ctx = nullcontext if self.autocast_scope == "loss" else self.autocast
        with loss_ctx():
            with forward_ctx():
                logits = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=self.pad_idx,
                label_smoothing=self.label_smoothing,
            )
        with torch.no_grad():
            pred = logits.argmax(-1)
            valid = tgt != self.pad_idx
            acc = ((pred == tgt) & valid).sum().float() / valid.sum().clamp(min=1)
            ppl = loss.detach().exp()
        return {"loss": loss, "token_acc": acc.detach(), "ppl": ppl}
