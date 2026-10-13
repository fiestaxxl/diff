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
        caption_dropout: float = 0.1,
        gate_mode: str = "threshold",
        gate_fraction: float = 0.41,
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
        # Classifier-free guidance is bought here: on this share of examples the caption
        # is masked out, so one set of weights learns both the conditional and the
        # unconditional score and sampling can extrapolate between them. Per example
        # rather than per step, so every batch trains both.
        if not 0.0 <= float(caption_dropout) < 1.0:
            raise ValueError(f"loss.caption_dropout={caption_dropout!r} must be in [0, 1)")
        self.caption_dropout = float(caption_dropout)
        # The cross-entropy and the reconstruction term run only on the low-noise part of
        # the batch, selected by alpha > threshold. How much of a batch that is depends on
        # the timestep draw, so it varies from step to step and shifts when the timestep
        # density changes: under uniform time about 41% of samples qualify, under
        # logit-normal about 36%, and the count fluctuates around that. "topk" instead
        # takes a fixed share of every batch, the highest-alpha samples, which makes the
        # amount of token-level supervision per step deterministic. It is the remaining
        # suspect for the run-to-run spread.
        if gate_mode not in ("threshold", "topk"):
            raise ValueError(f"loss.gate_mode={gate_mode!r}; expected 'threshold' or 'topk'")
        self.gate_mode = gate_mode
        if not 0.0 < float(gate_fraction) <= 1.0:
            raise ValueError(f"loss.gate_fraction={gate_fraction!r} must be in (0, 1]")
        self.gate_fraction = float(gate_fraction)
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
    def _gate(self, alpha_flat: torch.Tensor, threshold: float) -> torch.Tensor:
        """Which samples of the batch the token-level terms run on, as a boolean mask."""
        if self.gate_mode == "threshold":
            return alpha_flat > threshold
        count = max(1, int(round(self.gate_fraction * alpha_flat.numel())))
        mask = torch.zeros_like(alpha_flat, dtype=torch.bool)
        mask[torch.topk(alpha_flat, count).indices] = True
        return mask

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

        # Distinct padding: spread the padding class over several ids, assigned cyclically
        # by position, so the diffusion is not asked to produce one identical value at most
        # of the canvas. token_ids is replaced here, so both the diffusion target and the
        # cross-entropy target see the variants.
        variants = int(getattr(raw.config, "pad_variants", 1) or 1)
        if variants > 1:
            if pad_mask is None:
                raise ValueError(
                    "model.pad_variants > 1 needs 'attention_mask' in the batch: the "
                    "variants are assigned to the positions that are padding"
                )
            base = int(getattr(raw.config, "pad_variant_base", 0) or 0)
            if base <= 0:
                base = int(raw.config.vocab_size) - variants
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            cyclic = base + (positions % variants)
            token_ids = torch.where(pad_mask.bool(), token_ids,
                                    cyclic.unsqueeze(0).expand_as(token_ids))

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

        # The content channel, when the model has one: positive where the molecule
        # continues, negative past its end, diffused jointly with the embedding channels.
        # It is scaled to the embedding table's own per-element RMS because that table
        # grows about fiftyfold over a run, and a fixed +/-1 would start commensurate and
        # end negligible — the channel would quietly stop mattering as training went on.
        content_target = None
        if getattr(raw, "content_channel", False):
            if pad_mask is None:
                raise ValueError(
                    "model.content_channel needs 'attention_mask' in the batch: the "
                    "channel's target is which positions hold content"
                )
            scale = token_embeddings.detach().pow(2).mean().sqrt().clamp(min=1e-6)
            content_target = (pad_mask.to(x0.dtype) * 2.0 - 1.0).unsqueeze(-1) * scale
            x0 = torch.cat((x0, content_target), dim=-1)

        batch_size, seq_len, latent_dim = x0.shape

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
        raw_model = unwrap_model(model)

        # The caption, with classifier-free guidance dropout applied once so that the
        # self-conditioning pre-pass and the real pass see the same conditioning. The
        # task is not an nn.Module, so "are we training" comes from the model.
        text_kwargs: Dict[str, torch.Tensor] = {}
        if int(getattr(raw_model.config, "text_dim", 0) or 0):
            caption_ids = batch.get("caption_ids")
            has_encoder = getattr(raw_model, "text_encoder", None) is not None
            if caption_ids is not None and not has_encoder:
                raise ValueError(
                    "the batch carries 'caption_ids' but the model has no caption "
                    "encoder; set model.text_encoder, or use the smiles_text_npy dataset "
                    "and condition on the precomputed states"
                )
            if has_encoder and caption_ids is None:
                raise ValueError(
                    "the model has a caption encoder but the batch carries no "
                    "'caption_ids'; use the smiles_caption_npy dataset. An encoder fed "
                    "nothing gets no gradient, and a parameter with no gradient deadlocks "
                    "DDP at find_unused_parameters=false"
                )
            if caption_ids is not None:
                caption_attn = batch.get("caption_attn")
                if caption_attn is None:
                    caption_attn = torch.ones(caption_ids.shape, dtype=torch.bool,
                                              device=caption_ids.device)
                text_mask = caption_attn.bool()
                # The same draw, of the same shape, in the same place as the branch below:
                # an intervention that consumes randomness differently is not a controlled
                # comparison, and this project has already paid for that once.
                if raw_model.training and self.caption_dropout > 0:
                    keep = (torch.rand(caption_ids.shape[0], device=caption_ids.device)
                            >= self.caption_dropout)
                    text_mask = text_mask & keep[:, None]
                # The dropout lands on text_mask and never on caption_attn. What has to
                # stop seeing the caption is the cross-attention, not the encoder that
                # produces it: an all-False row into BERT is a softmax over nothing.
                text_kwargs = {"caption_ids": caption_ids, "caption_attn": caption_attn,
                               "text_mask": text_mask}
            else:
                text = batch.get("text")
                if text is None:
                    raise ValueError(
                        "the model has a text path but the batch carries no 'text'; "
                        "use the smiles_text_npy dataset"
                    )
                text_mask = batch.get("text_mask")
                if text_mask is None:
                    text_mask = torch.ones(text.shape[:2], dtype=torch.bool,
                                           device=text.device)
                if raw_model.training and self.caption_dropout > 0:
                    keep = (torch.rand(text.shape[0], device=text.device)
                            >= self.caption_dropout)
                    text_mask = text_mask.bool() & keep[:, None]
                text_kwargs = {"text": text, "text_mask": text_mask}

        x0_self = None
        if getattr(raw_model, "self_conditioning", False) and self.self_cond_prob > 0:
            # Two things here are about DDP, and both were bugs.
            #
            # The decision has to be identical on every rank. Drawing it locally made
            # some ranks run the extra pass and others not, which is a structural
            # divergence inside one iteration. Seeding a CPU generator with the step
            # gives every rank the same coin without any communication.
            #
            # And the extra pass must go through the unwrapped module, never the DDP
            # wrapper. DDP prepares its gradient reducer on every forward; a second
            # forward in one iteration leaves the buckets out of step with the other
            # ranks and the next iteration deadlocks in the all-reduce. It cost a
            # 400-step benchmark that logged step 1 and then hung forever.
            coin = torch.rand((), generator=torch.Generator().manual_seed(int(step)))
            if float(coin) < self.self_cond_prob:
                pre_kwargs = {"input_embeddings": x, "time": time,
                              "attention_mask": attn_mask}
                if getattr(raw_model, "length_conditioning", False):
                    pre_kwargs["length"] = pad_mask.long().sum(-1)
                pre_kwargs.update(text_kwargs)
                with torch.no_grad(), forward_ctx():
                    first = raw_model(**pre_kwargs)
                alpha_first = alpha.clamp(min=self.alpha_eps)
                if self.regime == "epsilon":
                    x0_self = ((x - beta * first) / alpha_first).detach()
                else:
                    x0_self = first.detach()

        forward_kwargs = {"input_embeddings": x, "time": time, "attention_mask": attn_mask}
        forward_kwargs.update(text_kwargs)
        if getattr(raw_model, "length_conditioning", False):
            if pad_mask is None:
                raise ValueError(
                    "model.length_conditioning needs 'attention_mask' in the batch"
                )
            forward_kwargs["length"] = pad_mask.long().sum(-1)
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

        # The token-level terms below read the embedding channels only: the content
        # channel is not a token and the readout has no row for it.
        if content_target is not None:
            x0_hat = x0_hat[..., : token_embeddings.shape[-1]]

        # ----- Reconstruction MSE (only at low-noise / high-alpha steps) -----
        alpha_for_gate = alpha.squeeze(-1).squeeze(-1)  # (B,)
        mse_t0_sample_mask = self._gate(alpha_for_gate, self.mse_t0_alpha_threshold)  # (B,)
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
        # x0 carries the content channel when the model has one; the readout does not
        # know about it, so the token terms see the embedding channels only. x0_hat was
        # already narrowed above.
        x0_tokens = x0 if content_target is None else x0[..., : token_embeddings.shape[-1]]
        logits = get_logits(x0_tokens if self.ce_input == "x0" else x0_hat)  # (B, L, V)

        ce_sample_mask = self._gate(alpha_for_gate, self.ce_alpha_threshold)  # (B,)

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
        caption_dropout: float = 0.0,
        autocast: Callable[[], Any] = nullcontext,
        autocast_scope: str = "loss",
        seq_len: int = 0,
        **kwargs: Any,
    ):
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing
        self.caption_dropout = float(caption_dropout)
        self.autocast = autocast
        self.autocast_scope = autocast_scope
        self.seq_len = seq_len

    def compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:
        token_ids = batch["token_ids"]  # (B, L)
        inp = token_ids[:, :-1]  # (B, L-1)
        tgt = token_ids[:, 1:]  # (B, L-1)

        # A caption-conditioned AR model is p(SMILES | caption); an unconditional one
        # ignores the batch's caption entirely. Which one this is comes from the model.
        raw = unwrap_model(model)
        text_kwargs: Dict[str, torch.Tensor] = {}
        if int(getattr(raw.config, "text_dim", 0) or 0):
            text = batch.get("text")
            if text is None:
                raise ValueError(
                    "the AR model has a text path but the batch carries no 'text'; "
                    "use the smiles_text_npy dataset"
                )
            mask = batch.get("text_mask")
            if mask is None:
                mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            if raw.training and self.caption_dropout > 0:
                keep = (torch.rand(text.shape[0], device=text.device)
                        >= self.caption_dropout)
                mask = mask.bool() & keep[:, None]
            text_kwargs = {"text": text, "text_mask": mask}

        loss_ctx = self.autocast if self.autocast_scope == "loss" else nullcontext
        forward_ctx = nullcontext if self.autocast_scope == "loss" else self.autocast
        with loss_ctx():
            with forward_ctx():
                logits = model(inp, **text_kwargs)
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


@registry.tasks.register("dpo")
class DiffusionDPOTask(Task):
    """Preference optimisation over the diffusion objective.

    Why this exists. The diffusion loss is an MSE in latent space, and across eleven
    configurations it correlated with sample quality at r = +0.46 -- the wrong sign. The
    quantity being minimised is not the quantity being measured. Preference optimisation
    closes that gap without needing the metric to be differentiable: it only needs to know
    which of two samples was better.

    Why exact match drives the preference. A caption-retrieval baseline that generates
    nothing reaches BLEU 0.786, MACCS 0.785 and FCD 0.48 on ChEBI-20, above the published
    MolT5-base row on each, while scoring 0.004 on exact match. Fingerprint similarity is
    reachable by lookup; exact match is not. So the reward that builds these pairs is
    dominated by exact match, and this loss is how that reward reaches the weights.

    The loss follows Diffusion-DPO: one shared timestep and one shared noise draw for the
    winner and the loser, so the comparison is not confounded by sampling different
    diffusion states for the two; the per-sample MSE of each under the trained model is
    measured against the same quantity under a frozen reference; and the difference of
    those differences goes through a log-sigmoid.

        inner = -beta * [ (mse_w - mse_w_ref) - (mse_l - mse_l_ref) ]
        loss  = -log sigmoid(inner)

    The reference model is the checkpoint training started from, frozen. Without it the
    loss would reward making the winner likely in absolute terms, which is ordinary
    fine-tuning and drifts; the reference is what keeps the update a preference.
    """

    def __init__(
        self,
        path,
        *,
        reference_model: Optional[nn.Module] = None,
        beta: float = 2000.0,
        pad_idx: int = 0,
        regime: str = "epsilon",
        x0_noise_std: float = 0.25,
        caption_dropout: float = 0.0,
        t_eps: float = 5e-5,
        autocast: Callable[[], Any] = nullcontext,
        autocast_scope: str = "loss",
        seq_len: int = 0,
        **kwargs: Any,
    ):
        self.path = path
        self.reference_model = reference_model
        # beta is large because the MSEs it multiplies are differences of per-sample
        # means over a 32-dimensional latent, which are small numbers; the published
        # Diffusion-DPO value for images is of the same order for the same reason.
        self.beta = float(beta)
        self.pad_idx = pad_idx
        self.regime = regime
        self.x0_noise_std = x0_noise_std
        self.caption_dropout = float(caption_dropout)
        self.t_eps = t_eps
        self.autocast = autocast
        self.autocast_scope = autocast_scope
        self.seq_len = seq_len

    def set_reference(self, model: nn.Module) -> None:
        """Freeze a copy of the starting weights as the reference."""
        self.reference_model = model
        for p in self.reference_model.parameters():
            p.requires_grad_(False)
        self.reference_model.eval()

    def _noised(self, raw, token_ids, t, noise):
        embeddings = raw.token_embedding(token_ids)
        x0 = embeddings + torch.randn_like(embeddings) * self.x0_noise_std
        alpha, beta = self.path.alpha(t), self.path.beta(t)
        return alpha * x0 + beta * noise

    @staticmethod
    def _per_sample_mse(prediction, target):
        return ((prediction - target) ** 2).mean(dim=(-1, -2))

    def compute_loss(
        self, model: nn.Module, batch: Dict[str, torch.Tensor], step: int = 0
    ) -> Dict[str, torch.Tensor]:
        if self.reference_model is None:
            raise ValueError(
                "the DPO task needs a frozen reference model; call set_reference() with "
                "a copy of the weights training started from"
            )
        raw = unwrap_model(model)
        accepted = batch["token_ids"]
        rejected = batch["rejected_token_ids"]
        accepted_mask = batch.get("attention_mask")
        rejected_mask = batch.get("rejected_attention_mask")

        batch_size = accepted.shape[0]
        device = accepted.device
        # One timestep and one noise draw shared by both members of the pair: the
        # comparison must not be confounded by evaluating them at different states.
        t = torch.rand((batch_size, 1, 1), device=device) * (1 - 2 * self.t_eps) + self.t_eps

        text_kwargs: Dict[str, torch.Tensor] = {}
        if int(getattr(raw.config, "text_dim", 0) or 0):
            text = batch.get("text")
            if text is None:
                raise ValueError("the model has a text path but the batch carries no 'text'")
            mask = batch.get("text_mask")
            if mask is None:
                mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            if raw.training and self.caption_dropout > 0:
                keep = (torch.rand(text.shape[0], device=text.device)
                        >= self.caption_dropout)
                mask = mask.bool() & keep[:, None]
            text_kwargs = {"text": text, "text_mask": mask}

        forward_ctx = nullcontext if self.autocast_scope == "loss" else self.autocast
        losses = {}
        for name, ids, pad in (("w", accepted, accepted_mask),
                               ("l", rejected, rejected_mask)):
            noise = torch.randn(
                (batch_size, ids.shape[1], raw.config.emb_dim), device=device)
            x = self._noised(raw, ids, t, noise)
            kwargs = dict(input_embeddings=x, time=t.squeeze(-1), **text_kwargs)
            if getattr(raw, "length_conditioning", False):
                if pad is None:
                    raise ValueError("length conditioning needs the pair's attention masks")
                kwargs["length"] = pad.long().sum(-1)
            with forward_ctx():
                prediction = model(**kwargs)
                with torch.no_grad():
                    reference = self.reference_model(**kwargs)
            losses[name] = self._per_sample_mse(prediction, noise)
            losses[name + "_ref"] = self._per_sample_mse(reference, noise)

        margin = ((losses["w"] - losses["w_ref"])
                  - (losses["l"] - losses["l_ref"]))
        loss = -F.logsigmoid(-self.beta * margin).mean()

        with torch.no_grad():
            # The share of pairs the model already orders correctly: the honest progress
            # signal here, since the loss value alone says little.
            ordered = (losses["w"] - losses["w_ref"]
                       < losses["l"] - losses["l_ref"]).float().mean()
        return {
            "loss": loss,
            "token_acc": ordered,
            "margin": margin.mean().detach(),
            "mse_w": losses["w"].mean().detach(),
            "mse_l": losses["l"].mean().detach(),
        }
