"""Score wrapper: eps/x prediction -> score, for use inside the SDE.

The denoiser and score formulas are carried over VERBATIM from
dimol/models/models.py.
"""
import torch
import torch.nn as nn


class DenoiserModel(nn.Module):
    """eps/x prediction turned into a score, plus the self-conditioning carry.

    A self-conditioned model wants its own previous estimate of x0 as a second input. The
    solver calls this wrapper once per step in order, so the estimate is simply kept here
    between calls, starting from zeros. ``reset()`` clears it, which matters when the same
    wrapper is reused for another batch.
    """

    def __init__(self, eps_model, path, regime='epsilon'):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime
        self._x0_self = None
        self.length = None  # set by the sampler when the model is length-conditioned
        # A second-order solver evaluates the model twice per step, the second time at a
        # point it may discard. That call must read the carry and must not write it, or
        # the next accepted step is self-conditioned on an off-trajectory estimate.
        self.freeze_carry = False

    def reset(self):
        self._x0_self = None

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t  = torch.clamp(self.path.beta(t),  min=1e-3)
        t_in    = t.squeeze(-1)
        raw = unwrap_model(self.eps_model)
        if getattr(raw, "self_conditioning", False):
            kwargs["x0_self"] = self._x0_self
        if getattr(raw, "length_conditioning", False) and self.length is not None:
            kwargs["length"] = self.length
        pred    = self.eps_model(x, t_in, **kwargs)

        if self.regime == 'epsilon':
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == 'x':
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        if not self.freeze_carry:
            self._x0_self = x0_pred.detach()
        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score


class ClampedDenoiserModel(nn.Module):
    """Denoiser whose x0 estimate is snapped towards the nearest token embedding.

    The clamping trick from Diffusion-LM: the reverse process drifts through a
    continuous space, but only the points that are token embeddings decode to anything,
    so pulling the x0 estimate onto that set at every step keeps the trajectory on the
    data manifold. The score formula is unchanged, only the x0 estimate it is built from.

    ``strength`` is how far to pull (1.0 = fully onto the embedding), and clamping is
    applied only once alpha exceeds ``from_alpha``, because early in the reverse process
    the estimate carries no information and snapping it would inject noise.
    """

    def __init__(self, eps_model, path, regime="epsilon", strength=1.0, from_alpha=0.5):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime
        self.strength = float(strength)
        self.from_alpha = float(from_alpha)

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        raw = unwrap_model(self.eps_model)
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t = torch.clamp(self.path.beta(t), min=1e-3)
        t_in = t.squeeze(-1)
        pred = self.eps_model(x, t_in, **kwargs)

        if self.regime == "epsilon":
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == "x":
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime to be 'epsilon' or 'x', got {self.regime}")

        if self.strength > 0:
            ids = raw.out_proj(x0_pred).argmax(-1)
            snapped = raw.token_embedding(ids)
            gate = (alpha_t > self.from_alpha).to(x0_pred.dtype)
            x0_pred = x0_pred + gate * self.strength * (snapped - x0_pred)

        score = (alpha_t * x0_pred - x) / (beta_t ** 2)
        return score


class GuidedDenoiserModel(nn.Module):
    """Classifier-free guidance: two forwards per step, extrapolated apart.

    The conditional prediction says where the caption points, the unconditional one says
    where the data manifold is anyway, and the difference is the part the caption is
    responsible for. Pushing along that difference sharpens the sample towards the text at
    the cost of diversity, which is the standard trade and the reason guidance exists.

    The unconditional pass is the same weights with the caption masked out, which is
    exactly what caption dropout trained: one model, two behaviours, no second network.

    ``scale`` of 0 reproduces the conditional model, so the guided and unguided paths are
    the same code and a sweep starts from the unguided point.
    """

    def __init__(self, eps_model, path, regime="epsilon", scale=1.0):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.regime = regime
        self.scale = float(scale)
        self._x0_self = None
        self.length = None
        self.text = None
        self.text_mask = None

    def reset(self):
        self._x0_self = None

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        raw = unwrap_model(self.eps_model)
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t = torch.clamp(self.path.beta(t), min=1e-3)
        t_in = t.squeeze(-1)

        shared = dict(kwargs)
        if getattr(raw, "self_conditioning", False):
            shared["x0_self"] = self._x0_self
        if getattr(raw, "length_conditioning", False) and self.length is not None:
            shared["length"] = self.length

        conditional = self.eps_model(x, t_in, text=self.text,
                                     text_mask=self.text_mask, **shared)
        if self.scale == 0.0:
            pred = conditional
        else:
            # the same weights with every caption token masked: the unconditional score
            empty = torch.zeros_like(self.text_mask)
            unconditional = self.eps_model(x, t_in, text=self.text,
                                           text_mask=empty, **shared)
            pred = unconditional + (1.0 + self.scale) * (conditional - unconditional)

        if self.regime == "epsilon":
            x0_pred = (x - beta_t * pred) / alpha_t
        elif self.regime == "x":
            x0_pred = pred
        else:
            raise ValueError(f"Expected regime 'epsilon' or 'x', got {self.regime}")

        self._x0_self = x0_pred.detach()
        return (alpha_t * x0_pred - x) / (beta_t ** 2)


class ARGuidedDenoiserModel(nn.Module):
    """Score guided by a caption-conditioned autoregressive model.

    The measured case for this. Six attempts to make the diffusion itself use the caption
    better -- a content channel in the latent, global conditioning through adaLN, starting
    the trajectory from a retrieved molecule, task-adapted caption features, distinct
    padding, and preference optimisation -- all landed inside seed noise or below it. One
    intervention worked: ranking several finished samples by log p(SMILES | caption) under
    a small autoregressive decoder, which took exact match from 0.052 to 0.121. So the
    autoregressive model holds caption information the diffusion does not use, and it was
    only ever consulted after the trajectory had finished, when nothing could be changed.

    This consults it during the trajectory instead. At each solver step the current x0
    estimate goes through the readout to a distribution over tokens; that distribution is
    fed to the autoregressive model as a SOFT embedding, p @ token_embedding, which keeps
    the whole path differentiable back to the latent; and the gradient of the caption
    log-likelihood is added to the score.

        score_guided = score(x, t) + scale * grad_x log p_ar(tokens(x) | caption)

    This is the same shape as the energy guidance in the predecessor work, where the guide
    was a hand-built grammar energy worth about seven points of validity. Here the guide is
    a learned conditional likelihood, and it is aimed at exact match -- the one metric a
    caption-retrieval baseline cannot reach, and therefore the one that measures generation
    rather than lookup.

    ``start_alpha`` keeps the guidance off early in the reverse process, where the x0
    estimate carries no information and its token distribution is nearly uniform, so the
    gradient would be noise.
    """

    def __init__(self, base, ar_model, *, scale=1.0,
                 start_alpha=0.5, temperature=1.0):
        super().__init__()
        self.base = base
        self.ar_model = ar_model
        for p in self.ar_model.parameters():
            p.requires_grad_(False)
        self.ar_model.eval()
        self.scale = float(scale)
        self.start_alpha = float(start_alpha)
        self.temperature = float(temperature)
        self.report = None   # set to a list to collect (|score|, |grad|, alpha) per step

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    # The sampler sets these per batch on whatever object it was handed, so they are
    # forwarded to the base model rather than stored here: holding the whole split would
    # pair a 250-molecule batch with 3,202 captions.
    @property
    def length(self):
        return getattr(self.base, "length", None)

    @length.setter
    def length(self, value):
        self.base.length = value

    @property
    def text(self):
        return getattr(self.base, "text", None)

    @text.setter
    def text(self, value):
        self.base.text = value

    @property
    def text_mask(self):
        return getattr(self.base, "text_mask", None)

    @text_mask.setter
    def text_mask(self, value):
        self.base.text_mask = value

    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        score = self.base(x, t, **kwargs)
        alpha = self.base.path.alpha(t)
        if float(alpha.mean()) < self.start_alpha or self.scale == 0.0:
            return score

        raw = unwrap_model(self.base.eps_model)
        beta = torch.clamp(self.base.path.beta(t), min=1e-3)
        alpha_c = torch.clamp(alpha, min=1e-3)

        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            # The base model's own x0 estimate, rebuilt from the prediction at x_grad so
            # the gradient reaches the latent.
            forward_kwargs = dict(kwargs)
            if self.text is not None:
                forward_kwargs.setdefault("text", self.text)
                forward_kwargs.setdefault("text_mask", self.text_mask)
            if getattr(raw, "length_conditioning", False) and self.length is not None:
                forward_kwargs.setdefault("length", self.length)
            pred = raw(x_grad, t.squeeze(-1), **forward_kwargs)
            if self.base.regime == "epsilon":
                x0 = (x_grad - beta * pred) / alpha_c
            else:
                x0 = pred
            emb_dim = raw.token_embedding.weight.shape[-1]
            logits = raw.out_proj(x0[..., :emb_dim])
            probs = torch.softmax(logits / self.temperature, dim=-1)

            ar = unwrap_model(self.ar_model)
            soft = probs @ ar.token_embedding.weight
            ar_logits = ar(inputs_embeds=soft[:, :-1], text=self.text,
                           text_mask=self.text_mask)
            target = probs[:, 1:]
            logp = torch.log_softmax(ar_logits.float(), dim=-1)
            # Expected log-likelihood of the next token under the diffusion's own
            # distribution: a soft version of "how plausible is this string to the
            # autoregressive model, given the caption".
            objective = (target * logp).sum(-1).mean()
            grad = torch.autograd.grad(objective, x_grad)[0]

        guided = score + self.scale * grad.detach()
        if self.report is not None:
            # The score is (alpha * x0 - x) / beta^2, which is enormous at small beta,
            # so the guidance scale has to be read against it rather than guessed. Without
            # this the first sweep ran at scales 0 to 100 and produced bit-identical
            # samples, because the term contributed nothing.
            self.report.append((float(score.norm()), float(grad.norm()),
                                float(alpha.mean())))
        return guided


class ARClampedDenoiserModel(nn.Module):
    """Clamping the x0 estimate to the token an autoregressive decoder prefers.

    The measurement this is built on. Ranking finished samples by log p(SMILES | caption)
    under a small autoregressive decoder takes exact match from 0.052 to 0.121. Feeding the
    same model's gradient into the reverse SDE, at a magnitude equal to the diffusion
    score's own norm, changes 24% of molecules and improves nothing. The difference is what
    the autoregressive model is shown: it was trained on discrete, complete token
    sequences, and a soft mixture ``p @ E`` is off its manifold. It is an excellent judge
    of strings and a useless judge of latents.

    So this hands it strings. Diffusion-LM clamping already snaps the x0 estimate onto the
    nearest token embedding at every late step -- the one place a continuous sampler
    touches discreteness. Here the snap target is not the nearest embedding but the token
    the autoregressive model prefers, given the draft's own prefix and the caption:

        draft      = argmax(out_proj(x0))
        logits_ar  = ar(draft, caption)            one forward pass, no gradients
        combined   = out_proj(x0) + weight * logits_ar_shifted
        x0         <- token_embedding(argmax(combined))

    The diffusion proposes the whole canvas in parallel, the autoregressive model revises
    it left to right, and the trajectory continues from the revision. That is the structure
    of speculative decoding, applied at every solver step rather than once at the end.

    ``weight`` is in logit units and both terms are log-odds over the same vocabulary, so
    1.0 weighs the two models equally. ``from_alpha`` keeps the revision off while the x0
    estimate carries no signal and its draft would be noise.
    """

    def __init__(self, eps_model, path, ar_model, *, regime="epsilon", weight=1.0,
                 strength=1.0, from_alpha=0.5, pad_idx=0,
                 ar_min_prob=0.0, draft_max_prob=1.0):
        super().__init__()
        self.eps_model = eps_model
        self.path = path
        self.ar_model = ar_model
        for p in self.ar_model.parameters():
            p.requires_grad_(False)
        self.ar_model.eval()
        self.regime = regime
        self.weight = float(weight)
        self.strength = float(strength)
        self.from_alpha = float(from_alpha)
        self.pad_idx = int(pad_idx)
        # Speculative decoding accepts the draft by default and intervenes only where the
        # verifier confidently objects. Forcing every disagreement is not that: measured
        # per-position agreement between these two models on the diffusion's own finished
        # samples is 80.8%, so rewriting every disagreement edits about a fifth of the
        # molecule at EVERY solver step, and the sample collapses -- validity 0.788 to
        # 0.13. These two thresholds restrict the edit to positions where the
        # autoregressive model is confident AND the diffusion readout is not.
        self.ar_min_prob = float(ar_min_prob)
        self.draft_max_prob = float(draft_max_prob)
        self._x0_self = None
        self.length = None
        self.text = None
        self.text_mask = None
        self.revised = 0      # tokens the autoregressive model changed, for reporting
        self.seen = 0

    def reset(self):
        self._x0_self = None

    @torch.no_grad()
    def forward(self, x, t, **kwargs):
        from dimol.training.distributed import unwrap_model

        raw = unwrap_model(self.eps_model)
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-3)
        beta_t = torch.clamp(self.path.beta(t), min=1e-3)
        t_in = t.squeeze(-1)

        if getattr(raw, "self_conditioning", False):
            kwargs["x0_self"] = self._x0_self
        if getattr(raw, "length_conditioning", False) and self.length is not None:
            kwargs["length"] = self.length
        if self.text is not None:
            kwargs.setdefault("text", self.text)
            kwargs.setdefault("text_mask", self.text_mask)

        pred = self.eps_model(x, t_in, **kwargs)
        if self.regime == "epsilon":
            x0 = (x - beta_t * pred) / alpha_t
        else:
            x0 = pred
        self._x0_self = x0.detach()

        if self.weight != 0.0 and float(alpha_t.mean()) > self.from_alpha:
            emb_dim = raw.token_embedding.weight.shape[-1]
            logits = raw.out_proj(x0[..., :emb_dim]).float()
            draft = logits.argmax(-1)

            ar = unwrap_model(self.ar_model)
            ar_logits = ar(draft[:, :-1], text=self.text, text_mask=self.text_mask).float()
            # ar_logits[:, i] predicts position i+1, so position 0 keeps the diffusion's
            # own opinion and the rest are shifted into place.
            shifted = torch.zeros_like(logits)
            shifted[:, 1:] = ar_logits

            combined = logits + self.weight * shifted
            revised = combined.argmax(-1)

            # Accept the draft unless the autoregressive model is confident and the
            # diffusion is not.
            if self.ar_min_prob > 0.0 or self.draft_max_prob < 1.0:
                ar_prob = torch.zeros_like(logits[..., 0])
                ar_prob[:, 1:] = ar_logits.softmax(-1).max(-1).values
                draft_prob = logits.softmax(-1).max(-1).values
                allow = (ar_prob >= self.ar_min_prob) & (draft_prob <= self.draft_max_prob)
                revised = torch.where(allow, revised, draft)

            # Never let the revision invent content where the draft had padding: the
            # length is supplied separately and the canvas tail is not the AR model's to
            # fill.
            revised = torch.where(draft == self.pad_idx, draft, revised)

            self.revised += int((revised != draft).sum())
            self.seen += int(revised.numel())

            snapped = raw.token_embedding(revised)
            if snapped.shape[-1] < x0.shape[-1]:
                snapped = torch.cat((snapped, x0[..., snapped.shape[-1]:]), dim=-1)
            x0 = x0 + self.strength * (snapped - x0)

        return (alpha_t * x0 - x) / (beta_t ** 2)
