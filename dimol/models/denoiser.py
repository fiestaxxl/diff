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
