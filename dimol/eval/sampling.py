"""Generate SMILES from noise: SDE integration plus logit decoding.

Length conditioning is the one addition to the original scheme, and it is here rather
than in the model because it needs no training. On a padded canvas the objective has a
degenerate optimum: two thirds of the training positions are padding, so a model can
lower its loss by ending sequences early, and the measurements say it does. Runs that
collapse to short strings score high on validity and badly on every distribution metric,
and that trade-off is most of the variance between runs.

Drawing the length from the corpus instead takes the decision away from the model. The
positions past the drawn length are known to be padding, so at every solver step they are
overwritten with their exact conditional value, alpha(t) * pad_embedding + beta(t) * noise,
which is the standard replacement method for conditional diffusion. The model then only
has to fill the canvas it is given.

Iterative refinement is the other addition, and it targets a different failure. The
readout decodes every canvas position independently from the final latent, so anything
that needs several positions to agree is left to chance: aromatic rings need five or six
mutually consistent atoms plus a matched ring digit, and the measurements say the model
produces the right number of rings and the wrong kind, 0.8 aromatic against the corpus
1.9, with no training knob moving it. Refinement re-noises the finished sample part of
the way back and denoises again, so each pass sees the decisions the last one made. It
costs sampling time and no training.

The length pinning below is measured to be nearly free and nearly useless on its own: the model still
ends the molecule early, and pinning the tail changes validity by a point. Forcing it to
fill the length instead, with ``length_floor``, is actively destructive, so the floor is
off by default. Making the model actually use the length needs it as a training input,
which is ``model.length_conditioning``.

The rest of the scheme is carried over unchanged from the old code
(``generate.py::generate`` and the sampling block in
``ConditionalGaussianDenoiserTrainerLite.evaluate``): p_simple -> Euler-Maruyama
over the ts grid -> out_proj -> argmax -> decode_batch(special_decode=True).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from dimol.diffusion.diff_eqs import LearnedScoreSDE
from dimol.diffusion.simulators import EulerMaruyamaSimulator, HeunSimulator
from dimol.models.denoiser import (
    ClampedDenoiserModel,
    DenoiserModel,
    GuidedDenoiserModel,
)
from dimol.training.distributed import unwrap_model


@dataclass
class SamplingParams:
    num_samples: int = 64
    num_timesteps: int = 300
    variance: float = 1.0
    t_start: float = 1e-4
    t_end: float = 0.999
    seed: int = 0
    batch_size: Optional[int] = None
    regime: str = "epsilon"
    progress: bool = False
    clamp_strength: float = 0.0   # pull the x0 estimate onto the nearest token embedding
    clamp_from_alpha: float = 0.5  # only once the estimate carries information
    decode: str = "argmax"        # argmax | grammar (see dimol/eval/decoding.py)
    allowed_brackets: Optional[frozenset] = None  # bracket atoms the corpus contains
    on_disallowed: str = "next_best"  # what to do with an atom outside that set
    strict: bool = False          # full connectivity check instead of bracket counting
    text: Optional[Any] = None    # frozen caption encodings, one row per sample
    text_mask: Optional[Any] = None
    guidance: float = 0.0         # classifier-free guidance scale; 0 is plain conditional
    length_prior: Optional[Any] = None  # 1-d array of token lengths to draw from
    length_exact: bool = False    # use row i of length_prior for sample i, rather than
                                  # drawing from it. Only honest when the length is known
                                  # or predicted; against a benchmark it is an oracle and
                                  # has to be labelled as one
    content_rule: str = "last_positive"  # how to turn the content channel into a stop.
                                  # "first_negative" truncates at the first position below
                                  # zero, which a single noisy position amputates: measured
                                  # MAE 21.49 and 78% of molecules cut short. "last_positive"
                                  # stops one past the last position the channel calls
                                  # content, measured MAE 3.22 on the same model and the
                                  # same samples. "run_3" / "run_5" need that many
                                  # consecutive negatives. Same latent, same weights: the
                                  # rule is worth 18 tokens of length error
    pin_canvas: bool = True       # overwrite every position past the length with the
                                  # padding embedding at each solver step. False still
                                  # gives the model the length as an input but leaves the
                                  # canvas alone, which is the only way to ask what the
                                  # model does about termination by itself: with pinning
                                  # on, the tail is padding by construction and any
                                  # statistic about it is a tautology
    length_floor: bool = False    # also forbid stopping before the drawn length
    refine_rounds: int = 0        # extra denoise-renoise cycles after the trajectory
    refine_t: float = 0.9         # how far back each cycle re-noises to
    refine_steps: int = 20        # solver steps per cycle
    solver: str = "euler_maruyama"  # euler_maruyama | heun. heun re-averages the drift
                                  # at the end of each step, holding the same noise
                                  # increment: second order in the drift for two model
                                  # evaluations per step, so N heun steps cost 2N euler
                                  # ones. Integrates the same SDE; nothing about the
                                  # path, the parameterisation or the loss changes
    time_grid: str = "uniform"    # uniform | data_dense | noise_dense | mid_dense | ends_dense
    time_grid_power: float = 2.0  # how strongly the two dense grids are skewed


def _content_stop(content: torch.Tensor, canvas: int, rule: str) -> torch.Tensor:
    """Turn the content channel into one stop position per sample.

    The channel crosses zero at the end of the molecule, but it is noisy, and how that
    noise is handled dominates the result: on one checkpoint the first-negative rule gave
    a length MAE of 21.49 with 78% of molecules truncated early, while last-positive gave
    3.22 on exactly the same latents.
    """
    negative = content < 0
    batch = content.shape[0]
    full = torch.full((batch,), canvas, dtype=torch.long, device=content.device)

    if rule == "first_negative":
        return torch.where(negative.any(-1), negative.float().argmax(-1), full)

    if rule == "last_positive":
        positive = content > 0
        # One past the last position the channel still calls content.
        flipped = positive.flip(-1).float()
        last = (canvas - 1) - flipped.argmax(-1)
        return torch.where(positive.any(-1), (last + 1).clamp(max=canvas), 
                           torch.zeros_like(full))

    if rule.startswith("run_"):
        k = int(rule.split("_", 1)[1])
        if k > canvas:
            return full
        # A run of k negatives starting at i: all of negative[i:i+k].
        windows = negative.unfold(-1, k, 1).all(-1)
        return torch.where(windows.any(-1), windows.float().argmax(-1), full)

    raise ValueError(
        f"sampling.content_rule={rule!r}; expected 'first_negative', 'last_positive' "
        "or 'run_<k>'"
    )


def _draw_lengths(prior, count: int, canvas: int, generator) -> torch.Tensor:
    """Sample `count` sequence lengths from an empirical length distribution."""
    lengths = torch.as_tensor(prior, dtype=torch.long).flatten()
    if lengths.numel() == 0:
        raise ValueError("the length prior is empty")
    lengths = lengths.clamp(1, canvas)
    idx = torch.randint(0, lengths.numel(), (count,), generator=generator)
    return lengths[idx]


def _time_grid(params: SamplingParams) -> torch.Tensor:
    """The t values the solver stops at, from noise (t_start) to data (t_end).

    The default is the original uniform grid. The others keep the same endpoints and the
    same number of steps but redistribute them, so the solver takes small steps where
    the trajectory moves fastest or where the score is least accurate. Which region that
    is depends on the model, so this is a knob and not a decision.
    """
    n = params.num_timesteps
    u = torch.linspace(0.0, 1.0, n)
    power = max(float(params.time_grid_power), 1e-3)
    kind = params.time_grid
    if kind == "uniform":
        pass
    elif kind == "data_dense":
        u = 1.0 - (1.0 - u) ** power  # small steps near t_end, the data end
    elif kind == "noise_dense":
        u = u**power  # small steps near t_start, the noise end
    elif kind == "mid_dense":
        # small steps in the middle, where a logit-normal training density puts most of
        # its mass, and coarse steps at both ends
        v = 2.0 * u - 1.0
        u = 0.5 + 0.5 * v.sign() * v.abs() ** power
    elif kind == "ends_dense":
        # the mirror image: coarse in the middle, small steps at both ends, where the
        # score is least well trained
        z = torch.erfinv(2.0 * u.clamp(1e-6, 1 - 1e-6) - 1.0) * (2.0**0.5)
        u = torch.sigmoid(z * power)
        u = (u - u[0]) / (u[-1] - u[0])
    else:
        raise ValueError(f"generate.time_grid={kind!r} is unknown")
    return params.t_start + (params.t_end - params.t_start) * u


@torch.no_grad()
def sample_smiles(
    model: torch.nn.Module,
    path: Any,
    tokenizer: Any,
    params: SamplingParams,
    device: str | torch.device,
    latent_sink: Optional[list] = None,
) -> List[str]:
    """Sample molecules. ``latent_sink``, if given, collects the final latent and the
    readout logits of every batch, which is what termination analysis needs: the decoded
    string says which token won, not how close the latent was to padding."""
    from dimol.eval.decoding import build_decoder

    raw = unwrap_model(model)
    get_logits = raw.out_proj
    needs_length = bool(getattr(raw, "length_conditioning", False))
    if needs_length and params.length_prior is None:
        raise ValueError(
            "this model was trained with length conditioning, so generate.length_prior "
            "has to say where the lengths come from"
        )
    decoder = build_decoder(tokenizer, canvas=path.p_simple.shape[0], mode=params.decode,
                            allowed_brackets=params.allowed_brackets,
                            on_disallowed=params.on_disallowed, strict=params.strict)

    text_all = None if params.text is None else torch.as_tensor(params.text)
    mask_all = None
    if text_all is not None:
        mask_all = (torch.ones(text_all.shape[:2], dtype=torch.bool)
                    if params.text_mask is None
                    else torch.as_tensor(params.text_mask).bool())
        if text_all.shape[0] != params.num_samples:
            raise ValueError(
                f"generate.num_samples is {params.num_samples} but {text_all.shape[0]} "
                "captions were given; one caption per sample"
            )

    if text_all is not None:
        score_model = GuidedDenoiserModel(model, path, regime=params.regime,
                                          scale=params.guidance)
    elif params.clamp_strength > 0:
        score_model = ClampedDenoiserModel(
            model, path, regime=params.regime,
            strength=params.clamp_strength, from_alpha=params.clamp_from_alpha,
        )
    else:
        score_model = DenoiserModel(model, path, regime=params.regime)
    sde = LearnedScoreSDE(path, score_model, params.variance)
    if params.solver == "heun":
        # The carry lives on the wrapper, and only DenoiserModel has one to protect.
        carrier = score_model if isinstance(score_model, DenoiserModel) else None
        simulator = HeunSimulator(sde, denoiser=carrier)
    elif params.solver == "euler_maruyama":
        simulator = EulerMaruyamaSimulator(sde)
    else:
        raise ValueError(
            f"sampling.solver={params.solver!r}; expected "
            "'euler_maruyama' or 'heun'"
        )

    canvas = path.p_simple.shape[0]
    pad_idx = int(getattr(raw.config, "pad_idx", 0))
    pad_embedding = None
    pad_variants = int(getattr(raw.config, "pad_variants", 1) or 1)
    pad_variant_ids = None
    if pad_variants > 1:
        base = int(getattr(raw.config, "pad_variant_base", 0) or 0)
        if base <= 0:
            base = int(raw.config.vocab_size) - pad_variants
        pad_variant_ids = list(range(base, base + pad_variants))
        if decoder is not None and hasattr(decoder, "is_stop"):
            for i in pad_variant_ids:
                decoder.is_stop[i] = True
    if params.length_prior is not None:
        pad_embedding = raw.token_embedding.weight[pad_idx].detach()
        if getattr(raw, "content_channel", False):
            # Pinning writes a whole latent, so it needs a value for the content channel
            # too: negative, at the same scale the loss uses, meaning "ended here".
            scale = raw.token_embedding.weight.detach().pow(2).mean().sqrt()
            pad_embedding = torch.cat((pad_embedding, -scale.view(1)))

    batch_size = params.batch_size or params.num_samples
    smiles: List[str] = []
    done = 0
    while done < params.num_samples:
        b = min(batch_size, params.num_samples - done)
        x0 = path.p_simple.sample(b, seed=params.seed + done)
        ts = (
            _time_grid(params)
            .view(1, params.num_timesteps, 1, 1)
            .expand(b, -1, -1, -1)
            .to(device)
        )
        on_step = None
        drawn_lengths = None
        if pad_embedding is not None:
            generator = torch.Generator(device="cpu").manual_seed(params.seed + done)
            if params.length_exact:
                lengths = (torch.as_tensor(params.length_prior, dtype=torch.long)
                           [done : done + b].clamp(1, canvas))
                if lengths.numel() != b:
                    raise ValueError(
                        "generate.length_exact needs one length per sample, and "
                        f"length_prior has {len(params.length_prior)} for "
                        f"{params.num_samples} samples"
                    )
            else:
                lengths = _draw_lengths(params.length_prior, b, canvas, generator)
            drawn_lengths = lengths
            if params.pin_canvas:
                positions = torch.arange(canvas).view(1, canvas)
                known = (positions >= lengths.view(b, 1)).to(device)  # True where padding
                known_mask = known.unsqueeze(-1)
                if pad_variant_ids is not None:
                    # One row per canvas position, cycling through the variants exactly as
                    # the loss assigns them.
                    cyc = [pad_variant_ids[i % len(pad_variant_ids)]
                           for i in range(canvas)]
                    table = raw.token_embedding.weight.detach()
                    pad_target = table[torch.as_tensor(cyc, device=table.device)]
                    if pad_embedding.shape[-1] != pad_target.shape[-1]:
                        extra = pad_embedding[pad_target.shape[-1]:]
                        pad_target = torch.cat(
                            (pad_target, extra.view(1, -1).expand(canvas, -1)), dim=-1)
                    pad_target = pad_target.unsqueeze(0)
                else:
                    pad_target = pad_embedding.view(1, 1, -1)

                def on_step(state, t_next, _mask=known_mask, _pad=pad_target):
                    alpha, beta = path.alpha(t_next), path.beta(t_next)
                    noise = torch.randn_like(state)
                    conditional = alpha * _pad + beta * noise
                    return torch.where(_mask, conditional, state)

                x0 = on_step(x0, ts[:, 0])

        if hasattr(score_model, "reset"):
            score_model.reset()  # self-conditioning must not carry across batches
        if needs_length:
            score_model.length = drawn_lengths.to(device)
        if text_all is not None:
            score_model.text = text_all[done : done + b].to(device).float()
            score_model.text_mask = mask_all[done : done + b].to(device)
        xts = simulator.simulate(x0, ts, use_bar=params.progress, on_step=on_step)

        for _ in range(max(int(params.refine_rounds), 0)):
            # re-noise the finished sample to refine_t and denoise it again, so every
            # position is decoded in the presence of the others' current values
            t_back = torch.full((b, 1, 1), float(params.refine_t), device=device)
            alpha, beta = path.alpha(t_back), path.beta(t_back)
            xts = alpha * xts + beta * torch.randn_like(xts)
            short = (
                torch.linspace(params.refine_t, params.t_end, int(params.refine_steps))
                .view(1, -1, 1, 1)
                .expand(b, -1, -1, -1)
                .to(device)
            )
            if on_step is not None:
                xts = on_step(xts, short[:, 0])
            if hasattr(score_model, "reset"):
                score_model.reset()
            xts = simulator.simulate(xts, short, use_bar=False, on_step=on_step)

        content_end = None
        full_latent = xts
        if getattr(raw, "content_channel", False):
            # The model's own decision about where the molecule stops: the first position
            # whose content channel has gone negative. This is what the channel exists for
            # — a length the model generated and revised at every solver step, rather than
            # one handed to it before the trajectory began.
            emb_dim = raw.token_embedding.weight.shape[-1]
            content = xts[..., emb_dim]
            xts = xts[..., :emb_dim]
            content_end = _content_stop(content, canvas, params.content_rule)

        logits = get_logits(xts)
        if latent_sink is not None:
            # The FULL latent, content channel included: the sink is a diagnostic hook and
            # narrowing it here would hide the one quantity worth inspecting.
            latent_sink.append((full_latent.detach().cpu(), logits.detach().cpu()))
        if decoder is None:
            ids = logits.softmax(-1).argmax(-1).detach().cpu().tolist()
            smiles.extend(tokenizer.decode_batch(ids, special_decode=True))
        else:
            # The floor is off by default and should stay off: forcing content into
            # every position up to the drawn length fills the tail with whatever token
            # ranks first among the non-stop candidates, which measured 0% usable and
            # 60-atom strings. A length-conditioned model is supposed to stop on its own.
            floors = drawn_lengths if params.length_floor else None
            if content_end is not None:
                # Truncate at the model's own stop before decoding, so the channel
                # actually governs the string rather than merely being trained.
                cut = content_end.view(-1, 1).to(logits.device)
                positions = torch.arange(logits.shape[1],
                                         device=logits.device).view(1, -1)
                blocked = (positions >= cut).unsqueeze(-1)
                mask = torch.zeros_like(logits)
                mask[..., pad_idx] = 1.0
                logits = torch.where(blocked, logits * 0.0 + mask * 1e4, logits)
            smiles.extend(decoder.decode(logits.detach(), min_length=floors))
        done += b
    return smiles


def validity(smiles_list: List[str]) -> float:
    """rdkit validity rate: a cheap proxy metric for the training log."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles_list:
        return 0.0
    n_valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return n_valid / len(smiles_list)
