import torch
from dimol.diffusion.diff_eqs import ODE, SDE
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from tqdm.auto import tqdm

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - t: time, shape (bs, 1, 1, 1) (num_samples, 1, 1)
            - dt: time, shape (bs, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w) (num_samples, seq_len, emb_dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_bar = False, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - ts: timesteps, shape (bs, nts, 1, 1, 1) (num_samples, nts, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
        """
        nts = ts.shape[1]
        # An optional hook run after every step, given (x, t_next). It exists for
        # conditional sampling: the caller can overwrite the part of the state it knows,
        # which is how length conditioning and any other inpainting is done.
        on_step = kwargs.pop("on_step", None)

        steps = range(nts - 1)
        for t_idx in tqdm(steps) if use_bar else steps:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            if on_step is not None:
                x = on_step(x, ts[:, t_idx + 1])
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - ts: timesteps, shape (bs, nts, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w) (num_samples, nts, seq_len, emb_dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt,t, **kwargs) * h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)


class HeunSimulator(Simulator):
    """Stochastic Heun: one Euler-Maruyama step, then the drift re-averaged at its end.

    Euler-Maruyama is first order, so its error per step is O(h^2) in the drift. On
    ZINC-250k that did not bind: 100 steps matched 1000. On ChEBI it does — 300 steps beat
    100 by 0.019 MACCS, while 600 added only 0.004 and cost 1.4 points of validity, which
    is what a discretisation-limited sampler trading one error for another looks like.

    The scheme is the standard stochastic Heun (as in EDM): the noise increment is drawn
    once and held fixed while the drift is evaluated at both ends of the step, then
    averaged. Holding the increment is what keeps the corrector a corrector rather than a
    second, independent stochastic step.

        d1 = drift(x, t)
        x~ = x + d1*h + g*sqrt(h)*z
        d2 = drift(x~, t+h)
        x' = x + (d1 + d2)/2*h + g*sqrt(h)*z        # same z

    Two model evaluations per step, so 150 Heun steps cost what 300 Euler steps cost.
    The path, the parameterisation and the loss are untouched: this only changes how the
    same reverse SDE is integrated.

    Safe at this end of the path: the drift carries a 1/alpha term and alpha -> 1 towards
    the data, so the corrector is evaluated where the coefficient is better behaved than
    at the point the predictor started from.
    """

    def __init__(self, sde: SDE, denoiser=None):
        self.sde = sde
        # The self-conditioning carry lives on the denoiser wrapper. The corrector reads
        # it and must not overwrite it; see DenoiserModel.freeze_carry.
        self.denoiser = denoiser

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        d1 = self.sde.drift_coefficient(xt, t, **kwargs)
        g = self.sde.diffusion_coefficient(xt, t, **kwargs)
        increment = g * torch.sqrt(h) * torch.randn_like(xt)
        euler = xt + d1 * h + increment

        frozen = self.denoiser is not None
        if frozen:
            self.denoiser.freeze_carry = True
        try:
            d2 = self.sde.drift_coefficient(euler, t + h, **kwargs)
        finally:
            if frozen:
                self.denoiser.freeze_carry = False
        return xt + 0.5 * (d1 + d2) * h + increment


class EulerMaruyamaSimulatorWithProjection(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)

    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - ts: timesteps, shape (bs, nts, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w) (num_samples, nts, seq_len, emb_dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]

        vocab = kwargs.pop('vocab')
        num_projections = kwargs.pop('num_projections', 0)
        alpha = kwargs.pop('alpha', 1)
        alpha = 0.05
        idx_projections = set([])
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]

            if t_idx in idx_projections:
                logits = self.sde.score_model.get_logits(x)
                # tokens = torch.distributions.Categorical(
                #     logits=logits / 2.0
                # ).sample()
                tokens = logits.argmax(-1)

                x_proj =  self.sde.score_model.embed_tokens(tokens)
                x = alpha * x + (1 - alpha) * x_proj
                
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )




def paren_violation_mask(probs, open_id, close_id):
    p_open = probs[..., open_id]   # (B,L)
    p_close = probs[..., close_id] # (B,L)

    balance = torch.cumsum(p_open - p_close, dim=1)  # (B,L)

    neg_violation = torch.relu(-balance)             # (B,L)
    tail_violation = torch.relu(balance[:, -1:])    # (B,1)

    # broadcast tail to sequence length
    tail_violation = tail_violation.expand(-1, probs.size(1))  # (B,L)

    mask = neg_violation + tail_violation  # (B,L)
    return mask.unsqueeze(-1)              # (B,L,1)

def ring_violation_mask(probs, ring_ids):
    """
    probs: (B, L, V)
    returns: (B, L, 1)
    """
    # total probability mass assigned to ring digits
    p_ring = probs[..., ring_ids].sum(-1)     # (B, L)

    # soft count of rings per sequence
    total = p_ring.sum(dim=1)                 # (B,)

    # odd/even penalty (0 when even)
    oddness = torch.abs(total - total.round())  # (B,)

    # broadcast to all positions
    return oddness[:, None, None].expand(
        probs.size(0), probs.size(1), 1
    )

def transition_violation_mask(probs, bad_T):
    """
    bad_T[i,j] = 1 if i→j illegal
    """
    prev = probs[:, :-1, :]        # (B,L-1,V)
    next = probs[:, 1:, :]

    violation = torch.einsum(
        "blv,vw,blw->bl",
        prev, bad_T, next
    )

    pad = torch.zeros_like(violation[:, :1])
    violation = torch.cat([pad, violation], dim=1)

    return violation.unsqueeze(-1)


def entropy_mask(probs, tau=3.0):
    ent = -(probs * probs.log()).sum(-1)
    return (ent > tau).float().unsqueeze(-1)


def grammar_violation_mask(
    logits,
    vocab
):
    probs = logits.softmax(-1)

    m1 = paren_violation_mask(probs, vocab["("], vocab[")"])
    ring_ids = [vocab[str(i)] for i in range(10) if str(i) in vocab]
    m2 = ring_violation_mask(probs, ring_ids)
    #m3 = transition_violation_mask(probs, vocab["bad_T"])
    m4 = entropy_mask(probs)

    # mask = torch.clamp(m1 + m2 + m3, 0, 1)
    mask = torch.clamp(m1 + m2, 0, 1)

    # gate by entropy
    mask = mask * m4

    return mask
