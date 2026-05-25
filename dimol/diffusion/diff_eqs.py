import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from dimol.diffusion.paths import ConditionalProbabilityPath

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
            - t: time, shape (bs, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w) (num_samples, seq_len, emb_dim)
        """
        pass


   
class LearnedScoreSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, score_model: nn.Module, sigma: float, eps:float = 1e-3):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.score_model = score_model
        self.sigma = sigma
        self.path = path
        self.eps = eps

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        alpha = self.path.alpha(t).clamp(min=self.eps)
        beta = self.path.beta(t)
        dt_alpha_t = self.path.alpha.dt(t) 
        dt_beta_t = self.path.beta.dt(t) 
        return (beta**2 * dt_alpha_t/(alpha) - dt_beta_t * beta + self.diffusion_coefficient(x,t)**2/2 )*self.score_model(x,t) + dt_alpha_t/alpha * x



    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.ones_like(x)#* torch.randn_like(x)


class CFGLearnedScoreSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, score_model: nn.Module, sigma: float, guidance_scale=1.0):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.score_model = score_model
        self.sigma = sigma
        self.path = path
        self.guidance_scale = guidance_scale

    def get_vf_model(self, t, score, x):
        alpha_t = torch.clamp(self.path.alpha(t), min=1e-4) # (num_samples, 1, 1, 1)
        beta_t = self.path.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.path.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.path.beta.dt(t) # (num_samples, 1, 1, 1)

        return (beta_t**2 * dt_alpha_t/alpha_t - beta_t*dt_beta_t)*score + dt_alpha_t/alpha_t*x

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, key_padding_mask) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        alpha = self.path.alpha(t)
        beta = self.path.beta(t)
        dt_alpha_t = self.path.alpha.dt(t) 
        dt_beta_t = self.path.beta.dt(t) 

        guided_score = self.score_model(x, t, text_cond=y, key_padding_mask=key_padding_mask)

        null_text = self.score_model.eps_model.null_text.expand_as(y)
        null_mask = torch.zeros_like(key_padding_mask)
        null_mask[:, 0] = True
        unguided_score = self.score_model(x, t, text_cond=null_text, key_padding_mask=null_mask)
        # unguided_score_field = 0


        mixed_score = (1-self.guidance_scale) * unguided_score + self.guidance_scale * guided_score

        out = self.get_vf_model(t, mixed_score, x) \
        + 0.5 * self.sigma**2 * mixed_score
        
        return out
        # coef_1 =  (beta**2 * dt_alpha_t/(alpha) - dt_beta_t * beta)
        # coef_2 = dt_alpha_t/alpha

        # guided_vector_field = coef_1 * guided_score_field + coef_2 * x
        # unguided_vector_field =  coef_1 * unguided_score_field + coef_2 * x


        # mixed_score_field = (1 - self.guidance_scale) * unguided_score_field + self.guidance_scale * guided_score_field

        # # mixed_vector_field = coef_1 * mixed_score_field + coef_2 * x
        # mixed_vector_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

        # return (mixed_vector_field + self.diffusion_coefficient(x,t)**2/2 * mixed_score_field)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor, y=None, key_padding_mask=None) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.ones_like(x)#* torch.randn_like(x)



class CFGVectorFieldODE(ODE):
    def __init__(self, net, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field