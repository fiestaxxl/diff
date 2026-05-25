from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.distributions as D
import math
import numpy as np

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """ 
    @abstractmethod
    def sample(self, num_samples: int, seed: int = 42) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer('dummy', torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
        
    def sample(self, num_samples, seed=42) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_rng = torch.Generator(device=self.dummy.device)
        sample_rng.manual_seed(42 + seed)
        return self.std * torch.randn(num_samples, *self.shape, generator=sample_rng, device=self.dummy.device)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, *self.shape)
        returns: shape (batch, 1)
        """
        D = math.prod(self.shape)
        log_norm = -0.5 * D * math.log(2 * math.pi * self.std ** 2)
        log_exp  = -0.5 * x.flatten(start_dim=1).pow(2).sum(dim=-1) / self.std ** 2
        return (log_norm + log_exp).view(-1, 1)

class GaussianMixture(torch.nn.Module, Sampleable):
    """
    Two-dimensional Gaussian mixture model, and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int, seed: int = 42) -> torch.Tensor:
        # sample_rng = torch.Generator(device=self.device)
        # sample_rng.manual_seed(42 + seed)
        # torch.manual_seed(42+seed)
        return self.distribution.sample(torch.Size((num_samples,)), 
                                                    #generator=sample_rng, 
                                                    ).to(self.device)

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

