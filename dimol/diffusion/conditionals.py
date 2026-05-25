import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from torch.func import vmap, jacrev

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1)), torch.zeros(1,1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1)), torch.ones(1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1)
    
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1)), torch.ones(1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1)), torch.zeros(1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor, eps = 1e-5) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1)

class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        return torch.ones_like(t)
        
class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """
    def __call__(self, t: torch.Tensor, eps = 1e-5) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return torch.clamp(1-t, min=eps)
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1) (num_samples, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1) (num_samples, 1, 1)
        """ 
        return - torch.ones_like(t)
    
class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return torch.sqrt(1-t)
        raise NotImplementedError("Fill me in for Question 2.1!")

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)



# class CosineAlpha(Alpha):
#     """
#     Implements alpha_t = cos^2(pi/2 * t)
#     """
#     def __call__(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             - t: time (num_samples, 1, 1, 1) or (num_samples, 1, 1)
#         Returns:
#             - alpha_t (same shape as input)
#         """
#         return torch.cos(0.5 * torch.pi * t) ** 2
    
#     def dt(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Evaluates d/dt alpha_t = -pi * cos(pi/2 * t) * sin(pi/2 * t)
#         = -pi/2 * sin(pi * t)
#         Args:
#             - t: time (num_samples, 1, 1, 1) or (num_samples, 1, 1)
#         Returns:
#             - d/dt alpha_t (same shape as input)
#         """
#         return -0.5 * torch.pi * torch.sin(torch.pi * t)

class CosineAlpha:
    """
    alpha(0)=0, alpha(1)=1
    """
    def __init__(self, device):
        self.device = device

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * torch.pi * t).to(self.device)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t).to(self.device)


class CosineBeta:
    """
    beta(0)=1, beta(1)=0
    """
    def __init__(self, device):
        self.device = device
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * torch.pi * t).to(self.device)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t).to(self.device)

# class CosineBeta(Beta):
#     """
#     Implements beta_t = sqrt(1 - cos^2(pi/2 * t)) = sin(pi/2 * t)
#     """
#     def __call__(self, t: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
#         """
#         Args:
#             - t: time (num_samples, 1, 1, 1) or (num_samples, 1, 1)
#             - eps: small epsilon for numerical stability
#         Returns:
#             - beta_t (same shape as input)
#         """
#         return torch.clamp(torch.sin(0.5 * torch.pi * t), min=eps)
    
#     def dt(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Evaluates d/dt beta_t = pi/2 * cos(pi/2 * t)
#         Args:
#             - t: time (num_samples, 1, 1, 1) or (num_samples, 1, 1)
#         Returns:
#             - d/dt beta_t (same shape as input)
#         """
#         return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)