import numpy as np
import scipy.signal
from typing import Dict, Union, Tuple, Optional, List, Callable
import torch
import torch.nn as nn


def combined_shape(length: int, shape: Optional[int] = None) -> Tuple:
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape) 
    else:
        return (length, *shape) # type: ignore

def mlp(sizes: List, activation: Callable, output_activation: Callable = nn.Identity) -> Callable:
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module: nn.Module) -> int:
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, 
                obs_dim: int, 
                act_dim: int, 
                hidden_sizes: Tuple, 
                activation: Callable, 
                act_offset: torch.Tensor, 
                act_limit: torch.Tensor
                ) -> None:
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit
        self.act_offset = act_offset

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Return output from network scaled to action space limits.
        return (self.act_limit * self.pi(obs)) + self.act_offset
        #return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, 
                obs_dim: int, 
                act_dim: int, 
                hidden_sizes: Tuple, 
                activation: Callable
                ) -> None:
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, 
                obs_dim: int, 
                act_dim: int, 
                act_offset: torch.Tensor, 
                act_limit: torch.Tensor, 
                hidden_sizes: Tuple = (256,256),
                activation: Callable = nn.ReLU
                ) -> None:
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        # act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_offset, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            #return self.pi(obs).numpy()
            return self.pi(obs).detach().cpu().numpy()
        