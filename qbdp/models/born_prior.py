from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BornPrior(nn.Module):
    """Classical Born-style discrete prior p_phi(b|o)=|psi_phi(b|o)|^2."""

    def __init__(self, obs_dim: int, num_modes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_modes),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: [batch, obs_dim]
        raw_amplitudes = self.net(obs)
        # raw_amplitudes: [batch, num_modes]
        amplitudes = F.normalize(raw_amplitudes, p=2, dim=-1, eps=1e-8)
        # amplitudes: [batch, num_modes]
        probs = amplitudes.square()
        # probs: [batch, num_modes]
        return amplitudes, probs
