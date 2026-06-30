from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim]
        batch = obs.shape[0]
        action_chunk = self.net(obs).view(batch, self.horizon, self.action_dim)
        # action_chunk: [batch, horizon, action_dim]
        return action_chunk


class CVAEActionChunkPolicy(nn.Module):
    """Conditional VAE baseline for action chunks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.latent_dim = latent_dim
        flat_action_dim = horizon * action_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + flat_action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def forward(self, obs: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [batch, obs_dim]
        # action_chunk: [batch, horizon, action_dim]
        batch = obs.shape[0]
        flat_chunk = action_chunk.reshape(batch, self.horizon * self.action_dim)
        # flat_chunk: [batch, horizon * action_dim]
        encoded = self.encoder(torch.cat([obs, flat_chunk], dim=-1))
        # encoded: [batch, hidden_dim]
        mu = self.mu(encoded)
        logvar = self.logvar(encoded).clamp(-8.0, 8.0)
        # mu/logvar: [batch, latent_dim]
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std
        # latent: [batch, latent_dim]
        reconstruction = self.decode(obs, latent)
        # reconstruction: [batch, horizon, action_dim]
        return reconstruction, mu, logvar

    def decode(self, obs: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        # obs: [batch, obs_dim]
        if latent is None:
            latent = torch.randn(obs.shape[0], self.latent_dim, device=obs.device)
        # latent: [batch, latent_dim]
        batch = obs.shape[0]
        flat_chunk = self.decoder(torch.cat([obs, latent], dim=-1))
        # flat_chunk: [batch, horizon * action_dim]
        action_chunk = flat_chunk.view(batch, self.horizon, self.action_dim)
        # action_chunk: [batch, horizon, action_dim]
        return action_chunk


def cvae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> torch.Tensor:
    recon_loss = F.mse_loss(reconstruction, target)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return recon_loss + beta * kl
