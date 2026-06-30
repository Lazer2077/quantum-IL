from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionSchedule:
    timesteps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alpha_bars


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # timestep: [batch]
        half = self.dim // 2
        frequencies = torch.exp(
            torch.linspace(0, -torch.log(torch.tensor(10000.0, device=timestep.device)), half, device=timestep.device)
        )
        angles = timestep.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        # embedding: [batch, time_dim]
        return embedding


class ModeConditionedDenoiser(nn.Module):
    """DDPM noise predictor conditioned on observation and a discrete mode label."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        num_modes: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.mode_embed = nn.Embedding(num_modes, time_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        flat_action_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat_action_dim + obs_dim + 2 * time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def forward(
        self,
        noisy_chunk: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor,
        mode_label: torch.Tensor,
    ) -> torch.Tensor:
        # noisy_chunk: [batch, horizon, action_dim]
        batch = noisy_chunk.shape[0]
        flat_chunk = noisy_chunk.reshape(batch, self.horizon * self.action_dim)
        # flat_chunk: [batch, horizon * action_dim]
        time_features = self.time_embed(timestep)
        # time_features: [batch, time_dim]
        mode_features = self.mode_embed(mode_label)
        # mode_features: [batch, time_dim]
        features = torch.cat([flat_chunk, obs, time_features, mode_features], dim=-1)
        # features: [batch, horizon * action_dim + obs_dim + 2 * time_dim]
        pred_noise = self.net(features).view(batch, self.horizon, self.action_dim)
        # pred_noise: [batch, horizon, action_dim]
        return pred_noise


class StandardDiffusionPolicy(nn.Module):
    """Diffusion-policy baseline without a Born prior or mode mixture."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        flat_action_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat_action_dim + obs_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flat_action_dim),
        )

    def forward(self, noisy_chunk: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # noisy_chunk: [batch, horizon, action_dim]
        batch = noisy_chunk.shape[0]
        flat_chunk = noisy_chunk.reshape(batch, self.horizon * self.action_dim)
        # flat_chunk: [batch, horizon * action_dim]
        time_features = self.time_embed(timestep)
        # time_features: [batch, time_dim]
        features = torch.cat([flat_chunk, obs, time_features], dim=-1)
        # features: [batch, horizon * action_dim + obs_dim + time_dim]
        pred_noise = self.net(features).view(batch, self.horizon, self.action_dim)
        # pred_noise: [batch, horizon, action_dim]
        return pred_noise


def diffusion_loss(
    model: nn.Module,
    clean_chunk: torch.Tensor,
    obs: torch.Tensor,
    schedule: DiffusionSchedule,
    mode_label: torch.Tensor | None = None,
) -> torch.Tensor:
    device = clean_chunk.device
    _, alpha_bars = schedule.tensors(device)
    batch = clean_chunk.shape[0]
    timestep = torch.randint(0, schedule.timesteps, (batch,), device=device)
    noise = torch.randn_like(clean_chunk)
    alpha_bar_t = alpha_bars[timestep].view(batch, 1, 1)
    noisy_chunk = alpha_bar_t.sqrt() * clean_chunk + (1.0 - alpha_bar_t).sqrt() * noise
    if mode_label is None:
        pred_noise = model(noisy_chunk, timestep, obs)
    else:
        pred_noise = model(noisy_chunk, timestep, obs, mode_label)
    return F.mse_loss(pred_noise, noise)
