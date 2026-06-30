from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticConfig:
    num_samples: int = 1024
    obs_dim: int = 8
    action_dim: int = 3
    horizon: int = 4
    num_modes: int = 4
    noise_std: float = 0.08
    seed: int = 7


class ActionChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor-backed dataset with observations, action chunks, and optional modes."""

    def __init__(
        self,
        observations: torch.Tensor,
        action_chunks: torch.Tensor,
        mode_labels: torch.Tensor | None = None,
    ) -> None:
        self.observations = observations.float()
        self.action_chunks = action_chunks.float()
        if mode_labels is None:
            mode_labels = torch.zeros(len(observations), dtype=torch.long)
        self.mode_labels = mode_labels.long()

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.observations[index],
            "action_chunk": self.action_chunks[index],
            "mode_label": self.mode_labels[index],
        }


def make_synthetic_expert_dataset(config: SyntheticConfig) -> ActionChunkDataset:
    """Create a small multimodal expert imitation dataset on CPU."""

    generator = torch.Generator().manual_seed(config.seed)
    obs = torch.randn(config.num_samples, config.obs_dim, generator=generator)

    mode_w = torch.randn(config.obs_dim, config.num_modes, generator=generator)
    logits = obs @ mode_w
    mode_labels = torch.distributions.Categorical(logits=logits).sample()

    time = torch.linspace(0.0, 1.0, config.horizon)
    chunks = []
    for mode in range(config.num_modes):
        phase = (mode + 1) * torch.pi / (config.num_modes + 1)
        basis = torch.stack(
            [
                torch.sin((mode + 1) * torch.pi * time + phase),
                torch.cos((mode + 1) * torch.pi * time),
                time * (mode + 1) / config.num_modes,
            ],
            dim=-1,
        )
        if config.action_dim > 3:
            pad = torch.zeros(config.horizon, config.action_dim - 3)
            basis = torch.cat([basis, pad], dim=-1)
        chunks.append(basis[:, : config.action_dim])
    templates = torch.stack(chunks, dim=0)

    obs_projection = torch.randn(config.obs_dim, config.horizon * config.action_dim, generator=generator)
    obs_effect = 0.15 * torch.tanh(obs @ obs_projection).view(
        config.num_samples, config.horizon, config.action_dim
    )
    action_chunks = templates[mode_labels] + obs_effect
    action_chunks = action_chunks + config.noise_std * torch.randn(
        action_chunks.shape, generator=generator
    )

    return ActionChunkDataset(obs, action_chunks, mode_labels)
