from __future__ import annotations

import torch


class KMeansActionTokenizer:
    """Small torch-only K-means tokenizer for action chunks."""

    def __init__(self, num_modes: int, iters: int = 25, seed: int = 0) -> None:
        self.num_modes = num_modes
        self.iters = iters
        self.seed = seed
        self.centroids: torch.Tensor | None = None

    def fit(self, action_chunks: torch.Tensor) -> torch.Tensor:
        flat = action_chunks.reshape(action_chunks.shape[0], -1).float()
        generator = torch.Generator(device=flat.device).manual_seed(self.seed)
        indices = torch.randperm(flat.shape[0], generator=generator, device=flat.device)[: self.num_modes]
        centroids = flat[indices].clone()
        labels = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
        for _ in range(self.iters):
            labels = torch.cdist(flat, centroids).argmin(dim=-1)
            for mode in range(self.num_modes):
                mask = labels == mode
                if mask.any():
                    centroids[mode] = flat[mask].mean(dim=0)
        self.centroids = centroids
        return labels.cpu()

    def predict(self, action_chunks: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise RuntimeError("Tokenizer must be fit before predict().")
        flat = action_chunks.reshape(action_chunks.shape[0], -1).float()
        return torch.cdist(flat, self.centroids.to(flat.device)).argmin(dim=-1).cpu()
