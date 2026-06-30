from __future__ import annotations

import torch
from torch.nn import functional as F


def rollout_guided_amplitude_refinement(
    amplitudes: torch.Tensor,
    returns: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Apply psi_i' = normalize(psi_i * exp(eta * R_i / 2))."""

    # amplitudes: [batch, num_modes] or [num_modes]
    # returns: [batch, num_modes] or [num_modes]
    scaled = amplitudes * torch.exp(0.5 * eta * returns)
    refined = F.normalize(scaled, p=2, dim=-1, eps=1e-8)
    # refined: same shape as amplitudes
    return refined
