"""Quantum-Born actor-critic: the Born factorization trained as an RL policy.

This is the reinforcement-learning counterpart of the imitation QBDP. Instead of
fitting a Born prior to k-means mode labels and a diffusion decoder to expert
chunks, the same factorization

    pi(a | o) = sum_b |psi_phi(b | o)|^2 * pi_theta(a | o, b)

is optimized to maximize environment return by policy gradient:

  * the discrete mode ``b`` is sampled from the Born distribution
    ``p_phi(b | o) = |psi_phi(b | o)|^2`` -- a quantum-inspired categorical
    *policy* over modes (not a prior fit to labels);
  * the continuous action is produced by a mode-conditioned one-step diffusion
    actor ``pi_theta(a | o, b)``, a Gaussian whose mean is a single reverse
    diffusion estimate -- keeping the diffusion mechanism, now as an RL actor;
  * a value head ``V_w(o)`` provides the critic baseline.

The joint policy log-probability is ``log p_phi(b | o) + log pi_theta(a | o, b)``
because ``(b, a)`` is sampled jointly, so a single PPO ratio updates both levels.

Connection to ``rollout_guided_amplitude_refinement``: the multiplicative update
``|psi'|^2 \\propto |psi|^2 * exp(eta * A)`` is the closed-form mirror-descent
(exponentiated-advantage) version of the Born-policy gradient step performed here
online; see ``qbdp.refinement``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from qbdp.models.born_prior import BornPrior
from qbdp.models.diffusion import DiffusionSchedule, ModeConditionedDenoiser


@dataclass(frozen=True)
class QuantumBornStep:
    """Per-step rollout outputs (batch dimension preserved)."""

    mode: torch.Tensor          # [batch]
    action: torch.Tensor        # [batch, action_dim]  (raw, pre-clip; for log-prob)
    clipped_action: torch.Tensor  # [batch, action_dim]  (sent to the env)
    log_prob: torch.Tensor      # [batch]  joint log pi(b, a | o)
    value: torch.Tensor         # [batch]


class QuantumBornActorCritic(nn.Module):
    """Born discrete policy + mode-conditioned diffusion actor + value critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_modes: int = 4,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        hidden_dim: int = 128,
        diffusion_steps: int = 10,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.schedule = DiffusionSchedule(timesteps=diffusion_steps)

        # Born categorical policy over modes: p_phi(b | o) = |psi_phi(b | o)|^2.
        self.prior = BornPrior(obs_dim, num_modes, hidden_dim)
        # Mode-conditioned one-step diffusion actor (horizon 1 = per-step action).
        self.denoiser = ModeConditionedDenoiser(
            obs_dim, action_dim, horizon=1, num_modes=num_modes, hidden_dim=hidden_dim, time_dim=time_dim
        )
        # Per-dimension Gaussian log-std in the squashed (tanh) action space.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        # Value critic V_w(o).
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        if action_low is None:
            action_low = -torch.ones(action_dim)
        if action_high is None:
            action_high = torch.ones(action_dim)
        # Affine map from tanh output in [-1, 1] to the env action bounds.
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    # ---- Born (high-level) policy -------------------------------------------------
    def born_distribution(self, obs: torch.Tensor) -> Categorical:
        # obs: [batch, obs_dim]
        _, probs = self.prior(obs)
        # probs: [batch, num_modes]  (already sums to 1 via the Born rule)
        return Categorical(probs=probs.clamp_min(1e-8))

    # ---- mode-conditioned diffusion (low-level) actor -----------------------------
    def _action_mean(self, obs: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim], mode: [batch]
        batch = obs.shape[0]
        timestep = torch.full((batch,), self.schedule.timesteps - 1, device=obs.device, dtype=torch.long)
        noisy_action = torch.zeros(batch, 1, self.action_dim, device=obs.device)
        # noisy_action: [batch, 1, action_dim]
        pred_noise = self.denoiser(noisy_action, timestep, obs, mode)
        # pred_noise: [batch, 1, action_dim]
        _, alpha_bars = self.schedule.tensors(obs.device)
        alpha_bar = alpha_bars[timestep].view(batch, 1, 1)
        denoised = (noisy_action - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        # denoised: [batch, 1, action_dim]
        squashed = torch.tanh(denoised.squeeze(1))
        # squashed: [batch, action_dim] in [-1, 1]
        return self.action_bias + self.action_scale * squashed

    def action_distribution(self, obs: torch.Tensor, mode: torch.Tensor) -> Normal:
        # obs: [batch, obs_dim], mode: [batch]
        mean = self._action_mean(obs, mode)
        # mean: [batch, action_dim]
        std = self.log_std.exp() * self.action_scale
        # std: [action_dim] -> broadcast to [batch, action_dim]
        return Normal(mean, std.expand_as(mean))

    # ---- rollout / update interfaces ---------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> QuantumBornStep:
        # obs: [batch, obs_dim]
        born = self.born_distribution(obs)
        mode = born.probs.argmax(dim=-1) if deterministic else born.sample()
        # mode: [batch]
        log_prob_mode = born.log_prob(mode)
        # log_prob_mode: [batch]
        dist = self.action_distribution(obs, mode)
        raw_action = dist.mean if deterministic else dist.sample()
        # raw_action: [batch, action_dim]
        log_prob_action = dist.log_prob(raw_action).sum(dim=-1)
        # log_prob_action: [batch]
        clipped = torch.max(torch.min(raw_action, self.action_bias + self.action_scale),
                            self.action_bias - self.action_scale)
        value = self.critic(obs).squeeze(-1)
        # value: [batch]
        return QuantumBornStep(
            mode=mode,
            action=raw_action,
            clipped_action=clipped,
            log_prob=log_prob_mode + log_prob_action,
            value=value,
        )

    def evaluate_actions(
        self, obs: torch.Tensor, mode: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (joint log_prob, entropy, value) for stored (mode, action)."""
        # obs: [batch, obs_dim], mode: [batch], action: [batch, action_dim]
        born = self.born_distribution(obs)
        log_prob_mode = born.log_prob(mode)
        dist = self.action_distribution(obs, mode)
        log_prob_action = dist.log_prob(action).sum(dim=-1)
        # entropy: Born categorical (mode exploration) + Gaussian (action exploration)
        entropy = born.entropy() + dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        # all: [batch]
        return log_prob_mode + log_prob_action, entropy, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim] -> [batch]
        return self.critic(obs).squeeze(-1)
