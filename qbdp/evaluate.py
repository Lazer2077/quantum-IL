from __future__ import annotations

import argparse

import torch

from qbdp.data.synthetic import SyntheticConfig, make_synthetic_expert_dataset
from qbdp.models.born_prior import BornPrior
from qbdp.models.diffusion import DiffusionSchedule, ModeConditionedDenoiser


def load_qbdp_checkpoint(
    checkpoint_path: str,
) -> tuple[dict, BornPrior, ModeConditionedDenoiser]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prior = BornPrior(checkpoint["obs_dim"], checkpoint["num_modes"], checkpoint["hidden_dim"])
    denoiser = ModeConditionedDenoiser(
        checkpoint["obs_dim"],
        checkpoint["action_dim"],
        checkpoint["horizon"],
        checkpoint["num_modes"],
        checkpoint["hidden_dim"],
    )
    prior.load_state_dict(checkpoint["prior"])
    denoiser.load_state_dict(checkpoint["denoiser"])
    prior.eval()
    denoiser.eval()
    return checkpoint, prior, denoiser


@torch.no_grad()
def sample_action_chunk(
    prior: BornPrior,
    denoiser: ModeConditionedDenoiser,
    obs: torch.Tensor,
    schedule: DiffusionSchedule,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    generator = None if seed is None else torch.Generator(device=obs.device).manual_seed(seed)
    _, probs = prior(obs)
    mode = probs.argmax(dim=-1)
    chunk = torch.randn(
        obs.shape[0],
        denoiser.horizon,
        denoiser.action_dim,
        generator=generator,
        device=obs.device,
    )
    betas, alpha_bars = schedule.tensors(obs.device)
    for index in reversed(range(schedule.timesteps)):
        timestep = torch.full((obs.shape[0],), index, dtype=torch.long, device=obs.device)
        pred_noise = denoiser(chunk, timestep, obs, mode)
        beta_t = betas[index]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = alpha_bars[index]
        chunk = (chunk - beta_t * pred_noise / (1.0 - alpha_bar_t).sqrt()) / alpha_t.sqrt()
        if index > 0:
            chunk = chunk + beta_t.sqrt() * torch.randn(
                chunk.shape,
                generator=generator,
                device=obs.device,
            )
    return chunk, mode


def evaluate(checkpoint_path: str, num_samples: int = 256) -> dict[str, float]:
    checkpoint, prior, _ = load_qbdp_checkpoint(checkpoint_path)
    dataset = make_synthetic_expert_dataset(
        SyntheticConfig(
            num_samples=num_samples,
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            horizon=checkpoint["horizon"],
            num_modes=checkpoint["num_modes"],
        )
    )
    with torch.no_grad():
        _, probs = prior(dataset.observations)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        top_mode = probs.argmax(dim=-1)
        mode_accuracy = (top_mode == dataset.mode_labels).float().mean()
    return {"prior_entropy": float(entropy), "synthetic_mode_accuracy": float(mode_accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QBDP checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="runs/latest/checkpoint.pt")
    parser.add_argument("--num-samples", type=int, default=256)
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.num_samples)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
