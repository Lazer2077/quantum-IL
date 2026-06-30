from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qbdp.data.synthetic import SyntheticConfig, make_synthetic_expert_dataset
from qbdp.models.baselines import BehaviorCloningPolicy, CVAEActionChunkPolicy, cvae_loss
from qbdp.models.born_prior import BornPrior
from qbdp.models.diffusion import DiffusionSchedule, ModeConditionedDenoiser, StandardDiffusionPolicy, diffusion_loss
from qbdp.tokenizer import KMeansActionTokenizer


def run(args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(args.seed)
    dataset = make_synthetic_expert_dataset(SyntheticConfig(num_samples=args.num_samples, seed=args.seed))
    tokenizer = KMeansActionTokenizer(args.num_modes, seed=args.seed)
    dataset.mode_labels = tokenizer.fit(dataset.action_chunks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    obs_dim = dataset.observations.shape[-1]
    horizon = dataset.action_chunks.shape[1]
    action_dim = dataset.action_chunks.shape[-1]
    schedule = DiffusionSchedule(timesteps=args.diffusion_steps)

    bc = BehaviorCloningPolicy(obs_dim, action_dim, horizon, args.hidden_dim)
    cvae = CVAEActionChunkPolicy(obs_dim, action_dim, horizon, hidden_dim=args.hidden_dim)
    dp = StandardDiffusionPolicy(obs_dim, action_dim, horizon, args.hidden_dim)
    prior = BornPrior(obs_dim, args.num_modes, args.hidden_dim)
    qbdp = ModeConditionedDenoiser(obs_dim, action_dim, horizon, args.num_modes, args.hidden_dim)

    optimizers = {
        "bc": torch.optim.AdamW(bc.parameters(), lr=args.lr),
        "cvae": torch.optim.AdamW(cvae.parameters(), lr=args.lr),
        "diffusion": torch.optim.AdamW(dp.parameters(), lr=args.lr),
        "qbdp": torch.optim.AdamW(list(prior.parameters()) + list(qbdp.parameters()), lr=args.lr),
    }
    metrics: dict[str, float] = {}

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            obs = batch["obs"]
            chunk = batch["action_chunk"]
            mode = batch["mode_label"]

            bc_loss = F.mse_loss(bc(obs), chunk)
            optimizers["bc"].zero_grad()
            bc_loss.backward()
            optimizers["bc"].step()

            recon, mu, logvar = cvae(obs, chunk)
            vae_loss = cvae_loss(recon, chunk, mu, logvar)
            optimizers["cvae"].zero_grad()
            vae_loss.backward()
            optimizers["cvae"].step()

            dp_loss = diffusion_loss(dp, chunk, obs, schedule)
            optimizers["diffusion"].zero_grad()
            dp_loss.backward()
            optimizers["diffusion"].step()

            _, probs = prior(obs)
            qbdp_loss = diffusion_loss(qbdp, chunk, obs, schedule, mode) + F.nll_loss(
                probs.clamp_min(1e-8).log(), mode
            )
            optimizers["qbdp"].zero_grad()
            qbdp_loss.backward()
            optimizers["qbdp"].step()

            metrics = {
                "bc_mse": float(bc_loss.detach()),
                "cvae_loss": float(vae_loss.detach()),
                "diffusion_loss": float(dp_loss.detach()),
                "qbdp_loss": float(qbdp_loss.detach()),
            }
            step += 1
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare QBDP with BC, CVAE, and diffusion baselines.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for key, value in run(args).items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
