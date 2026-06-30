from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qbdp.data.synthetic import SyntheticConfig, make_synthetic_expert_dataset
from qbdp.models.born_prior import BornPrior
from qbdp.models.diffusion import DiffusionSchedule, ModeConditionedDenoiser, diffusion_loss
from qbdp.tokenizer import KMeansActionTokenizer


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    dataset = make_synthetic_expert_dataset(
        SyntheticConfig(num_samples=args.num_samples, seed=args.seed, num_modes=args.num_modes)
    )
    tokenizer = KMeansActionTokenizer(args.num_modes, seed=args.seed)
    labels = tokenizer.fit(dataset.action_chunks)
    dataset.mode_labels = labels

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    prior = BornPrior(dataset.observations.shape[-1], args.num_modes, args.hidden_dim)
    denoiser = ModeConditionedDenoiser(
        dataset.observations.shape[-1],
        dataset.action_chunks.shape[-1],
        dataset.action_chunks.shape[1],
        args.num_modes,
        args.hidden_dim,
    )
    params = list(prior.parameters()) + list(denoiser.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    schedule = DiffusionSchedule(timesteps=args.diffusion_steps)

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            obs = batch["obs"]
            chunk = batch["action_chunk"]
            mode = batch["mode_label"]
            _, probs = prior(obs)
            prior_loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), mode)
            denoise_loss = diffusion_loss(denoiser, chunk, obs, schedule, mode)
            loss = prior_loss + denoise_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(
        {
            "prior": prior.state_dict(),
            "denoiser": denoiser.state_dict(),
            "tokenizer_centroids": tokenizer.centroids,
            "obs_dim": dataset.observations.shape[-1],
            "action_dim": dataset.action_chunks.shape[-1],
            "horizon": dataset.action_chunks.shape[1],
            "num_modes": args.num_modes,
            "hidden_dim": args.hidden_dim,
        },
        checkpoint,
    )
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train QBDP on synthetic expert data.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="runs/latest")
    return parser


def main() -> None:
    checkpoint = train(build_parser().parse_args())
    print(f"wrote {checkpoint}")


if __name__ == "__main__":
    main()
