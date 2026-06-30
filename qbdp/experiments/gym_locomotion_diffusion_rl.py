from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from qbdp.models.baselines import BehaviorCloningPolicy
from qbdp.models.diffusion import DiffusionSchedule, StandardDiffusionPolicy, diffusion_loss


ENV_IDS = ("Hopper-v5", "Walker2d-v5", "HalfCheetah-v5")


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    log_probs: torch.Tensor


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim]
        value = self.net(obs).squeeze(-1)
        # value: [batch]
        return value


class OneStepDiffusionActor(nn.Module):
    """Practical DPPO-lite actor for Gym actions.

    The denoiser learns the imitation objective. For online RL, a one-step
    reverse-diffusion estimate defines a Gaussian action mean so the policy has
    a tractable approximate log-probability.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.denoiser = StandardDiffusionPolicy(obs_dim, action_dim, horizon=1, hidden_dim=hidden_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def mean(self, obs: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
        # obs: [batch, obs_dim]
        batch = obs.shape[0]
        timestep = torch.full((batch,), schedule.timesteps - 1, device=obs.device, dtype=torch.long)
        # timestep: [batch]
        noisy_action = torch.zeros(batch, 1, self.action_dim, device=obs.device)
        # noisy_action: [batch, 1, action_dim]
        pred_noise = self.denoiser(noisy_action, timestep, obs)
        # pred_noise: [batch, 1, action_dim]
        _, alpha_bars = schedule.tensors(obs.device)
        alpha_bar = alpha_bars[timestep].view(batch, 1, 1)
        denoised = (noisy_action - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        # denoised: [batch, 1, action_dim]
        action_mean = torch.tanh(denoised.squeeze(1))
        # action_mean: [batch, action_dim]
        return action_mean

    def distribution(self, obs: torch.Tensor, schedule: DiffusionSchedule) -> torch.distributions.Normal:
        # obs: [batch, obs_dim]
        mean = self.mean(obs, schedule)
        # mean: [batch, action_dim]
        std = self.log_std.exp().expand_as(mean)
        # std: [batch, action_dim]
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor, schedule: DiffusionSchedule, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: [batch, obs_dim]
        dist = self.distribution(obs, schedule)
        action = dist.mean if deterministic else dist.rsample()
        # action: [batch, action_dim]
        clipped = action.clamp(-1.0, 1.0)
        # clipped: [batch, action_dim]
        log_prob = dist.log_prob(action).sum(dim=-1)
        # log_prob: [batch]
        return clipped, log_prob


def _discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def _collect_random_expert_data(env, episodes: int, max_steps: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    demos: list[tuple[float, list[torch.Tensor], list[torch.Tensor]]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        total = 0.0
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs_rows.append(torch.as_tensor(obs, dtype=torch.float32))
            action_rows.append(torch.as_tensor(action, dtype=torch.float32))
            total += float(reward)
            obs = next_obs
            if terminated or truncated:
                break
        demos.append((total, obs_rows, action_rows))
    demos.sort(key=lambda item: item[0], reverse=True)
    keep = max(1, len(demos) // 2)
    obs_tensors = [row for _, obs_rows, _ in demos[:keep] for row in obs_rows]
    action_tensors = [row for _, _, action_rows in demos[:keep] for row in action_rows]
    return torch.stack(obs_tensors), torch.stack(action_tensors)


def _sample_imitation_batch(
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, expert_obs.shape[0], (batch_size,))
    return expert_obs[indices], expert_actions[indices]


def _train_bc(
    env,
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> BehaviorCloningPolicy:
    obs_dim = expert_obs.shape[-1]
    action_dim = expert_actions.shape[-1]
    policy = BehaviorCloningPolicy(obs_dim, action_dim, horizon=1, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    for _ in range(steps):
        obs, actions = _sample_imitation_batch(expert_obs, expert_actions, batch_size)
        pred = policy(obs).squeeze(1)
        loss = F.mse_loss(torch.tanh(pred), actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy


def _train_diffusion_il(
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    schedule: DiffusionSchedule,
) -> OneStepDiffusionActor:
    actor = OneStepDiffusionActor(expert_obs.shape[-1], expert_actions.shape[-1], hidden_dim)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
    for _ in range(steps):
        obs, actions = _sample_imitation_batch(expert_obs, expert_actions, batch_size)
        loss = diffusion_loss(actor.denoiser, actions.unsqueeze(1), obs, schedule)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return actor


def _collect_actor_episode(
    env,
    actor: OneStepDiffusionActor,
    value_fn: ValueFunction,
    schedule: DiffusionSchedule,
    max_steps: int,
    gamma: float,
    seed: int,
) -> tuple[TransitionBatch, float]:
    obs, _ = env.reset(seed=seed)
    obs_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    log_prob_rows: list[torch.Tensor] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob = actor.act(obs_tensor, schedule)
            _ = value_fn(obs_tensor)
        action = action_tensor.squeeze(0).numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_rows.append(obs_tensor.squeeze(0))
        action_rows.append(action_tensor.squeeze(0))
        log_prob_rows.append(log_prob.squeeze(0))
        rewards.append(float(reward))
        obs = next_obs
        if terminated or truncated:
            break
    returns = _discounted_returns(rewards, gamma)
    return (
        TransitionBatch(
            obs=torch.stack(obs_rows),
            actions=torch.stack(action_rows),
            returns=returns,
            log_probs=torch.stack(log_prob_rows),
        ),
        sum(rewards),
    )


def _train_hybrid(
    env,
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    args: argparse.Namespace,
    schedule: DiffusionSchedule,
) -> OneStepDiffusionActor:
    actor = _train_diffusion_il(
        expert_obs,
        expert_actions,
        args.il_steps,
        args.batch_size,
        args.hidden_dim,
        args.lr,
        schedule,
    )
    value_fn = ValueFunction(expert_obs.shape[-1], args.hidden_dim)
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(value_fn.parameters()), lr=args.lr)
    for episode in range(args.rl_episodes):
        batch, _ = _collect_actor_episode(
            env, actor, value_fn, schedule, args.max_steps, args.gamma, args.seed + 1000 + episode
        )
        values = value_fn(batch.obs)
        advantages = batch.returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))
        dist = actor.distribution(batch.obs, schedule)
        log_probs = dist.log_prob(batch.actions).sum(dim=-1)
        pg_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, batch.returns)
        il_obs, il_actions = _sample_imitation_batch(expert_obs, expert_actions, args.batch_size)
        il_loss = diffusion_loss(actor.denoiser, il_actions.unsqueeze(1), il_obs, schedule)
        entropy = dist.entropy().sum(dim=-1).mean()
        loss = pg_loss + args.value_coef * value_loss + args.il_coef * il_loss - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(actor.parameters()) + list(value_fn.parameters()), 1.0)
        optimizer.step()
    return actor


def _evaluate_random(env, episodes: int, max_steps: int, seed: int) -> float:
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(max_steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(torch.tensor(returns).mean())


def _evaluate_bc(env, policy: BehaviorCloningPolicy, episodes: int, max_steps: int, seed: int) -> float:
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(max_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.tanh(policy(obs_tensor).squeeze(1)).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(torch.tensor(returns).mean())


def _evaluate_diffusion(
    env,
    actor: OneStepDiffusionActor,
    schedule: DiffusionSchedule,
    episodes: int,
    max_steps: int,
    seed: int,
) -> float:
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(max_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = actor.act(obs_tensor, schedule, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(torch.tensor(returns).mean())


def run_env(env_id: str, args: argparse.Namespace) -> dict[str, float]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("Gym locomotion experiments require optional dependency gymnasium[mujoco].") from exc

    env = gym.make(env_id)
    schedule = DiffusionSchedule(timesteps=args.diffusion_steps)
    expert_obs, expert_actions = _collect_random_expert_data(
        env, args.demo_episodes, args.max_steps, args.seed
    )
    bc = _train_bc(env, expert_obs, expert_actions, args.il_steps, args.batch_size, args.hidden_dim, args.lr)
    diffusion_il = _train_diffusion_il(
        expert_obs, expert_actions, args.il_steps, args.batch_size, args.hidden_dim, args.lr, schedule
    )
    hybrid = _train_hybrid(env, expert_obs, expert_actions, args, schedule)

    metrics = {
        "random": _evaluate_random(env, args.eval_episodes, args.max_steps, args.seed + 2000),
        "bc": _evaluate_bc(env, bc, args.eval_episodes, args.max_steps, args.seed + 3000),
        "diffusion_il": _evaluate_diffusion(
            env, diffusion_il, schedule, args.eval_episodes, args.max_steps, args.seed + 4000
        ),
        "diffusion_il_rl": _evaluate_diffusion(
            env, hybrid, schedule, args.eval_episodes, args.max_steps, args.seed + 5000
        ),
        "expert_buffer_size": float(expert_obs.shape[0]),
    }
    env.close()
    return metrics


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    return {env_id: run_env(env_id, args) for env_id in args.envs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gym MuJoCo diffusion-policy IL+RL baseline comparison.")
    parser.add_argument("--envs", nargs="+", default=list(ENV_IDS), choices=list(ENV_IDS))
    parser.add_argument("--demo-episodes", type=int, default=4)
    parser.add_argument("--il-steps", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--il-coef", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    results = run(build_parser().parse_args())
    header = "env,random,bc,diffusion_il,diffusion_il_rl,expert_buffer_size"
    print(header)
    for env_id, metrics in results.items():
        print(
            f"{env_id},{metrics['random']:.3f},{metrics['bc']:.3f},"
            f"{metrics['diffusion_il']:.3f},{metrics['diffusion_il_rl']:.3f},"
            f"{metrics['expert_buffer_size']:.0f}"
        )


if __name__ == "__main__":
    main()
