"""Train the Quantum-Born actor-critic as a pure RL algorithm on a Gym env.

This is reinforcement learning end to end: the agent interacts with the
environment, and the only learning signal is reward. There is no expert data and
no imitation loss anywhere. The quantum-inspired Born distribution defines the
discrete mode policy; a mode-conditioned one-step diffusion actor emits the
continuous action; both are optimized by a PPO-clipped policy gradient with a
value baseline. ``gymnasium[mujoco]`` is an optional extra.
"""
from __future__ import annotations

import argparse
from collections import deque
from typing import Any

import torch
from torch import nn

from qbdp.models.quantum_rl import QuantumBornActorCritic


def _import_gym() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Quantum-Born RL requires optional dependency gymnasium[mujoco] (extra: baselines)."
        ) from exc
    return gym


class RunningMeanStd:
    """Welford running observation statistics for input normalization."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, batch: torch.Tensor) -> None:
        # batch: [n, *shape]
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        n = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta.square() * self.count * n / total) / total
        self.count = total

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.mean) / (self.var + 1e-8).sqrt()).clamp(-10.0, 10.0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rewards/values/dones: [T]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def collect_rollout(env, model, obs_rms, state, steps):
    """Run the env for ``steps`` transitions; returns a buffer dict + final state."""
    obs, ep_return, recent = state
    buf = {k: [] for k in ("obs", "mode", "action", "log_prob", "value", "reward", "done")}
    raw_obs_seen = []
    for _ in range(steps):
        raw_obs_seen.append(obs)
        norm_obs = obs_rms.normalize(obs)        # model sees normalized obs ...
        step = model.act(norm_obs.unsqueeze(0))
        action = step.clipped_action.squeeze(0).numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buf["obs"].append(norm_obs)              # ... and the PPO update reuses the same normalized obs
        buf["mode"].append(step.mode.squeeze(0))
        buf["action"].append(step.action.squeeze(0))
        buf["log_prob"].append(step.log_prob.squeeze(0))
        buf["value"].append(step.value.squeeze(0))
        buf["reward"].append(float(reward))
        buf["done"].append(float(done))
        ep_return += float(reward)
        if done:
            recent.append(ep_return)
            ep_return = 0.0
            next_obs, _ = env.reset()
        obs = torch.as_tensor(next_obs, dtype=torch.float32)

    out = {
        "obs": torch.stack(buf["obs"]),
        "mode": torch.stack(buf["mode"]),
        "action": torch.stack(buf["action"]),
        "log_prob": torch.stack(buf["log_prob"]),
        "value": torch.stack(buf["value"]),
        "reward": torch.tensor(buf["reward"], dtype=torch.float32),
        "done": torch.tensor(buf["done"], dtype=torch.float32),
    }
    obs_rms.update(torch.stack(raw_obs_seen))  # running stats track RAW observations
    return out, (obs, ep_return, recent)


def ppo_update(model, optimizer, buf, args) -> dict[str, float]:
    with torch.no_grad():
        last_value = model.value(buf["obs"][-1].unsqueeze(0)).squeeze(0)
    advantages, returns = compute_gae(
        buf["reward"], buf["value"], buf["done"], last_value, args.gamma, args.lam
    )
    advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

    n = buf["obs"].shape[0]
    idx = torch.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    for _ in range(args.epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, args.minibatch):
            mb = perm[start:start + args.minibatch]
            log_prob, entropy, value = model.evaluate_actions(
                buf["obs"][mb], buf["mode"][mb], buf["action"][mb]
            )
            ratio = (log_prob - buf["log_prob"][mb]).exp()
            surr1 = ratio * advantages[mb]
            surr2 = ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (value - returns[mb]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy_mean.detach())
            updates += 1
    return {k: v / max(updates, 1) for k, v in stats.items()}


def evaluate(env, model, obs_rms, episodes: int, cap: int) -> float:
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        obs = torch.as_tensor(obs, dtype=torch.float32)
        total = 0.0
        for _ in range(cap):
            with torch.no_grad():
                step = model.act(obs_rms.normalize(obs).unsqueeze(0), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(step.clipped_action.squeeze(0).numpy())
            obs = torch.as_tensor(obs, dtype=torch.float32)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(torch.tensor(returns).mean())


def train_quantum_rl(args: argparse.Namespace) -> dict[str, float]:
    gym = _import_gym()
    torch.manual_seed(args.seed)
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    model = QuantumBornActorCritic(
        obs_dim, action_dim, num_modes=args.num_modes,
        action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
        action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
        hidden_dim=args.hidden_dim, diffusion_steps=args.diffusion_steps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    obs_rms = RunningMeanStd((obs_dim,))

    obs0, _ = env.reset(seed=args.seed)
    state = (torch.as_tensor(obs0, dtype=torch.float32), 0.0, deque(maxlen=50))

    steps_done = 0
    history: list[float] = []
    while steps_done < args.total_steps:
        buf, state = collect_rollout(env, model, obs_rms, state, args.rollout_steps)
        ppo_update(model, optimizer, buf, args)
        steps_done += args.rollout_steps
        recent = state[2]
        mean_return = float(torch.tensor(list(recent)).mean()) if recent else float("nan")
        history.append(mean_return)
        if args.verbose:
            print(f"  steps={steps_done:>7d}  train_return(mean last {len(recent)})={mean_return:8.2f}", flush=True)

    eval_env = gym.make(args.env)
    final_eval = evaluate(eval_env, model, obs_rms, args.eval_episodes, args.eval_cap)
    env.close()
    eval_env.close()
    return {
        "env": args.env,
        "final_eval_return": final_eval,
        "first_train_return": next((h for h in history if h == h), float("nan")),
        "last_train_return": next((h for h in reversed(history) if h == h), float("nan")),
        "total_steps": float(steps_done),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Quantum-Born actor-critic with PPO (pure RL).")
    parser.add_argument("--env", type=str, default="InvertedPendulum-v4")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    metrics = train_quantum_rl(build_parser().parse_args())
    print("=== Quantum-Born RL ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
