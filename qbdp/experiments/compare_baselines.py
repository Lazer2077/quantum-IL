"""Unified-protocol comparison of QBDP gym methods against SAC/PPO baselines.

The project's own gym locomotion experiment trains a behavior-cloning policy, a
diffusion imitation policy, and a one-step diffusion IL+RL actor from a weak
random-expert buffer. This module adds standard online-RL baselines (SAC and
PPO from stable-baselines3) and scores every method with one shared evaluation
protocol so the numbers are directly comparable.

stable-baselines3 and gymnasium[mujoco] are optional extras (install with the
``baselines`` extra); they are not part of the default CPU synthetic path.

Evaluation protocol (identical for all methods):
  ``--eval-episodes`` full native episodes (terminate on done/TimeLimit, hard
  capped at ``--eval-cap`` steps), deterministic actions, fixed seeds.

SAC and PPO receive an equal online-interaction budget (``--rl-timesteps`` env
steps). The behavior-cloning and diffusion methods are trained exactly as the
project does, from the same weak random-expert buffer. Per-method budgets are
reported so the comparison is not mislabeled as compute-equivalent.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from typing import Any, Callable

import torch

from qbdp.experiments.gym_locomotion_diffusion_rl import (
    ENV_IDS,
    _collect_random_expert_data,
    _train_bc,
    _train_diffusion_il,
    _train_hybrid,
)
from qbdp.experiments.gym_locomotion_diffusion_rl import build_parser as _gym_build_parser
from qbdp.models.diffusion import DiffusionSchedule

Policy = Callable[[Any], Any]


def _import_gym() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Baseline comparison requires optional dependency gymnasium[mujoco] (extra: baselines)."
        ) from exc
    return gym


def _import_sb3() -> tuple[Any, Any]:
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Baseline comparison requires optional dependency stable-baselines3 (extra: baselines)."
        ) from exc
    return SAC, PPO


def evaluate(env: Any, act: Policy, episodes: int, seed: int, cap: int) -> tuple[float, float]:
    """Score a policy under the shared protocol. Returns (mean, std) of returns."""
    returns: list[float] = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for _ in range(cap):
            obs, reward, terminated, truncated, _ = env.step(act(obs))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return mean, std


def _gym_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Reuse the gym experiment's default hyper-parameters, overriding shared ones."""
    gym_args = _gym_build_parser().parse_args([])
    for name in ("demo_episodes", "il_steps", "rl_episodes", "max_steps", "batch_size",
                 "hidden_dim", "diffusion_steps", "lr", "seed"):
        setattr(gym_args, name, getattr(args, name))
    return gym_args


def _bc_policy(bc: Any) -> Policy:
    def act(obs: Any) -> Any:
        # obs: [obs_dim] -> tensor [1, obs_dim]
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.tanh(bc(tensor).squeeze(1)).squeeze(0)
        # action: [action_dim]
        return action.numpy()
    return act


def _diffusion_policy(actor: Any, schedule: DiffusionSchedule) -> Policy:
    def act(obs: Any) -> Any:
        # obs: [obs_dim] -> tensor [1, obs_dim]
        tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor.act(tensor, schedule, deterministic=True)
        # action: [1, action_dim]
        return action.squeeze(0).numpy()
    return act


def run_env(env_id: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    gym = _import_gym()
    SAC, PPO = _import_sb3()
    gym_args = _gym_defaults(args)
    schedule = DiffusionSchedule(timesteps=gym_args.diffusion_steps)
    rows: list[dict[str, Any]] = []

    # --- Project methods: weak random-expert imitation (as in the repo) ---
    train_env = gym.make(env_id)
    expert_obs, expert_actions = _collect_random_expert_data(
        train_env, gym_args.demo_episodes, gym_args.max_steps, gym_args.seed
    )
    bc = _train_bc(train_env, expert_obs, expert_actions, gym_args.il_steps,
                   gym_args.batch_size, gym_args.hidden_dim, gym_args.lr)
    diffusion_il = _train_diffusion_il(
        expert_obs, expert_actions, gym_args.il_steps, gym_args.batch_size,
        gym_args.hidden_dim, gym_args.lr, schedule
    )
    hybrid = _train_hybrid(train_env, expert_obs, expert_actions, gym_args, schedule)
    train_env.close()

    buf = int(expert_obs.shape[0])
    project_methods = {
        "random": (lambda obs, e=None: None, "weak-buffer baseline"),
        "BC": (_bc_policy(bc), f"buffer={buf}, {gym_args.il_steps} steps"),
        "diffusion_IL": (_diffusion_policy(diffusion_il, schedule), f"buffer={buf}, {gym_args.il_steps} steps"),
        "diffusion_IL+RL": (_diffusion_policy(hybrid, schedule), f"buffer={buf} + {gym_args.rl_episodes} RL eps"),
    }

    eval_env = gym.make(env_id)
    random_act: Policy = lambda obs: eval_env.action_space.sample()
    for name, (act, budget) in project_methods.items():
        policy = random_act if name == "random" else act
        mean, std = evaluate(eval_env, policy, args.eval_episodes, args.seed + 2000, args.eval_cap)
        rows.append({"env": env_id, "method": name, "mean": mean, "std": std, "budget": budget})
        print(f"  [{env_id:14s}] {name:16s} {mean:9.2f} +/- {std:7.2f}   ({budget})", flush=True)
    eval_env.close()

    # --- Standard RL baselines: equal online interaction budget ---
    for algo_name, Algo, extra in (("SAC", SAC, {"learning_starts": 500}), ("PPO", PPO, {})):
        start = time.time()
        learn_env = gym.make(env_id)
        model = Algo("MlpPolicy", learn_env, seed=args.seed, device=args.device, verbose=0, **extra)
        model.learn(total_timesteps=args.rl_timesteps, progress_bar=False)
        score_env = gym.make(env_id)

        def sb3_act(obs: Any, _model: Any = model) -> Any:
            action, _ = _model.predict(obs, deterministic=True)
            return action

        mean, std = evaluate(score_env, sb3_act, args.eval_episodes, args.seed + 2000, args.eval_cap)
        budget = f"{args.rl_timesteps} env steps online"
        rows.append({"env": env_id, "method": algo_name, "mean": mean, "std": std, "budget": budget})
        print(f"  [{env_id:14s}] {algo_name:16s} {mean:9.2f} +/- {std:7.2f}   "
              f"({budget}, {time.time() - start:.0f}s)", flush=True)
        learn_env.close()
        score_env.close()
    return rows


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    print(f"=== Unified comparison (eval: {args.eval_episodes} full eps, "
          f"cap {args.eval_cap}, deterministic) ===", flush=True)
    for env_id in args.envs:
        print(f"\n--- {env_id} ---", flush=True)
        all_rows.extend(run_env(env_id, args))
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare QBDP gym methods against SAC/PPO baselines.")
    parser.add_argument("--envs", nargs="+", default=list(ENV_IDS),
                        help="Gym env ids (default repo v5 ids; use v4 ids on gymnasium<1.0).")
    parser.add_argument("--rl-timesteps", type=int, default=8000,
                        help="Equal online-interaction budget for SAC and PPO.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-cap", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for SB3 (cpu recommended for MlpPolicy).")
    # Hyper-parameters shared with the gym IL experiment.
    parser.add_argument("--demo-episodes", type=int, default=4)
    parser.add_argument("--il-steps", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", type=str, default="", help="Optional path to write a results CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)

    methods = ["random", "BC", "diffusion_IL", "diffusion_IL+RL", "SAC", "PPO"]
    print("\n================ SUMMARY (mean episodic return) ================")
    print(f"{'env':14s} " + " ".join(f"{m:>16s}" for m in methods))
    for env_id in args.envs:
        cells = []
        for method in methods:
            row = next((r for r in rows if r["env"] == env_id and r["method"] == method), None)
            cells.append(f"{row['mean']:16.2f}" if row else f"{'-':>16s}")
        print(f"{env_id:14s} " + " ".join(cells))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["env", "method", "mean", "std", "budget"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{row[k]:.3f}" if isinstance(row[k], float) else row[k])
                                 for k in writer.fieldnames})
        print(f"\nCSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
