from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RLBenchSampleBatch:
    observations: torch.Tensor
    actions: torch.Tensor


class LowDimBCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, obs_dim]
        action = self.net(obs)
        # action: [batch, action_dim]
        return action


def _low_dim_vector(obs: Any) -> torch.Tensor:
    return torch.as_tensor(obs.get_low_dim_data(), dtype=torch.float32).flatten()


def _joint_velocity_action(current_obs: Any, next_obs: Any, velocity_gain: float, max_velocity: float) -> torch.Tensor:
    current = torch.as_tensor(current_obs.joint_positions, dtype=torch.float32).flatten()
    next_position = torch.as_tensor(next_obs.joint_positions, dtype=torch.float32).flatten()
    velocity = (next_position - current) * velocity_gain
    velocity = velocity.clamp(-max_velocity, max_velocity)
    gripper = torch.tensor([float(next_obs.gripper_open)], dtype=torch.float32)
    return torch.cat([velocity, gripper])


def _hold_action(obs: Any) -> torch.Tensor:
    velocity = torch.zeros(7, dtype=torch.float32)
    gripper = torch.tensor([float(obs.gripper_open)], dtype=torch.float32)
    return torch.cat([velocity, gripper])


def _normalize_joint_velocity_action(action: torch.Tensor, max_velocity: float) -> list[float]:
    action = action.flatten().float().clone()
    action[:7] = action[:7].clamp(-max_velocity, max_velocity)
    action[7] = float(action[7] > 0.5)
    return action.tolist()


def _distance_to_target(obs: Any) -> float:
    gripper = torch.as_tensor(obs.gripper_pose[:3], dtype=torch.float32)
    target = torch.as_tensor(obs.task_low_dim_state[:3], dtype=torch.float32)
    return float(torch.linalg.vector_norm(gripper - target))


def _make_obs_config(ObservationConfig: Any) -> Any:
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.set_all_low_dim(True)
    return obs_config


def _import_rlbench() -> dict[str, Any]:
    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.utils import name_to_task_class
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RLBench imitation requires optional RLBench, PyRep, Gymnasium, and CoppeliaSim setup."
        ) from exc
    return {
        "MoveArmThenGripper": MoveArmThenGripper,
        "JointVelocity": JointVelocity,
        "Discrete": Discrete,
        "Environment": Environment,
        "ObservationConfig": ObservationConfig,
        "name_to_task_class": name_to_task_class,
    }


def _make_env(modules: dict[str, Any], headless: bool) -> Any:
    action_mode = modules["MoveArmThenGripper"](
        modules["JointVelocity"](),
        modules["Discrete"](),
    )
    return modules["Environment"](
        action_mode=action_mode,
        dataset_root="",
        obs_config=_make_obs_config(modules["ObservationConfig"]),
        headless=headless,
    )


def collect_live_demo_samples(
    task: Any,
    demos: int,
    max_attempts: int,
    velocity_gain: float,
    max_velocity: float,
) -> RLBenchSampleBatch:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for demo in task.get_demos(demos, live_demos=True, max_attempts=max_attempts):
        for step in range(len(demo) - 1):
            observations.append(_low_dim_vector(demo[step]))
            actions.append(_joint_velocity_action(demo[step], demo[step + 1], velocity_gain, max_velocity))
    if not observations:
        raise RuntimeError("No RLBench demo transitions were collected.")
    return RLBenchSampleBatch(torch.stack(observations), torch.stack(actions))


def train_bc(
    batch: RLBenchSampleBatch,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[LowDimBCPolicy, dict[str, float]]:
    torch.manual_seed(seed)
    count = batch.observations.shape[0]
    order = torch.randperm(count)
    train_count = max(1, int(0.8 * count))
    train_idx = order[:train_count]
    valid_idx = order[train_count:] if train_count < count else order[:train_count]

    obs_mean = batch.observations[train_idx].mean(dim=0)
    obs_std = batch.observations[train_idx].std(dim=0).clamp_min(1e-6)
    act_mean = batch.actions[train_idx].mean(dim=0)
    act_std = batch.actions[train_idx].std(dim=0).clamp_min(1e-6)

    policy = LowDimBCPolicy(batch.observations.shape[1], batch.actions.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    final_loss = 0.0
    for _ in range(steps):
        idx = train_idx[torch.randint(0, train_idx.numel(), (min(batch_size, train_idx.numel()),))]
        obs = (batch.observations[idx] - obs_mean) / obs_std
        action = (batch.actions[idx] - act_mean) / act_std
        pred = policy(obs)
        loss = F.mse_loss(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss)

    policy.obs_mean = obs_mean  # type: ignore[attr-defined]
    policy.obs_std = obs_std  # type: ignore[attr-defined]
    policy.act_mean = act_mean  # type: ignore[attr-defined]
    policy.act_std = act_std  # type: ignore[attr-defined]

    with torch.no_grad():
        valid_obs = (batch.observations[valid_idx] - obs_mean) / obs_std
        valid_pred = policy(valid_obs) * act_std + act_mean
        valid_mse = F.mse_loss(valid_pred, batch.actions[valid_idx])
    return policy, {"train_loss": final_loss, "valid_action_mse": float(valid_mse)}


def _policy_action(policy: LowDimBCPolicy, obs: Any, max_velocity: float) -> list[float]:
    vector = _low_dim_vector(obs)
    with torch.no_grad():
        norm_obs = (vector - policy.obs_mean) / policy.obs_std  # type: ignore[attr-defined]
        action = policy(norm_obs.unsqueeze(0)).squeeze(0)
        action = action * policy.act_std + policy.act_mean  # type: ignore[attr-defined]
    return _normalize_joint_velocity_action(action, max_velocity)


def evaluate_policy(
    task: Any,
    policy_fn: Callable[[Any], list[float]],
    episodes: int,
    max_steps: int,
) -> dict[str, float]:
    total_return = 0.0
    successes = 0
    initial_distances = []
    final_distances = []
    exceptions = 0
    for _ in range(episodes):
        _, obs = task.reset()
        initial_distances.append(_distance_to_target(obs))
        done = False
        episode_return = 0.0
        for _step in range(max_steps):
            try:
                obs, reward, done = task.step(policy_fn(obs))
            except Exception:
                exceptions += 1
                break
            episode_return += float(reward)
            if done:
                break
        total_return += episode_return
        successes += int(done)
        final_distances.append(_distance_to_target(obs))
    return {
        "mean_return": total_return / max(episodes, 1),
        "success_rate": successes / max(episodes, 1),
        "mean_initial_distance": sum(initial_distances) / max(len(initial_distances), 1),
        "mean_final_distance": sum(final_distances) / max(len(final_distances), 1),
        "planner_exceptions": float(exceptions),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, float | int | str]:
    modules = _import_rlbench()
    env = _make_env(modules, args.headless)
    env.launch()
    try:
        task = env.get_task(modules["name_to_task_class"](args.task))
        samples = collect_live_demo_samples(
            task,
            args.demos,
            args.max_demo_attempts,
            args.velocity_gain,
            args.max_velocity,
        )
        policy, train_metrics = train_bc(
            samples,
            hidden_dim=args.hidden_dim,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

        hold_metrics = evaluate_policy(
            task,
            lambda obs: _normalize_joint_velocity_action(_hold_action(obs), args.max_velocity),
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
        )
        bc_metrics = evaluate_policy(
            task,
            lambda obs: _policy_action(policy, obs, args.max_velocity),
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
        )
    finally:
        env.shutdown()

    metrics: dict[str, float | int | str] = {
        "task": args.task,
        "demo_transitions": int(samples.observations.shape[0]),
        "obs_dim": int(samples.observations.shape[1]),
        "action_dim": int(samples.actions.shape[1]),
        **train_metrics,
    }
    for key, value in hold_metrics.items():
        metrics[f"hold_{key}"] = value
    for key, value in bc_metrics.items():
        metrics[f"bc_{key}"] = value
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small optional RLBench low-dimensional imitation experiment.")
    parser.add_argument("--task", type=str, default="reach_target")
    parser.add_argument("--demos", type=int, default=3)
    parser.add_argument("--max-demo-attempts", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--velocity-gain", type=float, default=10.0)
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    metrics = run_experiment(build_parser().parse_args())
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
