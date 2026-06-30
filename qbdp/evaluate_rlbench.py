from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from qbdp.evaluate import load_qbdp_checkpoint, sample_action_chunk
from qbdp.models.diffusion import DiffusionSchedule


@dataclass(frozen=True)
class RLBenchImport:
    available: bool
    error: str | None = None
    Environment: Any | None = None
    ObservationConfig: Any | None = None
    MoveArmThenGripper: Any | None = None
    JointVelocity: Any | None = None
    Discrete: Any | None = None
    name_to_task_class: Any | None = None


def import_rlbench() -> RLBenchImport:
    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.utils import name_to_task_class
    except ImportError as exc:
        return RLBenchImport(False, str(exc))
    return RLBenchImport(
        True,
        Environment=Environment,
        ObservationConfig=ObservationConfig,
        MoveArmThenGripper=MoveArmThenGripper,
        JointVelocity=JointVelocity,
        Discrete=Discrete,
        name_to_task_class=name_to_task_class,
    )


def _low_dim_vector(obs: Any) -> torch.Tensor:
    if hasattr(obs, "get_low_dim_data"):
        value = obs.get_low_dim_data()
        return torch.as_tensor(value, dtype=torch.float32).flatten()
    if hasattr(obs, "low_dim_state"):
        value = obs.low_dim_state
        if callable(value):
            value = value()
        return torch.as_tensor(value, dtype=torch.float32).flatten()

    pieces = []
    for name in ("joint_positions", "joint_velocities", "gripper_open", "gripper_pose"):
        if hasattr(obs, name):
            pieces.append(torch.as_tensor(getattr(obs, name), dtype=torch.float32).flatten())
    if not pieces:
        raise ValueError("RLBench observation has no low-dimensional fields to evaluate.")
    return torch.cat(pieces)


def _fit_vector(vector: torch.Tensor, dim: int) -> tuple[torch.Tensor, bool]:
    vector = vector.flatten().float()
    if vector.numel() == dim:
        return vector, False
    if vector.numel() > dim:
        return vector[:dim], True
    return torch.cat([vector, torch.zeros(dim - vector.numel())]), True


def _fit_action(action: torch.Tensor, action_dim: int) -> tuple[list[float], bool]:
    vector, mismatch = _fit_vector(action.flatten(), action_dim)
    return vector.tolist(), mismatch


def _make_obs_config(ObservationConfig: Any) -> Any:
    obs_config = ObservationConfig()
    if hasattr(obs_config, "set_all"):
        obs_config.set_all(False)
    if hasattr(obs_config, "set_all_low_dim"):
        obs_config.set_all_low_dim(True)
    for name in ("joint_positions", "joint_velocities", "gripper_open", "gripper_pose"):
        if hasattr(obs_config, name):
            setattr(obs_config, name, True)
    return obs_config


def _make_environment(modules: RLBenchImport, dataset_root: str, headless: bool) -> tuple[Any, int]:
    try:
        action_mode = modules.MoveArmThenGripper(
            arm_action_mode=modules.JointVelocity(),
            gripper_action_mode=modules.Discrete(),
        )
    except TypeError:
        action_mode = modules.MoveArmThenGripper(modules.JointVelocity(), modules.Discrete())
    obs_config = _make_obs_config(modules.ObservationConfig)
    try:
        env = modules.Environment(
            action_mode=action_mode,
            dataset_root=dataset_root,
            obs_config=obs_config,
            headless=headless,
        )
    except TypeError:
        env = modules.Environment(action_mode, dataset_root, obs_config, headless)
    action_shape = getattr(env, "action_shape", None)
    if action_shape is None:
        action_shape = getattr(action_mode, "action_shape", None)
    if callable(action_shape):
        try:
            action_shape = action_shape()
        except TypeError:
            action_shape = None
    if action_shape is None:
        action_dim = 8
    elif isinstance(action_shape, int):
        action_dim = action_shape
    else:
        action_dim = int(action_shape[0])
    return env, action_dim


def evaluate_rlbench(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
    modules = import_rlbench()
    if not modules.available:
        return {
            "rlbench_available": False,
            "error": modules.error or "RLBench import failed.",
        }
    if args.dry_run_import:
        return {"rlbench_available": True}

    checkpoint, prior, denoiser = load_qbdp_checkpoint(args.checkpoint)
    schedule = DiffusionSchedule(timesteps=args.diffusion_steps)
    env, action_dim = _make_environment(modules, args.dataset_root, args.headless)
    task_class = modules.name_to_task_class(args.task)
    total_return = 0.0
    successes = 0
    obs_mismatch = False
    action_mismatch = int(checkpoint["action_dim"]) != action_dim

    env.launch()
    try:
        task = env.get_task(task_class)
        for episode in range(args.episodes):
            _, obs = task.reset()
            episode_return = 0.0
            done = False
            for step in range(args.max_steps):
                obs_vector, obs_changed = _fit_vector(_low_dim_vector(obs), int(checkpoint["obs_dim"]))
                obs_mismatch = obs_mismatch or obs_changed
                chunk, _ = sample_action_chunk(
                    prior,
                    denoiser,
                    obs_vector,
                    schedule,
                    seed=args.seed + episode * args.max_steps + step,
                )
                action, action_changed = _fit_action(chunk[0, 0], action_dim)
                action_mismatch = action_mismatch or action_changed
                obs, reward, done = task.step(action)
                episode_return += float(reward)
                if done:
                    break
            total_return += episode_return
            successes += int(done)
    finally:
        env.shutdown()

    return {
        "rlbench_available": True,
        "task": args.task,
        "episodes": args.episodes,
        "mean_return": total_return / max(args.episodes, 1),
        "success_rate": successes / max(args.episodes, 1),
        "obs_dim_mismatch": obs_mismatch,
        "action_dim_mismatch": action_mismatch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a QBDP checkpoint in optional RLBench.")
    parser.add_argument("--checkpoint", type=str, default="runs/latest/checkpoint.pt")
    parser.add_argument("--task", type=str, default="reach_target")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run-import", action="store_true")
    return parser


def main() -> None:
    metrics = evaluate_rlbench(build_parser().parse_args())
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
