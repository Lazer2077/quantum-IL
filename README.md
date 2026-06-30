# QBDP: Quantum-Born Diffusion Policy

QBDP is a CPU-first PyTorch research prototype for expert imitation learning with
a Born-style discrete mode prior and a mode-conditioned diffusion decoder:

```text
p(U | o) = sum_b p_phi(b | o) p_theta(U | o, b)
p_phi(b | o) = |psi_phi(b | o)|^2
```

The default implementation uses synthetic multimodal expert action chunks only.
PennyLane, Minari, Gymnasium, MuJoCo, RoboMimic, and ManiSkill are optional
extensions rather than required runtime dependencies.

## Run Without Installing

The project keeps `qbdp` as the internal package name, but the normal workflow
does not require packaging or installing the repository. Run scripts directly
from the repository root:

```bash
python scripts/train.py --steps 200 --batch-size 64 --num-samples 1024
python scripts/evaluate.py --checkpoint runs/latest/checkpoint.pt
```

If you prefer an editable install for development, it is still supported:

```bash
python -m pip install -e .
```

Optional integrations are separate from the default CPU synthetic path:

```bash
python -m pip install -e ".[minari]"
python -m pip install -e ".[robomimic]"
python -m pip install -e ".[quantum]"
python -m pip install -e ".[mujoco]"
python -m pip install -e ".[rlbench]"
```

## Train and Evaluate

The default evaluator scores the Born prior on held-out synthetic data:

```bash
python scripts/evaluate.py --checkpoint runs/latest/checkpoint.pt
```

An optional RLBench smoke/evaluation path probes RLBench imports and can run a
low-dimensional rollout when RLBench and CoppeliaSim are installed:

```bash
python scripts/evaluate_rlbench.py --dry-run-import
python scripts/evaluate_rlbench.py --checkpoint runs/latest/checkpoint.pt --task reach_target --episodes 1 --max-steps 40
```

The synthetic checkpoint dimensions may not match RLBench robot observations or
actions. The RLBench script pads or truncates vectors so the interface can be
tested, and it reports `obs_dim_mismatch` and `action_dim_mismatch` metrics when
that happens.

## Compare Baselines

```bash
python scripts/compare_synthetic.py --steps 25
```

Gymnasium MuJoCo locomotion smoke comparison:

```bash
python scripts/gym_locomotion_diffusion_rl.py
```

Compare the gym methods against SAC/PPO baselines (stable-baselines3) under one
shared evaluation protocol; see [`docs/BASELINE_COMPARISON.md`](docs/BASELINE_COMPARISON.md):

```bash
python -m pip install -e ".[baselines]"
python scripts/compare_baselines.py
```

Implemented policies:

- QBDP with a classical Born-style prior and mode-conditioned DDPM denoiser.
- Behavior cloning action-chunk regression.
- CVAE action-chunk policy.
- Standard diffusion-policy baseline without a Born prior.
- DPPO-inspired one-step diffusion actor for optional Gymnasium MuJoCo
  imitation-plus-RL experiments.

The Gym locomotion script compares random, BC, diffusion imitation-only, and
diffusion imitation-plus-RL on `Hopper-v5`, `Walker2d-v5`, and `HalfCheetah-v5`.
By default it uses a tiny weak expert buffer from high-return random rollouts, so
its default numbers are smoke-test results rather than benchmark claims.

## Verification

```bash
python -m compileall qbdp tests
python -m pytest -q
```
