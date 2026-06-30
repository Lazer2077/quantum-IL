# Gym Locomotion: QBDP Methods vs SAC/PPO Baselines

This report compares the project's gym locomotion policies against standard
online-RL baselines (SAC and PPO from stable-baselines3) under a single shared
evaluation protocol, so the numbers are directly comparable.

## Methods

- **random** — samples the action space; reference floor.
- **BC** — behavior cloning action regression on a weak random-expert buffer.
- **diffusion_IL** — diffusion imitation policy trained on the same buffer.
- **diffusion_IL+RL** — one-step diffusion actor with imitation loss plus an
  online policy-gradient term (the project's DPPO-inspired baseline).
- **SAC** — off-policy soft actor-critic (stable-baselines3, `MlpPolicy`).
- **PPO** — on-policy proximal policy optimization (stable-baselines3, `MlpPolicy`).

`random`, `BC`, `diffusion_IL`, and `diffusion_IL+RL` come straight from
`qbdp.experiments.gym_locomotion_diffusion_rl`; SAC/PPO are added by
`qbdp.experiments.compare_baselines`.

## Protocol

- **Evaluation (identical for every method):** 5 full native episodes
  (terminate on done / `TimeLimit`, hard cap 1000 steps), deterministic actions,
  fixed seeds.
- **Budgets (intentionally *not* compute-equivalent, reported per method):**
  - BC / diffusion_IL: weak random-expert buffer (top half of a few random
    rollouts), 20 imitation steps.
  - diffusion_IL+RL: the same, plus 3 short online RL episodes.
  - SAC / PPO: an equal online-interaction budget of **8000 env steps** each.

## Reproduce

```bash
# default repo env ids are MuJoCo v5 (gymnasium >= 1.0):
python scripts/compare_baselines.py

# this run used v4 ids because only gymnasium 0.29.1 was available:
python scripts/compare_baselines.py \
  --envs Hopper-v4 Walker2d-v4 HalfCheetah-v4 \
  --rl-timesteps 8000 --eval-episodes 5
```

Optional dependencies: `python -m pip install -e ".[baselines]"`
(installs `gymnasium[mujoco]` and `stable-baselines3`).

## Results

Mean episodic return over 5 evaluation episodes (higher is better). MuJoCo
**v4** environments, SAC/PPO at 8000 online steps, seed 7, SB3 on CPU.

| Env | random | BC | diffusion_IL | diffusion_IL+RL | SAC | PPO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hopper-v4 | 12.55 | 39.78 | 51.36 | 45.46 | **249.19** | 61.34 |
| Walker2d-v4 | 6.41 | 15.92 | 34.37 | 41.83 | 157.07 | **252.74** |
| HalfCheetah-v4 | -260.96 | -1.06 | **-0.23** | -0.30 | -46.28 | -0.95 |

Standard deviations and per-method budgets are printed by the script and can be
written with `--csv`.

## Interpretation

- **Online RL wins where 8000 steps is enough to learn.** SAC is strongest on
  Hopper (249) and PPO on Walker2d (253) — both far above the imitation methods,
  which only ever see a tiny weak-expert buffer.
- **The diffusion imitation methods are stable but capped by data quality.**
  `diffusion_IL` / `diffusion_IL+RL` consistently beat plain BC and random, but
  cannot exceed their weak random-expert source. The online RL term in
  `diffusion_IL+RL` helps on Walker2d (41.8 vs 34.4) and not on Hopper.
- **HalfCheetah is unlearned by everyone at this budget.** Random scores deeply
  negative (-261) because untrained actions drive large negative reward; the
  imitation methods stay near 0 by staying passive, while SAC's online
  exploration is still net-negative (-46) at only 8000 steps.
- **These are smoke-scale results, not a benchmark.** SAC/PPO normally need
  hundreds of thousands to millions of steps; the imitation methods use a
  deliberately weak buffer. The value here is that the full comparison pipeline
  is operational end-to-end and the relative behavior is sensible.

## Environment Notes

- Run on a shared CPU/GPU host with no project-specific install: the experiment
  reused an existing conda environment (gymnasium 0.29.1, mujoco 2.3.7,
  torch 2.8.0) and stable-baselines3 was added as a lightweight optional extra.
- MuJoCo physics steps headless with **no display/X server**, since these
  locomotion environments are never rendered (only `step`/`reset`).
- `v4` ids were used because the available gymnasium is 0.29.1; the script
  defaults to the repo's `v5` ids on gymnasium >= 1.0.
