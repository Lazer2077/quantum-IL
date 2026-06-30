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

## Extensive Multi-Seed Sweep (4 envs × 3 seeds, 100k online steps)

A larger, broader follow-up: a fourth environment (**Ant-v4**), three seeds per
cell for error bars, an equal **100k** online-step budget for SAC and PPO
(12.5× the 8000-step run), larger imitation budgets (20 demo episodes, 200
imitation steps, 10 online RL episodes), and 10 evaluation episodes. Run as 12
single-thread CPU jobs in parallel on an idle 64-core host; total wall time
≈ 41 minutes.

```bash
# per (env, seed) job; aggregate across seeds afterwards
python scripts/compare_baselines.py --envs <ENV> \
  --rl-timesteps 100000 --eval-episodes 10 \
  --demo-episodes 20 --il-steps 200 --rl-episodes 10 --max-steps 200 \
  --hidden-dim 128 --batch-size 64 --diffusion-steps 20 --seed <SEED>
```

Mean ± std of episodic return across 3 seeds (10 eval episodes each):

| Env | random | BC | diffusion_IL | diffusion_IL+RL | SAC (100k) | PPO (100k) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hopper-v4 | 19±3 | 74±43 | 43±6 | 113±2 | **2056±1229** | 760±418 |
| Walker2d-v4 | 1±1 | 40±18 | 67±10 | 166±85 | **1409±984** | 264±12 |
| HalfCheetah-v4 | -302±15 | -2±0 | -1±0 | -1±0 | **3418±1622** | 478±104 |
| Ant-v4 | -40±35 | 993±2 | 1000±3 | **1002±1** | 732±141 | 96±36 |

### Interpretation

- **With enough budget, online SAC dominates the locomotion tasks.** At 100k
  steps SAC wins Hopper (2056), Walker2d (1409), and HalfCheetah (3418) by large
  margins — an order of magnitude above the weak-expert imitation methods, whose
  ceiling is the quality of the random-expert buffer.
- **SAC variance is large at this budget** (Hopper ±1229, HalfCheetah ±1622):
  100k is mid-training, so across seeds some runs have taken off and others are
  still climbing. The means are not converged numbers.
- **PPO ≪ SAC at equal budget.** On-policy PPO is far less sample-efficient; at
  100k it still beats imitation on Hopper/HalfCheetah but trails SAC everywhere.
- **Ant-v4 inverts the ranking, and that is informative.** Ant pays a ~+1
  per-step survival reward, so simply staying upright for 1000 steps scores
  ≈1000. The imitation methods learn exactly that passive-survival behavior
  (≈1000), while undertrained SAC (732) and PPO (96) are still unstable and fall
  early. This is a reward-structure-plus-budget artifact, not evidence that
  imitation beats RL — at higher budgets SAC adds forward locomotion and passes
  the survival floor.
- **The online RL term in diffusion_IL+RL consistently helps** over plain
  diffusion_IL / BC on Hopper and Walker2d.

These remain research smoke results: 100k steps is well short of the
hundreds-of-thousands-to-millions typically used to converge these tasks, and
the imitation methods use a single deliberately weak random-expert buffer.
