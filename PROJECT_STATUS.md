# Project Status

- Initial scaffold: QBDP package, synthetic data, Born prior, mode-conditioned diffusion decoder, baselines, optional dataset loaders, refinement utility, experiment script, and tests.

- 2026-06-23T12:32:59-04:00: Added QBDP scaffold with diffusion and CVAE baselines, optional dataset loaders, rollout-guided refinement, experiment script, docs, and tests.

- 2026-06-23T12:33:20-04:00: Fixed CVAE baseline loss wiring in the synthetic comparison experiment.

- 2026-06-23T13:59:26-04:00: Fixed editable install package discovery to include only qbdp.

- 2026-06-23T14:00:14-04:00: Verified editable install, short training, evaluation, and synthetic comparison run; cleaned checkpoint loading warning.

- 2026-06-23T14:16:47-04:00: Added DPPO-inspired Gymnasium MuJoCo locomotion baseline and smoke comparison results for Hopper-v5, Walker2d-v5, and HalfCheetah-v5.

- 2026-06-23T17:17:25-04:00: Added normal script entry points for training, evaluation, synthetic comparison, and Gym locomotion experiments.

- 2026-06-29T22:19:21-04:00: Added script-first RLBench import probe and optional QBDP rollout evaluator.

- 2026-06-30T18:29:25+00:00: Added SAC/PPO baseline comparison (scripts/compare_baselines.py, qbdp.experiments.compare_baselines) under a unified eval protocol, with docs/BASELINE_COMPARISON.md results and a baselines optional extra.

- 2026-06-30T20:52:47+00:00: Added extensive multi-seed baseline sweep (4 envs incl. Ant-v4, 3 seeds, 100k SAC/PPO) to docs/BASELINE_COMPARISON.md; SAC dominates locomotion at budget, Ant survival-reward inversion noted.

- 2026-06-30T22:21:26+00:00: Added Quantum-Born RL: the Born factorization trained as a pure RL algorithm (qbdp.models.quantum_rl + qbdp.experiments.quantum_rl, scripts/train_quantum_rl.py) via PPO policy gradient on Born modes + mode-conditioned diffusion actor + value critic; solves InvertedPendulum-v4 (eval 1000) from reward only. Docs in docs/QUANTUM_RL.md.

- 2026-06-30T15:16:58-04:00: Installed and validated optional RLBench path with CoppeliaSim/PyRep environment setup and action-shape fallback.

- 2026-06-30T15:30:51-04:00: Added optional RLBench low-dimensional imitation diagnostic and recorded initial reach_target BC rollout results.
