# QBDP Report

## Objective

The target policy factorization is:

```text
p(U | o) = sum_b p_phi(b | o) p_theta(U | o, b)
```

where `o` is an observation, `U` is a horizon-length action chunk, `b` is a
discrete mode/bitstring, `p_phi(b | o)=|psi_phi(b | o)|^2` is a Born-style prior,
and `p_theta(U | o, b)` is a conditional diffusion decoder.

## Implementation Notes

The current implementation is CPU-first and PyTorch-only by default. Synthetic
expert data is generated from observation-dependent multimodal action templates.
Action chunks are tokenized with a torch-only K-means module.

The Born prior is implemented as:

```text
amplitudes = normalize(mlp(obs))
probs = amplitudes ** 2
```

The QBDP decoder uses the DDPM noise-prediction objective:

```text
pred_noise = denoiser(noisy_chunk, timestep, obs, mode_label)
L = ||pred_noise - noise||_2^2
```

Added baselines:

- Standard diffusion policy: removes `b` and predicts denoising noise from
  `(noisy_chunk, timestep, obs)`.
- CVAE action-chunk policy: encodes `(obs, U)` into a Gaussian latent and decodes
  `U` from `(obs, z)`.
- Behavior cloning: direct action-chunk regression from `obs`.
- DPPO-inspired Gym locomotion baseline: a one-step diffusion actor receives
  imitation denoising loss and online policy-gradient loss at the same time.
  This is an engineering baseline for optional MuJoCo smoke tests, not a full
  reproduction of the DPPO paper.

Rollout-guided amplitude refinement is implemented as:

```text
psi_i' = normalize(psi_i * exp(eta * R_i / 2))
```

Dataset extensions are optional:

- Minari/D4RL MuJoCo loader via `qbdp.data.minari_loader`.
- RoboMimic HDF5 loader via `qbdp.data.robomimic_loader`.
- RLBench smoke/evaluation via `qbdp.evaluate_rlbench`, keeping RLBench and
  CoppeliaSim outside the default dependency path.

The RLBench evaluator reuses the trained Born prior and denoiser to sample the
first action from a diffusion chunk. It uses low-dimensional RLBench
observations, adapts vector sizes by padding or truncating when a synthetic
checkpoint does not match the robot task, and reports mismatch flags so these
smoke results are not confused with a task-trained policy.

## Experiment Results

Synthetic comparison smoke tests run through:

```bash
python scripts/compare_synthetic.py --steps 25
```

This command reports final training losses for BC, CVAE, standard diffusion, and
QBDP on the synthetic dataset. Full benchmark tables are pending longer runs.

## Gym Locomotion Smoke Result

Baseline selected: Diffusion Policy Policy Optimization (DPPO) inspired
diffusion policy fine-tuning. The implemented local baseline is deliberately
small: it uses a one-step diffusion actor with an approximate Gaussian
log-probability for policy-gradient updates, plus a denoising imitation loss.

Command:

```bash
python scripts/gym_locomotion_diffusion_rl.py --demo-episodes 3 --il-steps 5 --rl-episodes 2 --eval-episodes 2 --max-steps 50 --batch-size 16 --hidden-dim 32 --diffusion-steps 8
```

Results are average return over two short evaluation episodes:

| Env | Random | BC | Diffusion IL | Diffusion IL + RL |
| --- | ---: | ---: | ---: | ---: |
| Hopper-v5 | 12.121 | 77.827 | 52.046 | 38.114 |
| Walker2d-v5 | 4.934 | 7.730 | 16.534 | 37.595 |
| HalfCheetah-v5 | -4.789 | -0.871 | -0.595 | -0.817 |

Interpretation: the full environment and training loop are operational. Because
the default expert buffer is only the top half of random rollouts and the run is
intentionally tiny, these numbers are smoke-test comparisons rather than final
research results. Longer runs should use real expert data from Minari/D4RL or
another expert checkpoint.

## RLBench Evaluation Path

Import probe:

```bash
python scripts/evaluate_rlbench.py --dry-run-import
```

Single-task rollout when RLBench and CoppeliaSim are available:

```bash
python scripts/evaluate_rlbench.py --checkpoint runs/latest/checkpoint.pt --task reach_target --episodes 1 --max-steps 40
```

This is currently an integration test for the QBDP sampling path on RLBench
rather than a benchmark result, because the default checkpoint is trained on
synthetic data.
