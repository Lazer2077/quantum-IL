# Codex Prompt

You are working on QBDP: Quantum-Born Diffusion Policy.

Research objective:

```text
p(U | o) = sum_b p_phi(b | o) p_theta(U | o, b)
p_phi(b | o) = |psi_phi(b | o)|^2
```

Priorities:

- Preserve the package name `qbdp`.
- Prefer normal script entry points under `scripts/` for user-facing commands.
- Keep default training and tests CPU-only with PyTorch and synthetic data.
- Treat PennyLane, Minari, Gymnasium, MuJoCo, RoboMimic, ManiSkill, and RLBench
  as optional extensions.
- Maintain baselines for BC, CVAE, standard diffusion policy, and QBDP variants.
- Maintain optional Gymnasium MuJoCo locomotion comparisons for BC, diffusion
  imitation-only, and diffusion imitation-plus-RL.
- Update report/status docs after major feature changes.
- Run compile and pytest checks before handing off.
