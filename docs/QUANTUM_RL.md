# Quantum-Born RL: the Born policy as a reinforcement-learning algorithm

The original QBDP is an **imitation-learning** method: the Born prior
`p_phi(b | o) = |psi_phi(b | o)|^2` is fit to k-means mode labels by
cross-entropy, and the mode-conditioned diffusion decoder is fit to expert
action chunks by the DDPM loss. Both objectives require expert data.

This module makes the *same* quantum-inspired factorization a **reinforcement
learning** algorithm: it is trained by maximizing environment return through
interaction, with reward as the only signal. There is no expert dataset and no
imitation loss anywhere.

## Mechanism

The policy keeps the QBDP factorization but reinterprets every part as a
trainable RL policy:

```text
pi(a | o) = sum_b |psi_phi(b | o)|^2 * pi_theta(a | o, b)
```

- **High-level (quantum-inspired) policy.** The discrete mode is sampled from the
  Born distribution `b ~ Categorical(|psi_phi(b | o)|^2)`. This is a *policy* over
  modes, not a prior fit to labels. Its entropy term encourages keeping several
  modes "in superposition" for exploration.
- **Low-level actor.** A mode-conditioned **one-step diffusion** estimate yields a
  Gaussian mean over the (bounded) action, `pi_theta(a | o, b)`. The diffusion
  mechanism is retained, now as an RL actor rather than a chunk reconstructor.
- **Critic.** A value head `V_w(o)` provides the advantage baseline.

Because `(b, a)` is sampled jointly, the joint log-probability is
`log p_phi(b | o) + log pi_theta(a | o, b)`, so a single PPO-clipped ratio
updates both levels at once. Training uses on-policy rollouts, GAE advantages,
observation normalization, advantage normalization, entropy regularization, and
gradient clipping — a standard actor-critic loop driven purely by reward.

### Relation to amplitude refinement

`qbdp.refinement.rollout_guided_amplitude_refinement` applies
`|psi'|^2 \propto |psi|^2 * exp(eta * R)`. That is the closed-form
mirror-descent / exponentiated-advantage update of the Born categorical — i.e.
the soft-greedy version of the very policy-gradient step this algorithm performs
online. The IL refinement utility and the RL training rule are two views of the
same Born-policy improvement operator.

## Run

```bash
python -m pip install -e ".[baselines]"      # gymnasium[mujoco] (+ stable-baselines3)
python scripts/train_quantum_rl.py --env InvertedPendulum-v4 --total-steps 60000
python scripts/train_quantum_rl.py --env Hopper-v4 --total-steps 200000 --rollout-steps 4096
```

Key flags: `--num-modes` (Born modes), `--diffusion-steps`, `--rollout-steps`,
`--epochs`, `--clip`, `--entropy-coef`, `--gamma`, `--lam`.

## Validation: it learns from reward

`InvertedPendulum-v4`, 40k environment steps, seed 0, CPU. Stochastic training
return (mean of the last 50 episodes) versus environment steps:

| steps | 2k | 8k | 16k | 24k | 32k | 41k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train return | 7.4 | 24.3 | 53.1 | 106.3 | 211.9 | 348.5 |

The return rises monotonically from a near-random 7.4, and the **deterministic
(greedy) policy reaches 1000.0 — the maximum for the task** (the pole is balanced
for the full episode). This confirms the method optimizes reward end to end with
no expert supervision.

It is an on-policy actor-critic, so it is more sample-efficient than the IL
baselines on this metric but less so than off-policy SAC; see
[`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md) for the SAC/PPO context.
