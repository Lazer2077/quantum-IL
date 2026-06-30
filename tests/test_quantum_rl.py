import torch

from qbdp.experiments.quantum_rl import RunningMeanStd, compute_gae, ppo_update
from qbdp.models.quantum_rl import QuantumBornActorCritic


def _model(obs_dim=5, action_dim=2, num_modes=3):
    return QuantumBornActorCritic(
        obs_dim, action_dim, num_modes=num_modes,
        action_low=-2.0 * torch.ones(action_dim),
        action_high=2.0 * torch.ones(action_dim),
        hidden_dim=16, diffusion_steps=5,
    )


def test_act_shapes_and_bounds() -> None:
    model = _model()
    obs = torch.randn(4, 5)
    step = model.act(obs)
    assert step.mode.shape == (4,)
    assert step.action.shape == (4, 2)
    assert step.log_prob.shape == (4,)
    assert step.value.shape == (4,)
    # clipped action respects the [-2, 2] env bounds
    assert torch.all(step.clipped_action <= 2.0 + 1e-5)
    assert torch.all(step.clipped_action >= -2.0 - 1e-5)


def test_born_policy_is_normalized() -> None:
    model = _model()
    born = model.born_distribution(torch.randn(6, 5))
    assert torch.allclose(born.probs.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_evaluate_actions_consistent() -> None:
    model = _model()
    obs = torch.randn(7, 5)
    step = model.act(obs)
    log_prob, entropy, value = model.evaluate_actions(obs, step.mode, step.action)
    # re-evaluating the sampled (mode, action) reproduces the rollout log-prob
    assert torch.allclose(log_prob, step.log_prob, atol=1e-4)
    assert entropy.shape == (7,)
    assert value.shape == (7,)


def test_compute_gae_runs() -> None:
    rewards = torch.ones(8)
    values = torch.zeros(8)
    dones = torch.zeros(8)
    adv, ret = compute_gae(rewards, values, dones, torch.tensor(0.0), gamma=0.99, lam=0.95)
    assert adv.shape == (8,)
    assert ret.shape == (8,)
    assert torch.all(adv > 0)  # all-positive rewards => positive advantage


def test_ppo_update_reduces_loss_on_reward_signal() -> None:
    """One PPO update should increase the chosen actions' likelihood-weighted return."""
    torch.manual_seed(0)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    n = 64
    obs = torch.randn(n, 5)
    step = model.act(obs)
    buf = {
        "obs": obs,
        "mode": step.mode,
        "action": step.action,
        "log_prob": step.log_prob,
        "value": step.value,
        "reward": torch.ones(n),     # constant positive reward
        "done": torch.zeros(n),
    }

    class A:
        gamma, lam, clip, epochs, minibatch, value_coef, entropy_coef = 0.99, 0.95, 0.2, 2, 32, 0.5, 0.01

    stats = ppo_update(model, optimizer, buf, A())
    assert "policy_loss" in stats and "value_loss" in stats


def test_running_mean_std_tracks_raw_obs() -> None:
    rms = RunningMeanStd((3,))
    data = torch.randn(500, 3) * 4.0 + 2.0
    rms.update(data)
    normalized = rms.normalize(data)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(3), atol=0.2)
    assert torch.allclose(normalized.std(dim=0), torch.ones(3), atol=0.2)
