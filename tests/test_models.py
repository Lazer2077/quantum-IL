import torch

from qbdp.models.baselines import BehaviorCloningPolicy, CVAEActionChunkPolicy
from qbdp.models.born_prior import BornPrior
from qbdp.models.diffusion import DiffusionSchedule, ModeConditionedDenoiser, StandardDiffusionPolicy, diffusion_loss
from qbdp.refinement import rollout_guided_amplitude_refinement


def test_born_prior_outputs_normalized_probabilities() -> None:
    prior = BornPrior(obs_dim=8, num_modes=4, hidden_dim=16)
    amplitudes, probs = prior(torch.randn(5, 8))
    assert amplitudes.shape == (5, 4)
    assert probs.shape == (5, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_policy_shapes_and_losses() -> None:
    obs = torch.randn(6, 8)
    chunks = torch.randn(6, 4, 3)
    mode = torch.randint(0, 4, (6,))
    timestep = torch.randint(0, 10, (6,))

    denoiser = ModeConditionedDenoiser(8, 3, 4, 4, hidden_dim=16)
    assert denoiser(chunks, timestep, obs, mode).shape == chunks.shape
    assert diffusion_loss(denoiser, chunks, obs, DiffusionSchedule(timesteps=10), mode).ndim == 0

    standard = StandardDiffusionPolicy(8, 3, 4, hidden_dim=16)
    assert standard(chunks, timestep, obs).shape == chunks.shape
    assert diffusion_loss(standard, chunks, obs, DiffusionSchedule(timesteps=10)).ndim == 0

    bc = BehaviorCloningPolicy(8, 3, 4, hidden_dim=16)
    assert bc(obs).shape == chunks.shape

    cvae = CVAEActionChunkPolicy(8, 3, 4, hidden_dim=16)
    reconstruction, mu, logvar = cvae(obs, chunks)
    assert reconstruction.shape == chunks.shape
    assert mu.shape == logvar.shape == (6, 8)


def test_rollout_guided_refinement_normalizes_amplitudes() -> None:
    amplitudes = torch.nn.functional.normalize(torch.ones(2, 4), dim=-1)
    returns = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    refined = rollout_guided_amplitude_refinement(amplitudes, returns, eta=0.2)
    assert refined.shape == amplitudes.shape
    assert torch.allclose(refined.norm(dim=-1), torch.ones(2), atol=1e-6)
