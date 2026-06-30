import argparse

import pytest

from qbdp.data.minari_loader import load_minari_action_chunks
from qbdp.data.robomimic_loader import load_robomimic_action_chunks
from qbdp.data.synthetic import SyntheticConfig, make_synthetic_expert_dataset
from qbdp.evaluate_rlbench import import_rlbench
from qbdp.experiments.compare_synthetic import run
from qbdp.tokenizer import KMeansActionTokenizer


def test_synthetic_dataset_and_tokenizer() -> None:
    dataset = make_synthetic_expert_dataset(SyntheticConfig(num_samples=32, seed=1))
    assert len(dataset) == 32
    labels = KMeansActionTokenizer(num_modes=4, iters=2).fit(dataset.action_chunks)
    assert labels.shape == (32,)


def test_compare_experiment_smoke() -> None:
    metrics = run(
        argparse.Namespace(
            steps=1,
            batch_size=8,
            num_samples=16,
            num_modes=4,
            hidden_dim=16,
            diffusion_steps=5,
            lr=1e-3,
            seed=3,
        )
    )
    assert {"bc_mse", "cvae_loss", "diffusion_loss", "qbdp_loss"} <= set(metrics)


def test_optional_loaders_raise_helpful_import_errors_when_missing() -> None:
    try:
        import minari  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional"):
            load_minari_action_chunks("missing-dataset")

    try:
        import h5py  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional"):
            load_robomimic_action_chunks("missing.hdf5")


def test_rlbench_import_probe_is_non_throwing() -> None:
    result = import_rlbench()
    assert isinstance(result.available, bool)
    if not result.available:
        assert result.error
