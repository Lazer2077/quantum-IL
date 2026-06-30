from __future__ import annotations

import torch

from qbdp.data.synthetic import ActionChunkDataset


def load_minari_action_chunks(
    dataset_id: str,
    horizon: int = 4,
    max_episodes: int | None = None,
) -> ActionChunkDataset:
    """Load Minari/D4RL-style MuJoCo expert data if Minari is installed.

    This adapter is intentionally optional. The default test and training path does not
    import Minari, Gymnasium, MuJoCo, or D4RL.
    """

    try:
        import minari  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Minari support is optional. Install qbdp[minari] to use this loader."
        ) from exc

    dataset = minari.load_dataset(dataset_id)
    obs_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []

    for episode_index, episode in enumerate(dataset.iterate_episodes()):
        if max_episodes is not None and episode_index >= max_episodes:
            break
        observations = torch.as_tensor(episode.observations, dtype=torch.float32)
        actions = torch.as_tensor(episode.actions, dtype=torch.float32)
        usable = min(observations.shape[0] - horizon, actions.shape[0] - horizon + 1)
        for start in range(max(0, usable)):
            obs_chunks.append(observations[start].flatten())
            action_chunks.append(actions[start : start + horizon])

    if not obs_chunks:
        raise ValueError(f"No action chunks could be extracted from {dataset_id!r}.")

    return ActionChunkDataset(torch.stack(obs_chunks), torch.stack(action_chunks))
