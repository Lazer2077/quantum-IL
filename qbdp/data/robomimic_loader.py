from __future__ import annotations

import torch

from qbdp.data.synthetic import ActionChunkDataset


def load_robomimic_action_chunks(
    hdf5_path: str,
    obs_key: str = "robot0_eef_pos",
    horizon: int = 4,
    max_demos: int | None = None,
) -> ActionChunkDataset:
    """Load RoboMimic HDF5 demonstrations if h5py is installed."""

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "RoboMimic support is optional. Install qbdp[robomimic] to use this loader."
        ) from exc

    obs_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    with h5py.File(hdf5_path, "r") as handle:
        demos = list(handle["data"].keys())
        if max_demos is not None:
            demos = demos[:max_demos]
        for demo in demos:
            group = handle["data"][demo]
            observations = torch.as_tensor(group["obs"][obs_key][()], dtype=torch.float32)
            actions = torch.as_tensor(group["actions"][()], dtype=torch.float32)
            usable = min(observations.shape[0] - horizon, actions.shape[0] - horizon + 1)
            for start in range(max(0, usable)):
                obs_rows.append(observations[start].flatten())
                action_rows.append(actions[start : start + horizon])

    if not obs_rows:
        raise ValueError(f"No action chunks could be extracted from {hdf5_path!r}.")

    return ActionChunkDataset(torch.stack(obs_rows), torch.stack(action_rows))
