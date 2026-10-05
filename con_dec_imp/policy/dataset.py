"""Training dataset of the Diffusion Policy (paper, Appendix H, "Simulation demonstration data").

The demonstration files store *absolute* end-effector poses as actions, one 10-vector
``(position, rot6d, gripper)`` per 0.1 s step. The policy is trained on *relative* chunks
(:mod:`con_dec_imp.policy.actions`): the H-step chunk starting at step ``t`` is expressed
in the end-effector frame observed at ``t``, one anchor for the whole chunk. Chunks that
run past the end of a demonstration repeat its last action (robomimic's sequence padding).

Normalization follows the chunks the dataset actually serves: the min-max statistics are
taken over every relative chunk row, padding included. The rotation entries (rot6d) are
left unnormalized, since they already lie in ``[-1, 1]``; the gripper command is min-max
normalized like the position. Observations are returned for step ``t`` only (one-step
observation history).
"""

from __future__ import annotations

import numpy as np
import torch
from robomimic.utils import obs_utils
from robomimic.utils.dataset import MetaDataset, SequenceDataset, action_stats_to_normalization_stats

from con_dec_imp.policy.actions import ACTION_DIM, absolute_to_relative

ROT6D = slice(3, 9)
"""Rotation entries of an action step."""


def identity_rotation_normalization(stats: dict) -> dict:
    """Set unit scale and zero offset on the rot6d entries of robomimic normalization stats."""
    for key_stats in stats.values():
        key_stats["scale"][..., ROT6D] = 1.0
        key_stats["offset"][..., ROT6D] = 0.0
    return stats


class RelativeChunkDataset(SequenceDataset):
    """robomimic sequence dataset that serves relative action chunks.

    Requires the observation key ``eef_pose`` (position + rot6d), one-step observation
    history and end-of-demonstration padding (``pad_seq_length=True``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "eef_pose" not in self.obs_keys:
            raise ValueError("the chunk anchor needs the observation key 'eef_pose'")
        if self.n_frame_stack != 1 or not self.pad_seq_length:
            raise ValueError("relative chunks need frame_stack=1 and pad_seq_length=True")
        if tuple(self.action_keys) != ("actions",):
            raise ValueError("expected the single action key 'actions'")

    def get_action_traj(self, ep: str) -> dict:
        """All relative chunk rows of one demonstration, ``(T * H, 10)``, for the statistics."""
        actions = self.hdf5_file[f"data/{ep}/actions"][()].astype("float32")
        anchors = self.hdf5_file[f"data/{ep}/obs/eef_pose"][()].astype("float32")
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D actions, got {actions.shape}")
        num_steps, horizon = actions.shape[0], self.seq_length
        padded = np.concatenate([actions, np.repeat(actions[-1:], horizon - 1, axis=0)], axis=0)
        chunks = padded[np.arange(num_steps)[:, None] + np.arange(horizon)[None, :]]
        relative = absolute_to_relative(torch.from_numpy(chunks), torch.from_numpy(anchors))
        return {"actions": relative.numpy().reshape(-1, ACTION_DIM)}

    def get_action_normalization_stats(self) -> dict:
        """Min-max statistics of the relative chunk rows, identity on the rotation entries."""
        if self.action_normalization_stats is None:
            stats = action_stats_to_normalization_stats(self.get_action_stats(), self.action_config)
            self.action_normalization_stats = identity_rotation_normalization(stats)
        return self.action_normalization_stats

    def get_item(self, index: int) -> dict:
        """Observation at step ``t`` and the normalized relative chunk ``t .. t + H - 1``."""
        demo_id = self._index_to_demo_id[index]
        t = index - self._demo_id_to_start_indices[demo_id]
        chunk = self.get_dataset_sequence_from_demo(
            demo_id, index_in_demo=t, keys=self.action_keys, seq_length=self.seq_length
        )["actions"]
        obs = self.get_obs_sequence_from_demo(demo_id, index_in_demo=t, keys=self.obs_keys, seq_length=1)
        anchor = torch.from_numpy(np.asarray(obs["eef_pose"][0], dtype=np.float32))
        relative = absolute_to_relative(torch.from_numpy(np.asarray(chunk, dtype=np.float32)), anchor)
        actions = obs_utils.normalize_dict(
            {"actions": relative.numpy()}, normalization_stats=self.get_action_normalization_stats()
        )["actions"]
        return {"obs": obs, "actions": actions, "index": index}


class RelativeChunkMetaDataset(MetaDataset):
    """Several relative-chunk datasets with one set of action statistics over all of them.

    Used for the multi-task policy: robomimic draws samples with per-dataset weights
    (``normalize_weights_by_ds_size=True`` with equal weights samples each object equally).
    """

    def __init__(self, datasets, ds_weights, normalize_weights_by_ds_size=False):
        super().__init__(datasets, ds_weights, normalize_weights_by_ds_size)
        self.set_action_normalization_stats(identity_rotation_normalization(self.action_normalization_stats))


def make_dataset(config, obs_keys, filter_key: str):
    """Dataset of one split (``filter_key``) over all files in ``config.train.data``.

    Its action statistics are computed here, from this split. The file handles are closed
    afterwards so that every data-loader worker opens its own.
    """
    kwargs = dict(
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        load_next_obs=config.train.hdf5_load_next_obs,
        filter_by_attribute=filter_key,
    )
    datasets = [RelativeChunkDataset(hdf5_path=entry["path"], **kwargs) for entry in config.train.data]
    if len(datasets) == 1:
        dataset = datasets[0]
        dataset.get_action_normalization_stats()
    else:
        dataset = RelativeChunkMetaDataset(
            datasets,
            [entry.get("weight", 1.0) for entry in config.train.data],
            normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        )
    for member in datasets:
        member.close_and_delete_hdf5_handle()
    return dataset
