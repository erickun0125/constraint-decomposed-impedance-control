"""Demonstration files in robomimic's HDF5 layout (paper, Appendix H, "Simulation demonstration data").

One file holds the successful demonstrations of one object, written in the order the
episodes finished. Step ``t`` pairs the observation at the start of a 0.1 s policy period
with the waypoint commanded for that period::

    data/                         attrs: env_args, obs_meta (JSON)
        demo_<i>/                 attrs: num_samples (T), success
            obs/eef_pose          (T, 9)  float32  TCP position (environment frame) + rot6d
            obs/gripper           (T, 1)  float32  finger opening, 1 open, 0 closed
            obs/wrist_cam         (T, H, W, 3) uint8, gzip
            actions               (T, 10) float32  waypoint: TCP position (environment frame),
                                                   rot6d, gripper command (1 close, 0 open)
            rewards               (T,)    float64  zeros (the environments define no reward)
            dones                 (T,)    bool     True at the last step
    mask/train, mask/valid        demonstration names: the first ``train_fraction`` train

Root attribute: ``action_frame = "absolute"`` (the training dataset converts the actions to
relative chunks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import h5py
import numpy as np

OBS_KEYS = ("eef_pose", "gripper")
"""Low-dimensional observation keys."""

IMAGE_KEYS = ("wrist_cam",)
"""Image observation keys."""

IMAGE_COMPRESSION_LEVEL = 4
"""gzip level of the image datasets."""


@dataclass
class Demonstration:
    """One episode, ``T`` policy steps."""

    eef_pose: np.ndarray
    """``(T, 9)`` TCP position (environment frame) and rot6d at the start of each step."""
    gripper: np.ndarray
    """``(T, 1)`` finger opening at the start of each step."""
    wrist_cam: np.ndarray
    """``(T, H, W, 3)`` uint8 wrist image at the start of each step."""
    actions: np.ndarray
    """``(T, 10)`` commanded waypoint (environment frame) and gripper command."""
    rewards: np.ndarray
    """``(T,)`` environment reward summed over each step (zero: no reward is defined)."""

    @classmethod
    def from_steps(cls, steps: list[dict[str, np.ndarray]]) -> Demonstration:
        """Stack per-step records keyed by the field names."""
        return cls(**{f.name: np.stack([step[f.name] for step in steps]) for f in fields(cls)})

    def __len__(self) -> int:
        return len(self.actions)


class DemoWriter:
    """Writes demonstrations one by one; :meth:`close` adds the train/valid split masks.

    The file is written under a temporary name and appears at ``path`` only when closed.

    Args:
        path: Output file.
        env_name: Environment id stored in ``data.attrs["env_args"]``.
    """

    def __init__(self, path: Path, env_name: str):
        self.path = Path(path)
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._file = h5py.File(self._partial, "w")
        self._data = self._file.create_group("data")
        self._data.attrs["env_args"] = json.dumps({"env_name": env_name, "env_kwargs": {}, "type": 1})  # robomimic env type 1
        self._data.attrs["obs_meta"] = json.dumps(
            {"state_obs_keys": list(OBS_KEYS), "image_obs_keys": list(IMAGE_KEYS)}
        )
        self._file.attrs["action_frame"] = "absolute"
        self.num_demos = 0

    def write(self, demo: Demonstration) -> None:
        """Append ``demo`` as ``demo_<num_demos>``."""
        group = self._data.create_group(f"demo_{self.num_demos}")
        group["obs/eef_pose"] = demo.eef_pose.astype(np.float32)
        group["obs/gripper"] = demo.gripper.astype(np.float32)
        group.create_dataset(
            "obs/wrist_cam", data=demo.wrist_cam.astype(np.uint8),
            compression="gzip", compression_opts=IMAGE_COMPRESSION_LEVEL,
        )
        group["actions"] = demo.actions.astype(np.float32)
        group["rewards"] = demo.rewards.astype(np.float64)
        dones = np.zeros(len(demo), dtype=bool)
        dones[-1] = True
        group["dones"] = dones
        group.attrs["num_samples"] = len(demo)
        group.attrs["success"] = True
        self.num_demos += 1

    def close(self, train_fraction: float) -> Path:
        """Write the ``mask/train`` and ``mask/valid`` splits; return the file path."""
        if self.num_demos == 0:
            raise RuntimeError("no demonstrations were written")
        names = [f"demo_{i}" for i in range(self.num_demos)]
        num_train = int(self.num_demos * train_fraction)
        self._file["mask/train"] = np.array(names[:num_train], dtype="S")
        self._file["mask/valid"] = np.array(names[num_train:], dtype="S")
        self._file.close()
        self._partial.replace(self.path)
        return self.path

    def discard(self) -> None:
        """Close and delete the unfinished file."""
        self._file.close()
        self._partial.unlink(missing_ok=True)
