"""Train the Diffusion Policy on scripted demonstrations (paper, Appendix H, Diffusion Policy table).

One demonstration file trains a per-object policy; several files with multi_task.json train
one multi-task policy that samples the objects equally (``normalize_weights_by_ds_size``)
and receives no task identifier:

    python scripts/train_policy.py --config configs/policy/single_task.json \\
        --dataset datasets/revolute.hdf5 --out outputs/train/revolute

    python scripts/train_policy.py --config configs/policy/multi_task.json \\
        --dataset datasets/revolute.hdf5 datasets/cylindrical.hdf5 \\
                  datasets/planar.hdf5 datasets/universal.hdf5 --out outputs/train/multitask

Recipe (configs/policy): AdamW with betas (0.95, 0.999) and weight decay 1e-6; learning rate
1e-4 with 500 warm-up steps and cosine decay; batch 64; 500 steps per epoch; EMA of the
weights; 100 DDPM training steps; ResNet-18 + spatial-softmax image encoder. Actions are
relative pose chunks (:mod:`con_dec_imp.policy.dataset`). Validation uses the
``mask/valid`` demonstrations of every file (40 of 400), normalized with the statistics of
that split. A checkpoint ``model_epoch_<N>.pth`` is written every
``experiment.save.every_n_epochs`` epochs.

robomimic advances the learning-rate schedule on validation batches too, so the validation
cadence is part of the schedule. The paper checkpoints used validation every 10 epochs for
the revolute, cylindrical and planar objects (single_task.json), every 50 for the universal
joint (``--validate_every 50``) and every 200 for the multi-task policy (multi_task.json,
2000 epochs). The schedule always spans ``train.num_epochs``; ``--stop_epoch`` ends training
earlier without changing it.

The configs are complete robomimic configs. Training reads their ``train``, ``observation``
and ``algo`` sections and the epoch, validation and save settings of ``experiment``; the
rollout, rendering and video settings are not used, and evaluation samples with DDIM
(:data:`con_dec_imp.settings.ESTIMATION`), not with the configs' inference settings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.config import config_factory
from robomimic.utils import file_utils, obs_utils, tensor_utils, train_utils
from robomimic.utils.log_utils import DataLogger, PrintLogger
from torch.utils.data import DataLoader

from con_dec_imp.policy.dataset import make_dataset

ACTION_RANGE_TOLERANCE = 1e-3
"""Allowed excess of the normalized actions over [-1, 1] (float32 rounding of rot6d entries)."""


class DiffusionPolicyTrainer(DiffusionPolicyUNet):
    """robomimic's Diffusion Policy with the AdamW betas of the config.

    The network, noise schedule and EMA are robomimic's; only the optimizer betas and the
    one-time action range check differ.
    """

    def _create_optimizers(self) -> None:
        super()._create_optimizers()
        betas = tuple(self.optim_params["policy"]["betas"])
        for group in self.optimizers["policy"].param_groups:
            group["betas"] = betas

    def process_batch_for_training(self, batch: dict) -> dict:
        """Slice the observation and prediction horizons; range-check the first batch.

        Relative rotation entries are unnormalized and can exceed 1 by float32 rounding, so
        the check allows :data:`ACTION_RANGE_TOLERANCE` and clamps the checked batch.
        """
        horizon = self.algo_config.horizon
        actions = batch["actions"][:, : horizon.prediction_horizon, :]
        if not self.action_check_done:
            bound = 1.0 + ACTION_RANGE_TOLERANCE
            if not torch.all((actions >= -bound) & (actions <= bound)):
                raise ValueError("normalized actions leave [-1, 1]; check the action normalization")
            actions = actions.clamp(-1.0, 1.0)
            self.action_check_done = True
        processed = {
            "obs": {k: v[:, : horizon.observation_horizon] for k, v in batch["obs"].items()},
            "goal_obs": batch.get("goal_obs", None),
            "actions": actions,
        }
        return tensor_utils.to_device(tensor_utils.to_float(processed), self.device)


def load_config(path: str, datasets: list[str], output: Path):
    """robomimic config from a JSON file, with the dataset paths and output directory set.

    Returns the config and the validation cadence ``experiment.validation_every_n_epochs``,
    which this script handles itself.
    """
    with open(path) as f:
        external = json.load(f)
    validate_every = external["experiment"].pop("validation_every_n_epochs")
    config = config_factory(external["algo_name"])
    with config.unlocked():
        config.update(external)
        config.train.data = [{"path": str(Path(p).resolve()), "weight": 1.0} for p in datasets]
        config.train.output_dir = str(output)
    return config, validate_every


def save_checkpoint(path: Path, model, config, shape_meta: dict, action_stats: dict) -> None:
    """robomimic-style checkpoint (weights, EMA weights, optimizer, config, normalization)."""
    torch.save(
        {
            "algo_name": config.algo_name,
            "config": config.dump(),
            "shape_metadata": shape_meta,
            "action_normalization_stats": tensor_utils.to_list(action_stats),
            "model": model.serialize(),
        },
        path,
    )


def train(config, output: Path, validate_every: int, stop_epoch: int) -> None:
    """robomimic's training loop with sparse validation and periodic checkpoints."""
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.train.cuda else "cpu")
    logging = config.experiment.logging
    if logging.terminal_output_to_txt:
        sys.stdout = sys.stderr = PrintLogger(str(output / "log.txt"))
    data_logger = DataLogger(str(output), config, log_tb=logging.log_tb, log_wandb=False)

    obs_utils.initialize_obs_utils_with_config(config)
    shape_meta = file_utils.get_shape_metadata_from_dataset(
        dataset_config=config.train.data[0],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
    )
    obs_keys = shape_meta["all_obs_keys"]
    trainset = make_dataset(config, obs_keys, config.train.hdf5_filter_key)
    action_stats = trainset.get_action_normalization_stats()
    validate = config.experiment.validate
    validset = make_dataset(config, obs_keys, config.train.hdf5_validation_filter_key) if validate else None
    print(trainset, validset, sep="\n")

    def loader(dataset, num_workers: int) -> DataLoader:
        sampler = dataset.get_dataset_sampler()
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config.train.batch_size,
            shuffle=sampler is None,
            num_workers=num_workers,
            drop_last=True,
        )

    train_loader = loader(trainset, config.train.num_data_workers)
    valid_loader = loader(validset, min(config.train.num_data_workers, 1)) if validate else None

    train_steps = config.experiment.epoch_every_n_steps
    num_epochs = config.train.num_epochs
    with config.unlocked():  # length of the cosine schedule
        config.algo.optim_params.policy.num_train_batches = train_steps
        config.algo.optim_params.policy.num_epochs = num_epochs
    model = DiffusionPolicyTrainer(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    with open(output / "config.json", "w") as f:
        f.write(config.dump())

    for epoch in range(1, stop_epoch + 1):
        logs = {"Train": train_utils.run_epoch(model, train_loader, epoch, num_steps=train_steps)}
        model.on_epoch_end(epoch)
        if validate and (epoch % validate_every == 0 or epoch == num_epochs):
            with torch.no_grad():
                logs["Valid"] = train_utils.run_epoch(
                    model,
                    valid_loader,
                    epoch,
                    validate=True,
                    num_steps=config.experiment.validation_epoch_every_n_steps,
                )
        for split, log in logs.items():
            print(f"{split} Epoch {epoch}\n{json.dumps(log, sort_keys=True, indent=4)}")
            for key, value in log.items():
                group = f"Timing_Stats/{split}_{key[5:]}" if key.startswith("Time_") else f"{split}/{key}"
                data_logger.record(group, value, epoch)
        if epoch % config.experiment.save.every_n_epochs == 0:
            save_checkpoint(output / f"model_epoch_{epoch}.pth", model, config, shape_meta, action_stats)
    data_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="robomimic config (configs/policy/*.json)")
    parser.add_argument("--dataset", required=True, nargs="+", help="demonstration HDF5 file(s)")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--validate_every", type=int, default=None,
        help="validation cadence in epochs (default: experiment.validation_every_n_epochs)",
    )
    parser.add_argument(
        "--stop_epoch", type=int, default=None,
        help="last epoch to train; the schedule still spans train.num_epochs (default: train.num_epochs)",
    )
    args = parser.parse_args()

    if len(set(map(os.path.realpath, args.dataset))) != len(args.dataset):
        parser.error("--dataset lists the same file twice")
    config, validate_every = load_config(args.config, args.dataset, args.out.resolve())
    if args.validate_every is not None:
        validate_every = args.validate_every
    stop_epoch = config.train.num_epochs if args.stop_epoch is None else args.stop_epoch
    if not 1 <= stop_epoch <= config.train.num_epochs:
        parser.error(f"--stop_epoch must lie in [1, {config.train.num_epochs}]")
    args.out.mkdir(parents=True, exist_ok=False)
    train(config, args.out.resolve(), validate_every, stop_epoch)


if __name__ == "__main__":
    main()
