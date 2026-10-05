"""Convert a robomimic training checkpoint into a weight-only release checkpoint.

The release file holds the EMA weights, the config sections needed to rebuild the
network, the observation shapes and the action normalization
(:func:`con_dec_imp.policy.diffusion_policy.release_checkpoint`).

usage: python scripts/export_policy.py <training_checkpoint.pth> --out <release.pth>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from con_dec_imp.policy.diffusion_policy import release_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="robomimic training checkpoint (model_epoch_*.pth)")
    parser.add_argument("--out", type=Path, required=True, help="release checkpoint to write")
    args = parser.parse_args()

    training = torch.load(args.source, map_location="cpu", weights_only=True)
    release = release_checkpoint(training)
    del training
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, args.out)

    reloaded = torch.load(args.out, map_location="cpu", weights_only=True)
    if reloaded["model"].keys() != release["model"].keys() or not all(
        torch.equal(reloaded["model"][name], tensor) for name, tensor in release["model"].items()
    ):
        raise RuntimeError("the written checkpoint does not reload to the same weights")

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB, sha256 {digest}")


if __name__ == "__main__":
    main()
