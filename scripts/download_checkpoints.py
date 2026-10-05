"""Download the policy checkpoints of the paper into ``checkpoints/``.

usage: python scripts/download_checkpoints.py [--out checkpoints] [--only revolute multitask]

Each file holds the EMA weights of one Diffusion Policy and the configuration needed to
rebuild it (about 355 MB); the four per-object policies and the multi-task policy are
listed in ``CHECKPOINTS`` with their SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/erickun0125/constraint-decomposed-impedance-control/releases/download/v1.0.0"

CHECKPOINTS = {
    "revolute": "5f125e5c6814a5401c2e7a37d103c70c102ce4e3a2f54a6d91b89ca2acac117f",
    "cylindrical": "1f2dfc11128e5fe6e7ff17395c699b4beb0cb955929e9f072c77d0f5368659e1",
    "planar": "ea4d4b40bb568293251955f9727484c91090930ad1c0812986f7cd212e98be66",
    "universal": "6080bfcb6802deed0e63cf52f0f3e79b86511b81485f0025c34448ae38a03ab9",
    "multitask": "01d720113a889ed74b0667d60afd8abd363c9285d7d22bc0e6f5881b33457d86",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints",
                        help="directory to save the checkpoints in")
    parser.add_argument("--only", nargs="+", choices=sorted(CHECKPOINTS), default=sorted(CHECKPOINTS),
                        help="checkpoints to download (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.only:
        path = args.out / f"{name}.pth"
        if path.exists() and sha256(path) == CHECKPOINTS[name]:
            print(f"{path} is up to date")
            continue
        partial = path.with_suffix(".pth.part")
        print(f"downloading {name}.pth ...", flush=True)
        with urllib.request.urlopen(f"{RELEASE_URL}/{name}.pth") as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        if sha256(partial) != CHECKPOINTS[name]:
            partial.unlink()
            raise RuntimeError(f"checksum mismatch for {name}.pth")
        partial.replace(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
