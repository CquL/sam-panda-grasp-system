#!/usr/bin/env python3
"""Inspect LeRobot dataset inputs and actions.

Example:
    python learn/inspect_lerobot_shapes.py --repo-id lhj/panda_pickcube_demo_v1
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
from collections.abc import Mapping


def describe_value(name: str, value, indent: str = "") -> None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if isinstance(value, str):
        preview = value[:80]
    elif shape is None:
        preview = repr(value)
    elif shape == ():
        preview = str(value.item())
    else:
        preview = ""

    print(f"{indent}{name}: type={type(value).__name__}, shape={shape}, dtype={dtype}, value={preview}")


def describe_mapping(title: str, item: Mapping[str, object]) -> None:
    print(f"\n{title}")
    for key, value in item.items():
        describe_value(key, value, indent="  ")


def preload_conda_libstdcpp() -> None:
    """Prefer conda's newer C++ runtime before torchcodec loads FFmpeg libs."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    candidate = Path(conda_prefix) / "lib" / "libstdc++.so.6"
    if candidate.exists():
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lhj/panda_pickcube_demo_v1")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    preload_conda_libstdcpp()

    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id)

    print(dataset)
    print(f"Number of frames: {len(dataset)}")

    print("\nDataset features")
    for key, feature in dataset.features.items():
        print(f"  {key}: {feature}")

    sample = dataset[args.idx]
    describe_mapping(f"Single frame sample at idx={args.idx}", sample)

    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
    describe_mapping(f"Batch sample with batch_size={args.batch_size}", batch)

    print("\nInterpretation")
    print("  observation.* are model inputs.")
    print("  action is the supervised target the policy learns to predict.")
    print("  next.* and indexes are labels/metadata used for training, episode slicing, or evaluation.")


if __name__ == "__main__":
    main()
