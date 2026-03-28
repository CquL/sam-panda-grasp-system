# Environment Guide

## Why this directory exists

This repository uses a split runtime:

- **ROS / catkin side**
  Usually Ubuntu 20.04 + ROS Noetic + system Python 3.8
- **ML / perception side**
  A dedicated Python 3.9 environment for SAM, GraspNet, Open3D, and related packages

That split is normal for this project, but it is also the main source of cross-machine setup pain.

## Reference Files

- `anygrasp_env.reference.yml`
  A starting Conda environment definition for the ML side.
- `../requirements/anygrasp-reference.txt`
  A reference pip package list extracted from the observed local environment.

Neither file is a perfect lockfile. They are meant to document a realistic baseline and reduce guesswork.

## Reference Host Snapshot

| Item | Value |
| --- | --- |
| OS | Ubuntu 20.04.6 LTS |
| ROS | Noetic |
| ROS Python | 3.8.10 |
| ML Python | 3.9.25 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Driver / CUDA | 570.169 / 12.8 |

## Recommended Recreation Flow

If you use Conda:

```bash
conda env create -f environment/anygrasp_env.reference.yml
conda activate anygrasp_env
pip install -r requirements/anygrasp-reference.txt
```

Important:

1. Install **Torch separately** with a build that matches your local CUDA runtime.
2. Keep `cv_bridge` and ROS Python bindings on the ROS side instead of trying to rebuild them inside the ML environment.
3. Point `ANYGRASP_PYTHON` to this environment's Python interpreter.
4. Keep `ROS_PYTHON_EXEC` pointing to the ROS-side interpreter, usually `/usr/bin/python3`.

## Why Torch Is Not Hard-Pinned Here

The observed local environment uses `torch 2.8.0+cu128`.

Torch is intentionally not hard-wired into the reference environment file because:

- GPU and CUDA combinations vary across machines
- a mismatched wheel is one of the easiest ways to make the stack fail
- the correct choice depends on the target machine, not only on this repository

## Minimum Practical Expectation

If you want the highest chance of success on a fresh machine:

1. Match Ubuntu 20.04 + ROS Noetic first.
2. Recreate a dedicated Python 3.9 ML environment second.
3. Validate SAM only.
4. Validate GraspNet only.
5. Attempt the full Panda + MoveIt + Gazebo path last.
