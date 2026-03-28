# Compatibility Matrix

## Executive Summary

This repository now documents a **real reference platform** rather than a vague “Python 3 + ROS1” requirement.

That does **not** mean the project is fully locked down. It means contributors and users now have a realistic target to align with when reproducing the stack on another machine.

## Reference Platform

| Layer | Reference Baseline | Status |
| --- | --- | --- |
| OS | Ubuntu 20.04.6 LTS | Observed locally |
| ROS | Noetic | Observed locally |
| Catkin workspace layout | Repository root with `src/` | Expected |
| ROS Python | `/usr/bin/python3` = Python 3.8.10 | Observed locally |
| ML Python | `anygrasp_env` = Python 3.9.25 | Observed locally |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU | Observed locally |
| Driver | 570.169 | Observed locally |
| CUDA runtime | 12.8 | Observed locally |

## Core Python Packages Seen in the ML Environment

| Package | Version | Notes |
| --- | --- | --- |
| `torch` | `2.8.0+cu128` | CUDA-sensitive; install to match your machine |
| `numpy` | `1.23.4` | Observed locally |
| `open3d` | `0.19.0` | Used in point-cloud workflows |
| `opencv-python` | `4.11.0.86` | Used by perception / camera utilities |
| `openai` | `2.26.0` | Used by VLM-facing nodes |
| `segment-anything` | `1.0` | SAM runtime dependency |

## Runtime Split You Should Expect

This project is easiest to understand as two cooperating runtimes:

| Runtime | Typical Interpreter | Main Responsibility |
| --- | --- | --- |
| ROS / catkin side | `/usr/bin/python3` on Ubuntu 20.04 | ROS nodes, launch chain, `cv_bridge`, MoveIt integration |
| ML / perception side | Dedicated Python 3.9 environment | SAM, GraspNet, Open3D, Torch, OpenAI SDK |

This split is normal for the current project, but it is also the main portability risk.

## What Looks Safest Right Now

| Combination | Confidence | Why |
| --- | --- | --- |
| Ubuntu 20.04 + ROS Noetic + separate Python 3.9 ML env | Highest | Closest to the observed baseline |
| NVIDIA GPU + CUDA-compatible Torch build | High | Aligns with current GraspNet and SAM inference assumptions |
| Perception-only path before full robot integration | High | Smallest validation surface |

## What Looks Risky or Untested

| Combination | Risk |
| --- | --- |
| Ubuntu 22.04 or other non-reference distros | ROS1 and package compatibility drift |
| One single Python interpreter for both ROS and ML sides | `cv_bridge` / Torch / Open3D mismatch risk |
| CPU-only execution for the full stack | Likely too slow or not representative for GraspNet |
| Different CUDA major version without reinstalling Torch appropriately | Import or runtime failure risk |

## Known Sensitive Boundaries

1. `cv_bridge` depends on the ROS-side Python environment.
2. `torch` must match the CUDA runtime expected by the local machine.
3. `open3d` and `segment_anything` are sensitive to the ML interpreter and native library stack.
4. `ANYGRASP_PYTHON` and `ROS_PYTHON_EXEC` are not cosmetic variables; they are how the two runtimes stay separated.

## Recommended Validation Ladder

1. Run `bash scripts/check_export_env.sh`
2. Validate `roslaunch sam_perception sam_py.launch`
3. Validate `roslaunch sam_perception run_graspnet.launch`
4. Validate `roslaunch sam_perception llm.launch` if VLM is needed
5. Attempt `roslaunch sam_perception system_new.launch` only after the previous steps are stable

## Related Files

- `environment/anygrasp_env.reference.yml`
- `requirements/anygrasp-reference.txt`
- `docs/QUICKSTART.md`
- `docs/PORTABILITY_AUDIT.md`
