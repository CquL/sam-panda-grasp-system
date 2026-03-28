# Quickstart

## 1. What this guide aims to do

This guide helps a new machine reach one of two goals:

- **Perception-only validation**
- **Full integrated Panda + MoveIt + Gazebo run**

The first is much easier than the second.

## 2. Reference platform

The current exported repository is closest to this observed baseline:

| Item | Reference |
| --- | --- |
| OS | Ubuntu 20.04.6 LTS |
| ROS | Noetic |
| ROS-side Python | 3.8.10 |
| ML-side Python | 3.9.25 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Driver / CUDA | 570.169 / 12.8 |

This is a **reference baseline**, not a guarantee that only this exact machine can run the project.

## 3. Understand the two Python runtimes

This project uses a split runtime:

- **ROS / catkin side**
  Usually system Python on Ubuntu 20.04 + ROS Noetic.
- **ML / perception side**
  A dedicated Python 3.9 environment for SAM, GraspNet, Open3D, and the OpenAI SDK.

If you collapse everything into one interpreter without checking ABI compatibility, `cv_bridge`, `open3d`, and Torch-related imports are the first likely failure points.

## 4. Recreate the ML environment

If you use Conda, the closest starting point is:

```bash
conda env create -f environment/anygrasp_env.reference.yml
conda activate anygrasp_env
pip install -r requirements/anygrasp-reference.txt
```

Important notes:

- Install **Torch separately** with a build that matches your local CUDA runtime.
- Do **not** try to install `cv_bridge` from pip for this workflow; keep ROS Python bindings on the ROS side.
- If you do not use Conda, mirror the versions from:
  - `requirements/anygrasp-reference.txt`
  - `docs/COMPATIBILITY.md`

## 5. Prepare the ROS base system

This repository assumes a Linux + ROS1 + catkin ecosystem. In practice, you should prepare:

- ROS1 Noetic with Python 3
- Gazebo
- MoveIt
- Franka-related ROS packages
- OpenCV / `cv_bridge`
- ROS topic and transform tools such as `tf`

If your ROS environment is already available, `rosdep` is a good first pass:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

You may still need extra Franka / Gazebo packages depending on your machine and overlay setup.

## 6. Required files and variables

### Mandatory

- `third_party/graspnet-baseline/checkpoint-rs.tar`
  This file is already included in the export.
- `sam_vit_b_01ec64.pth`
  This file is **not** included and must be added manually.

### Environment variables

```bash
export ANYGRASP_PYTHON=/path/to/your/python
export ROS_PYTHON_EXEC=/usr/bin/python3
export DASHSCOPE_API_KEY=your_key
export SAM_CHECKPOINT_PATH=/absolute/path/to/sam_vit_b_01ec64.pth
```

Optional:

```bash
export LIBFFI_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export GRASPNET_ROOT=/absolute/path/to/graspnet-baseline
```

## 7. Run the environment self-check

```bash
bash scripts/check_export_env.sh
```

If the script reports failures, fix those first.

## 8. Build workspace

From repository root:

```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## 9. Perception-only validation path

Start SAM:

```bash
roslaunch sam_perception sam_py.launch
```

Start GraspNet:

```bash
roslaunch sam_perception run_graspnet.launch
```

Optional VLM planner:

```bash
roslaunch sam_perception llm.launch
```

## 10. Full integrated path

```bash
roslaunch sam_perception system_new.launch
```

This path additionally assumes:

- Panda MoveIt configuration is valid
- Gazebo world assets are complete
- Franka-related packages are available
- scheduler and execution chain are compatible with your machine

## 11. Known setup traps

### Trap 1: SAM starts but checkpoint is missing

Symptom:

- `sam_node.py` exits immediately with checkpoint not found.

Fix:

- put the checkpoint in `src/sam_perception/models/`
- or set `SAM_CHECKPOINT_PATH`

### Trap 2: GraspNet import fails

Symptom:

- `grasp_from_sam.py` cannot import GraspNet modules.

Fix:

- verify `third_party/graspnet-baseline/`
- or set `GRASPNET_ROOT`

### Trap 3: ROS / Gazebo / Franka packages are missing

Symptom:

- launch files fail with package-not-found errors
- `rospack find` cannot resolve `gazebo_ros`, `franka_description`, or related packages

Fix:

- source your ROS installation before building or launching
- run `rosdep install --from-paths src --ignore-src -r -y`
- install the missing Franka / Gazebo / MoveIt packages into the target machine

### Trap 4: Python environment mismatch

Symptom:

- `cv_bridge`, `open3d`, `segment_anything`, or `torch` import/runtime errors

Fix:

- use a dedicated Python environment
- point `ANYGRASP_PYTHON` to that interpreter

### Trap 5: The simulation runs but the exported cracker box looks simpler

Symptom:

- the supermarket scene loads, but the `Cracker_Box` object no longer uses the original textured mesh from the local machine.

Fix / explanation:

- this export intentionally replaced the machine-local mesh dependency with a self-contained primitive placeholder
- portability improved, but the visual asset is no longer identical to the original local scene

## 12. Honest expectation management

If you are testing on a fresh machine, the likely path is:

1. First make `sam_perception` run.
2. Then validate `graspnet-baseline`.
3. Then validate Gazebo + MoveIt.
4. Only after that attempt the full system launch.
