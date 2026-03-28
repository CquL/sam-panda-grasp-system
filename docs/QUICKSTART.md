# Quickstart

## 1. What this guide aims to do

This guide helps a new machine reach one of two goals:

- **Perception-only validation**
- **Full integrated Panda + MoveIt + Gazebo run**

The first is much easier than the second.

## 2. Required baseline environment

This repository assumes a Linux + ROS1 + catkin ecosystem. In practice, you should expect to prepare:

- ROS1 with Python 3
- Gazebo
- MoveIt
- Franka-related ROS packages
- OpenCV / `cv_bridge`
- PyTorch environment compatible with your GPU and CUDA

## 3. Required files and variables

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

## 4. Run the environment self-check

```bash
bash scripts/check_export_env.sh
```

If the script reports failures, fix those first.

## 5. Build workspace

From repository root:

```bash
catkin_make
source devel/setup.bash
```

## 6. Perception-only validation path

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

## 7. Full integrated path

```bash
roslaunch sam_perception system_new.launch
```

This path additionally assumes:

- Panda MoveIt configuration is valid
- Gazebo world assets are complete
- Franka-related packages are available
- scheduler and execution chain are compatible with your machine

## 8. Known setup traps

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

### Trap 3: Gazebo world fails to load object assets

Symptom:

- object meshes missing in supermarket scenes

Fix:

- vendor the missing models
- or replace external references with repository-local `model://` assets

### Trap 4: Python environment mismatch

Symptom:

- `cv_bridge`, `open3d`, `segment_anything`, or `torch` import/runtime errors

Fix:

- use a dedicated Python environment
- point `ANYGRASP_PYTHON` to that interpreter

## 9. Honest expectation management

If you are testing on a fresh machine, the likely path is:

1. First make `sam_perception` run.
2. Then validate `graspnet-baseline`.
3. Then validate Gazebo + MoveIt.
4. Only after that attempt the full system launch.
