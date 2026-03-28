# Sam Grasp System Export

This repository is a GitHub-safe export of the local robot grasping project assembled from several workspaces on the machine. It was prepared as a separate copy so the original runtime workspaces remain unchanged.

## Included directories

- `src/sam_perception`: ROS package for SAM segmentation, VLM-assisted target selection, and point-cloud generation.
- `src/panda_moveit_config`: MoveIt configuration used by the launch files referenced from `sam_perception`.
- `src/panda_pick_place`: Panda pick-and-place package used by the integrated launch flow.
- `third_party/graspnet-baseline`: Local GraspNet baseline code and local checkpoint used by `grasp_from_sam.py`.
- `src/CMakeLists.txt`: Catkin workspace top-level file.

## What was intentionally changed in this export

- The original `sam_vit_b_01ec64.pth` file was not committed because it exceeds GitHub's normal file size limit.
- Hardcoded DashScope API keys were removed. The exported scripts now read `DASHSCOPE_API_KEY` and fall back to `OPENAI_API_KEY`.
- `src/sam_perception/setup.py` was corrected to export the actual `sam_perception` Python package.
- `src/sam_perception/scripts/grasp_from_sam.py` now supports a repo-local default `third_party/graspnet-baseline` path and can still be overridden with `GRASPNET_ROOT`.
- `src/sam_perception/scripts/sam_node.py` now supports `SAM_CHECKPOINT_PATH` and respects the `~model_type` ROS parameter.

## Required manual step after cloning

Download the SAM checkpoint and place it at:

`src/sam_perception/models/sam_vit_b_01ec64.pth`

Or export:

`export SAM_CHECKPOINT_PATH=/absolute/path/to/sam_vit_b_01ec64.pth`

## Environment variables

- `DASHSCOPE_API_KEY`: API key for DashScope-compatible VLM calls.
- `OPENAI_API_KEY`: Optional fallback if you prefer to reuse the same env var name.
- `GRASPNET_ROOT`: Optional override for the GraspNet baseline directory.
- `SAM_CHECKPOINT_PATH`: Optional override for the SAM checkpoint path.

## Notes

- This export is designed to preserve the project structure without changing the original local workspaces.
- Some launch files still reference the original Conda interpreter path `/home/lhj/anaconda3/envs/anygrasp_env/bin/python`. Update those launch prefixes if your environment differs.
- `panda_moveit_config` and `panda_pick_place` were copied from separate local workspaces so the integrated launch chain is captured in one place.

