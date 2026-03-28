# Contributing

## Project Intent

This repository is a **research-grade robotics integration project** that is being actively cleaned up into a more portable public export.

Good contributions are the ones that make the project:

- easier to understand
- easier to reproduce
- less dependent on machine-local assumptions
- safer to share publicly

## High-Value Contribution Areas

- Environment automation
- Documentation clarity
- Launch-file cleanup
- Portability fixes
- Runtime smoke tests
- Licensing and provenance cleanup

## Contribution Rules for This Repository

1. Do not reintroduce machine-local absolute paths.
2. Do not commit secrets, tokens, or API keys.
3. Prefer environment variables over hardcoded interpreter paths.
4. Keep the export repository self-contained whenever possible.
5. If a scene asset cannot be shipped, document the gap explicitly instead of hiding it.

## Before Opening a Pull Request

Run these checks locally when possible:

```bash
bash -n scripts/check_export_env.sh
python3 -m py_compile \
  src/sam_perception/scripts/check_cam.py \
  src/sam_perception/scripts/check_models.py \
  src/sam_perception/scripts/grasp_from_sam.py \
  src/sam_perception/scripts/llm_planner.py \
  src/sam_perception/scripts/sam_node.py \
  src/sam_perception/scripts/wrist_vlm_node.py \
  src/sam_perception/src/sam_perception/core_grasp.py \
  src/sam_perception/src/sam_perception/core_llm.py \
  src/sam_perception/src/sam_perception/core_sam.py \
  src/panda_pick_place/scripts/control_panel.py \
  src/panda_pick_place/scripts/demo.py \
  src/panda_pick_place/scripts/gazebo_interface.py \
  src/panda_pick_place/scripts/grasp_test.py \
  src/panda_pick_place/scripts/gtsp_scheduler.py \
  src/panda_pick_place/scripts/motion_controller.py \
  src/panda_pick_place/scripts/moveit_learn.py \
  src/panda_pick_place/scripts/new.py
grep -R -nE '/home/[^/]+/' \
  src/sam_perception \
  src/panda_moveit_config \
  src/panda_pick_place
```

The final `grep` should return nothing.

## Documentation Standard

If you change anything about:

- launch entry points
- environment variables
- external dependencies
- model weights
- world assets

then update the relevant docs in the same change:

- `README.md`
- `docs/QUICKSTART.md`
- `docs/COMPATIBILITY.md`
- `docs/PORTABILITY_AUDIT.md`
- `docs/THIRD_PARTY.md`

## What Not To Promise

Until a clean-machine acceptance run exists, avoid claiming:

- one-click setup
- plug-and-play portability
- full cross-machine validation

It is better for the repository to be honest and precise than flashy but misleading.
