# Portability Audit

## Executive Verdict

**Verdict: reproducible with setup, now backed by a reference environment and CI, but still not plug-and-play.**

From a professional team perspective, this repository is already strong as a **research system snapshot**, but it is not yet a **distribution-grade robotics product**.

## Reviewer Matrix

| Role | Verdict | Main Concern | Severity |
| --- | --- | --- | --- |
| Release / Platform Engineer | Better than before, still not one-command onboarding | ROS base stack and CUDA-aware Python environment are not fully automated | Blocking |
| Robotics Integration Engineer | Technically coherent | Full integration still depends on external ROS ecosystems and a simplified exported world asset | High |
| Perception / ML Engineer | Core model chain is meaningful | SAM weights and Torch/CUDA compatibility are still manual | High |
| Security Engineer | Safer than the original local workspace | Licensing and provenance are still incomplete | Medium |
| QA / Reliability Engineer | Baseline repo validation exists now | No clean-machine end-to-end runtime acceptance yet | High |
| Developer Experience Engineer | Documentation is substantially better | Setup still assumes ROS experience | Medium |

## 1. Release / Platform Engineer Review

### What is good

- The repository now contains the previously scattered packages in one place.
- Absolute Python interpreter paths in key launch files were replaced with environment-variable-based overrides.
- A self-check script was added to reduce setup ambiguity.
- A reference Python environment and compatibility matrix now describe the baseline more concretely.
- A lightweight GitHub Actions workflow now validates repository hygiene and Python syntax.

### What still blocks portability

1. The repository still does not provide a fully automated bootstrap such as:
   - Dockerfile
   - Dev Container
   - `rosdep`-driven install script
   - clean-machine setup script
2. `sam_vit_b_01ec64.pth` is intentionally omitted.
3. Full-stack ROS dependencies are still assumed to exist on the target machine.
4. Torch installation remains CUDA-sensitive and is not hard-locked inside the repository.

### Release verdict

**Not ready for honest “clone and run” distribution.**

## 2. Robotics Integration Engineer Review

### What is good

- The dataflow from bbox -> mask -> point cloud -> grasp pose is clear.
- `system_new.launch` acts as a recognizable integration entry point.
- Panda, MoveIt, Gazebo, and scheduler are conceptually wired together in a coherent way.
- The exported `supermarket.world` no longer depends on a machine-local `Cracker_Box` mesh path.

### What still blocks portability

1. Full execution still depends on external ROS packages such as Franka, MoveIt, Gazebo, and link-attacher components.
2. The exported `Cracker_Box` is now represented by a self-contained primitive placeholder, which improves portability but not visual fidelity.
3. Certain launch paths are legacy and still mix historical flows with the newer main pipeline.

### Integration verdict

**Strong prototype, still environment-coupled.**

## 3. Perception / ML Engineer Review

### What is good

- The project combines VLM grounding, SAM segmentation, point-cloud extraction, and GraspNet in a useful robotics stack.
- `grasp_from_sam.py` supports a repository-local default GraspNet root.
- The included `checkpoint-rs.tar` improves reproducibility for GraspNet-side inference.
- The reference platform now records the real observed versions of `torch`, `open3d`, `opencv-python`, `openai`, and `segment-anything`.

### What still blocks portability

1. SAM weights must be provided manually.
2. CUDA / PyTorch / Open3D / `cv_bridge` compatibility is still not hard-pinned end to end.
3. The project still relies on a split runtime:
   - ROS / catkin side on system Python
   - ML / perception side on a dedicated Python environment
4. Some scripts assume a Conda-style mixed environment and manually rewrite `sys.path`.
5. `llm_planner.py` still uses terminal `input()` and is not yet automation-friendly.

### ML verdict

**Research-ready, not packaging-ready.**

## 4. Security Engineer Review

### Improvements already made

- Hardcoded DashScope API keys were removed from the exported repository.
- Key-based scripts now read from environment variables.
- `.env.example` and environment docs make secret expectations more explicit.

### Remaining issues

1. Several package licenses still remain `TODO`.
2. Third-party provenance should be documented more explicitly.
3. There is still no top-level `LICENSE` / `NOTICE` strategy for the combined export.

### Security verdict

**Much safer than the local original, but compliance is incomplete.**

## 5. QA / Reliability Engineer Review

### What is good

- The repository includes a validation entry script.
- Core launch entry points are documented.
- A GitHub Actions workflow now checks shell syntax, Python syntax, required docs, and machine-local path regressions.
- A reference machine profile is now documented.

### What is missing

1. No runtime smoke-test matrix.
2. No automated `roslaunch`-level validation in CI.
3. No recorded clean-machine acceptance run from `clone` to stable launch.
4. No regression tests for topic contracts or launch composition.

### QA verdict

**Repository-level checks exist, runtime validation still manual.**

## 6. Developer Experience / Documentation Review

### What is good

- The repository now has architecture, quickstart, compatibility, portability audit, and third-party documents.
- The root README has a clearer system narrative and launch guidance.
- The environment baseline is no longer abstract; it references an actual observed working stack.
- The repository is now much easier to understand than the original local layout.

### What can still be improved

1. Add screenshots or actual runtime figures from RViz / Gazebo.
2. Add one “golden path” demo with expected topics and screenshots.
3. Separate the “main path” from legacy experiments more aggressively.

## Blocking Issues Before Claiming “Download and Run”

| Blocker | Why it matters | Owner |
| --- | --- | --- |
| Missing SAM checkpoint in repo | Perception startup fails without it | ML / Release |
| No automated ROS / Gazebo / Franka bootstrap | New machines still require manual ecosystem setup | Release / Robotics |
| No hard-locked Python / CUDA stack | Same repo may behave differently across machines | Release / ML |
| No clean-machine test record | Cannot honestly claim plug-and-play readiness | QA |

## Recommended Upgrade Path

### Phase 1: Make the repo honestly reproducible

1. Add Docker or Dev Container support.
2. Add a documented `rosdep` bootstrap path.
3. Record one clean-machine acceptance run and publish the exact steps.

### Phase 2: Make the repo onboarding-friendly

1. Add runtime smoke tests for the perception-only path.
2. Add screenshots and demo artifacts.
3. Publish a “golden path” walkthrough with expected outputs.

### Phase 3: Make the repo distribution-grade

1. Add end-to-end CI or nightly validation on a prepared robotics runner.
2. Finalize licensing and third-party notices.
3. Separate the core supported path from legacy experiments and copied launch files.
