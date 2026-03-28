# Portability Audit

## Executive Verdict

**Verdict: reproducible with setup, not yet plug-and-play.**

From a professional team perspective, this repository is already strong as a **research system snapshot**, but it is not yet a **distribution-grade robotics product**.

## Reviewer Matrix

| Role | Verdict | Main Concern | Severity |
| --- | --- | --- | --- |
| Release / Platform Engineer | Not ready for one-command onboarding | Environment is not fully pinned | Blocking |
| Robotics Integration Engineer | Technically coherent | External simulation assets still missing | Blocking |
| Perception / ML Engineer | Core model chain is meaningful | Model weights and CUDA compatibility are manual | High |
| Security Engineer | Better than original local version | Secret management improved, licensing still incomplete | Medium |
| QA / Reliability Engineer | Useful for manual validation | No clean-machine acceptance run or CI | High |
| Developer Experience Engineer | Documentation now much better | Still requires advanced ROS background | Medium |

## 1. Release / Platform Engineer Review

### What is good

- The repository now contains the previously scattered packages in one place.
- Absolute Python interpreter paths in key launch files were replaced with environment-variable-based overrides.
- A self-check script was added to reduce setup ambiguity.

### What still blocks portability

1. The repository does not yet provide a locked environment definition such as:
   - `environment.yml`
   - Dockerfile
   - Dev Container
   - reproducible install script
2. `sam_vit_b_01ec64.pth` is intentionally omitted.
3. Full-stack ROS dependencies are assumed to exist on the target machine.

### Release verdict

**Not ready for “clone and run” distribution.**

## 2. Robotics Integration Engineer Review

### What is good

- The dataflow from bbox -> mask -> point cloud -> grasp pose is clear.
- `system_new.launch` acts as a recognizable integration entry point.
- Panda, MoveIt, Gazebo, and scheduler are conceptually wired together in a coherent way.

### What still blocks portability

1. `supermarket.world` still references a machine-local Cracker Box mesh path:
   - `/home/lhj/.gazebo/models/Cracker_Box/textured.obj`
2. Some Gazebo worlds depend on models that are not shipped inside this repository.
3. Certain launch paths are legacy and mix historical pipelines with current pipelines.

### Integration verdict

**Strong prototype, but still environment-coupled.**

## 3. Perception / ML Engineer Review

### What is good

- The project combines VLM grounding, SAM segmentation, point-cloud extraction, and GraspNet in a useful robotics stack.
- `grasp_from_sam.py` now supports a repository-local default GraspNet root.
- The included `checkpoint-rs.tar` improves reproducibility for GraspNet-side inference.

### What still blocks portability

1. SAM weights must be provided manually.
2. CUDA / PyTorch / Open3D / `cv_bridge` compatibility is not pinned.
3. Some scripts assume a Conda-style mixed environment and manually rewrite `sys.path`.
4. `llm_planner.py` still uses terminal `input()` and is not yet automation-friendly.

### ML verdict

**Research-ready, not packaging-ready.**

## 4. Security Engineer Review

### Improvements already made

- Hardcoded DashScope API keys were removed from the exported repository.
- Key-based scripts now read from environment variables.

### Remaining issues

1. Several package licenses still remain `TODO`.
2. Third-party provenance should be documented more explicitly.
3. There is no formal secret-handling policy or `.env` onboarding guidance beyond the example file.

### Security verdict

**Much safer than the local original, but compliance is incomplete.**

## 5. QA / Reliability Engineer Review

### What is good

- The repository now includes a validation entry script.
- Core launch entry points are documented.

### What is missing

1. No CI pipeline.
2. No smoke-test matrix.
3. No “tested on machine X / distro Y / CUDA Z” record.
4. No automated proof that a clean checkout can reach a stable launch state.

### QA verdict

**Manual validation only at the moment.**

## 6. Developer Experience / Documentation Review

### What is good

- The repository now has architecture, quickstart, and portability audit documents.
- The root README has a clearer system narrative and launch guidance.
- The repository is now easier to understand than the original local layout.

### What can still be improved

1. Add screenshots or actual runtime figures from RViz / Gazebo.
2. Add a compatibility matrix:
   - Ubuntu version
   - ROS distro
   - Python version
   - CUDA version
   - GPU driver
3. Add one “golden path” demo with expected outputs.

## Blocking Issues Before Claiming “Download and Run”

| Blocker | Why it matters | Owner |
| --- | --- | --- |
| Missing SAM checkpoint in repo | Perception startup fails without it | ML / Release |
| Missing Gazebo model assets | Worlds may fail to load correctly | Robotics |
| No pinned Python / CUDA environment | Same repo may behave differently across machines | Release / ML |
| No clean-machine test record | Cannot honestly claim plug-and-play readiness | QA |

## Recommended Upgrade Path

### Phase 1: Make the repo honestly reproducible

1. Add `environment.yml` or Docker.
2. Vendor or document all missing simulation assets.
3. Record one tested reference machine.

### Phase 2: Make the repo onboarding-friendly

1. Add setup scripts.
2. Add smoke tests.
3. Add screenshots and demo artifacts.

### Phase 3: Make the repo distribution-grade

1. Add CI.
2. Finalize licensing.
3. Separate “core path” from “legacy experiments”.
