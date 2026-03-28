# Architecture

## 1. System View

```mermaid
flowchart TD
    subgraph Inputs
        A1[User Command]
        A2[RGB Camera]
        A3[Depth Camera]
        A4[CameraInfo]
        A5[Wrist Camera]
    end

    subgraph Perception
        B1[LLM Planner]
        B2[SAM Perception Node]
        B3[Wrist VLM Node]
    end

    subgraph Geometry
        C1[Object Point Cloud]
        C2[Background Point Cloud]
    end

    subgraph Grasping
        D1[GraspNet Inference]
        D2[Candidate Grasp Poses]
    end

    subgraph Execution
        E1[GTSP Scheduler]
        E2[MoveIt]
        E3[Panda in Gazebo]
    end

    A1 --> B1
    A2 --> B1
    A2 --> B2
    A3 --> B2
    A4 --> B2
    B1 --> B2
    B2 --> C1
    B2 --> C2
    C1 --> D1
    D1 --> D2
    D2 --> E1
    E1 --> E2
    E2 --> E3
    A5 --> B3
```

## 2. Main Runtime Chains

### Chain A: Command -> Segmentation -> Grasp

```mermaid
sequenceDiagram
    participant U as User
    participant L as llm_planner.py
    participant S as sam_node.py
    participant G as grasp_from_sam.py
    participant M as MoveIt / Scheduler

    U->>L: Natural-language command
    L->>S: Publish /sam/prompt_bbox
    S->>S: Run SAM on RGB image
    S->>G: Publish /sam_perception/object_cloud
    G->>G: Run GraspNet inference
    G->>M: Publish grasp candidates
```

### Chain B: Wrist Recheck

```mermaid
sequenceDiagram
    participant W as Wrist Camera
    participant V as wrist_vlm_node.py
    participant P as Pick Pipeline

    W->>V: RGB image
    P->>V: Trigger message
    V->>P: /wrist_vlm/bbox
```

## 3. Key ROS Topics

| Topic | Publisher | Consumer | Meaning |
| --- | --- | --- | --- |
| `/camera/color/image_raw` | RGB camera | `sam_node.py`, `llm_planner.py`, `yolov5_ros` | Main color stream |
| `/camera/depth/image_raw` | Depth camera | `sam_node.py` | Depth stream for 3D reconstruction |
| `/camera/color/camera_info` | Camera driver | `sam_node.py` | Intrinsics |
| `/sam/prompt_bbox` | `llm_planner.py` | `sam_node.py` | Target bounding boxes |
| `/sam_perception/object_cloud` | `sam_node.py` | `grasp_from_sam.py` | Foreground object cloud |
| `/sam_perception/background_cloud` | `sam_node.py` | downstream debug / environment logic | Background cloud |
| `/graspnet/grasp_pose_array_raw` | `grasp_from_sam.py` | scheduler / planner | Candidate grasp poses |
| `/graspnet/grasp_info_raw` | `grasp_from_sam.py` | scheduler / planner | Width, score, depth triplets |
| `/wrist_camera/color/image_raw` | Wrist camera | `wrist_vlm_node.py` | Wrist re-observation |
| `/wrist_vlm/trigger` | pick pipeline | `wrist_vlm_node.py` | Trigger VLM re-detection |
| `/wrist_vlm/bbox` | `wrist_vlm_node.py` | pick pipeline | Wrist-stage target box |

## 4. Package Layout

### `src/sam_perception`

- `scripts/sam_node.py`: SAM segmentation, mask fusion, point-cloud generation.
- `scripts/grasp_from_sam.py`: GraspNet inference from segmented object cloud.
- `scripts/llm_planner.py`: VLM-assisted bbox generation from user command.
- `scripts/wrist_vlm_node.py`: Wrist-camera target re-localization.
- `launch/*.launch`: Runtime entry points.

### `src/panda_pick_place`

- Gazebo worlds and object models.
- Demo scripts and scheduler.
- Panda-side runtime logic for pick and place.

### `src/panda_moveit_config`

- MoveIt launch chain.
- Panda planning config.
- Gazebo bridge and controller configuration.

### `third_party/graspnet-baseline`

- Local GraspNet inference code.
- `checkpoint-rs.tar` included in this export.

## 5. Launch Entry Points

| Launch file | Purpose | Recommended use |
| --- | --- | --- |
| `sam_perception/sam_py.launch` | Start SAM segmentation node only | Validate perception |
| `sam_perception/run_graspnet.launch` | Start GraspNet node only | Validate point-cloud -> grasp inference |
| `sam_perception/llm.launch` | Start VLM command planner | Validate text/image grounding |
| `sam_perception/system_new.launch` | Full integrated system | Main end-to-end launch |
| `panda_pick_place/supermarket_sim.launch` | Legacy YOLO + grasp detector simulation flow | Optional / historical |

## 6. Architecture Risks

### Good news

- The core perception-to-grasp chain is conceptually clean.
- Major modules are separated by ROS topics.
- The export now groups previously scattered workspaces into one repo.

### Current engineering debt

- Some integrated flows still depend on external ROS ecosystems not vendored here.
- World assets are not fully self-contained.
- Runtime behavior still depends on Python environment compatibility.
- Some packages remain research-grade and rely on manual operator knowledge.
