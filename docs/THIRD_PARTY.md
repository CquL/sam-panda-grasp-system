# Third-Party Components

这个仓库不是纯从零开始写成的单体项目，而是一个围绕机器人抓取任务搭建的系统集成仓库。为方便后续维护和开源合规，这里把主要第三方组成说明清楚。

## Included Third-Party Code

### `third_party/graspnet-baseline`

- Role:
  提供 GraspNet 推理代码和本地抓取候选生成逻辑。
- Why included:
  `sam_perception/scripts/grasp_from_sam.py` 直接依赖它的模型结构和 checkpoint。
- Current status:
  已作为第三方目录纳入当前仓库，并保留本地 `checkpoint-rs.tar`。

### `src/yolov5_ros`

- Role:
  提供 YOLOv5 在 ROS 环境中的检测链路。
- Why included:
  某些仿真启动流仍然会用到它。
- Current status:
  被视为“可选但已包含”的支持包，而不是当前主路径的唯一依赖。

### `src/panda_moveit_config`

- Role:
  Panda 机械臂的 MoveIt / Gazebo 启动与规划配置。
- Why included:
  主系统 launch 文件直接引用这些配置。
- Current status:
  属于系统运行核心依赖之一。

## Local / External Resource Dependencies Still Not Fully Vendored

### SAM Checkpoint

- File:
  `sam_vit_b_01ec64.pth`
- Status:
  未随仓库直接提供，因为文件超过 GitHub 常规单文件限制。

### Gazebo Model Assets

- Example:
  `Cracker_Box` 在某些世界文件中仍引用机器本地路径。
- Status:
  这部分仍然是当前跨机器可移植性的主要阻塞项之一。

## Ownership Interpretation

从工程结构上，可以把这个仓库理解为三层：

1. **你的系统集成工作**
   主要体现在 `sam_perception`、Panda 联动、launch 组织、任务链串接、环境整理和实验结构设计。
2. **你本地项目中的支持包**
   例如 `panda_pick_place`、`detection_msgs`、`grasp_detector_ros`。
3. **外部引入或基于第三方扩展的组件**
   例如 `graspnet-baseline`、`yolov5_ros` 及其中再包含的 YOLOv5 源码。

## Recommended Next Step

如果你希望这个仓库进一步朝“成熟开源项目”靠拢，建议后续增加：

- 根目录 `LICENSE` / `NOTICE`
- 第三方组件来源链接与许可证清单
- 哪些目录是原创，哪些目录是 vendored / adapted 的更明确说明
