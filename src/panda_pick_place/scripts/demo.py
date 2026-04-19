#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import json
import os
import time
import rospy
import numpy as np
import cv2
import moveit_commander
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Vector3Stamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.srv import GetModelState, GetModelProperties
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

try:
    from gazebo_interface import GazeboCubeManager
    from gazebo_interface import GazeboLinkAttacher
    from motion_controller import ArmController, GripperActionController
except ImportError:
    rospy.logwarn("无法导入自定义控制器，请确保相关文件在搜索路径中。")


class PickPlaceDemo:
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.bridge = CvBridge()

        rospy.sleep(1.0)
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        ground_pose.pose.position.z = -0.02
        ground_pose.pose.orientation.w = 1.0
        self.scene.add_box("absolute_ground", ground_pose, size=(2.0, 2.0, 0.04))
        rospy.loginfo("🌍 已将隐形刚体地面 [absolute_ground] 强制加入 MoveIt 碰撞空间！")
        self.attached_object_name = "grasped_shelf_item"
        self.attached_object_size = None
        self.enable_attached_collision_object = rospy.get_param(
            "~enable_attached_collision_object", False
        )

        self.is_running_task = False
        self.stop_requested = False
        self.last_failure_stage = ""
        self.last_failure_reason = ""
        self.arm.move_group.set_planning_time(5.0)
        self.arm.move_group.set_goal_position_tolerance(0.01)
        self.arm.move_group.set_goal_orientation_tolerance(0.05)
        self.arm.move_group.set_goal_joint_tolerance(0.05)
        self.arm.move_group.set_max_velocity_scaling_factor(0.30)
        self.arm.move_group.set_max_acceleration_scaling_factor(0.25)

        self.basket_x = float(rospy.get_param("~basket_x", 0.393296))
        self.basket_y = float(rospy.get_param("~basket_y", -0.315779))
        self.basket_z = float(rospy.get_param("~basket_z", 0.140719))
        self.basket_size = (0.36, 0.26, 0.16)
        self.basket_wall_thickness = 0.01
        self.use_basket_slot_grid = bool(rospy.get_param("~use_basket_slot_grid", False))
        self.place_hover_height = float(rospy.get_param("~place_hover_height", 0.05))
        self.place_release_height_offset = float(rospy.get_param("~place_release_height_offset", 0.02))
        self.place_orientation_mode = str(rospy.get_param("~place_orientation_mode", "preserve")).strip().lower()
        if self.place_orientation_mode not in ("preserve", "downward"):
            rospy.logwarn(
                f"⚠️ 未知的 place_orientation_mode={self.place_orientation_mode}，回退到 preserve。"
            )
            self.place_orientation_mode = "preserve"
        self.place_allow_position_only_fallback = bool(
            rospy.get_param("~place_allow_position_only_fallback", False)
        )
        self.use_fixed_shelf_collision = bool(rospy.get_param("~use_fixed_shelf_collision", True))
        self.shelf_model_name = str(rospy.get_param("~shelf_model_name", "narrow_supermarket_shelf_enclosed_0"))
        self.shelf_pose_fallback = {
            "x": float(rospy.get_param("~shelf_pose_fallback_x", 0.737098)),
            "y": float(rospy.get_param("~shelf_pose_fallback_y", -0.148598)),
            "z": float(rospy.get_param("~shelf_pose_fallback_z", 0.205537)),
            "yaw": float(rospy.get_param("~shelf_pose_fallback_yaw", 0.0)),
        }
        self.force_shelf_normal_approach = bool(rospy.get_param("~force_shelf_normal_approach", True))
        self.execution_shelf_clearance_min = float(rospy.get_param("~execution_shelf_clearance_min", 0.025))
        self.simple_front_grasp_mode = bool(rospy.get_param("~simple_front_grasp_mode", True))
        self.enable_gazebo_attach = bool(rospy.get_param("~enable_gazebo_attach", False))
        self.gazebo_attach_distance_threshold = float(rospy.get_param("~gazebo_attach_distance_threshold", 0.18))
        self.gazebo_attach_link_name = str(rospy.get_param("~gazebo_attach_link_name", "panda_link8"))
        self.drop_counter = 0
        self.current_item_collision_size = None
        self.last_grasp_width_hint = 0.04
        self.last_grasp_depth_hint = 0.03
        self.gazebo_model_states = None
        self.attached_gazebo_model = None
        self.attached_gazebo_link = None

        self.grasp_pose_array_received = None
        self.grasp_infos = []

        self.debug_rgb = None
        self.debug_K = None
        self.show_grasp_debug = rospy.get_param("~show_grasp_debug", True)
        self.save_task_keyframes = rospy.get_param("~save_task_keyframes", False)
        self.figure_output_dir = os.path.expanduser(
            rospy.get_param("~figure_output_dir", "~/grasp_robot_ws/thesis_figures")
        )
        self.task_figure_index = 0
        # self.camera_frame = rospy.get_param("~camera_frame", "depth_camera_link")
        self.gripper_draw_half_width = rospy.get_param("~gripper_draw_half_width", 0.04)
        self.camera_frame = rospy.get_param("~camera_frame", "wrist_camera_optical_link")
        self.pre_grasp_back_distance = rospy.get_param("~pre_grasp_back_distance", 0.12)
        self.pre_grasp_back_distance_candidates = [
            float(v) for v in rospy.get_param("~pre_grasp_back_distance_candidates", [0.12, 0.15, 0.18])
        ]
        self.wrist_observation_backoff = rospy.get_param("~wrist_observation_backoff", 0.04)
        self.wrist_depth_band = rospy.get_param("~wrist_depth_band", 0.05)
        self.wrist_front_depth_percentile = rospy.get_param("~wrist_front_depth_percentile", 20.0)
        self.insert_extra_depth_min = rospy.get_param("~insert_extra_depth_min", 0.06)
        self.insert_depth_margin = rospy.get_param("~insert_depth_margin", 0.035)
        self.insert_extra_depth_max = rospy.get_param("~insert_extra_depth_max", 0.09)
        self.use_pose_preserving_transport = rospy.get_param("~use_pose_preserving_transport", False)
        self.simple_place_drop = bool(rospy.get_param("~simple_place_drop", True))
        self.refresh_collision_after_each_place = bool(
            rospy.get_param("~refresh_collision_after_each_place", False)
        )
        self.transport_step_size = float(rospy.get_param("~transport_step_size", 0.025))
        self.insert_step_size = float(rospy.get_param("~insert_step_size", 0.025))
        self.retreat_step_size = float(rospy.get_param("~retreat_step_size", 0.025))
        self.lift_step_size = float(rospy.get_param("~lift_step_size", 0.012))
        self.grasp_force = float(rospy.get_param("~grasp_force", 50.0))
        self.grasp_width_margin = float(rospy.get_param("~grasp_width_margin", 0.002))
        self.grasp_command_width_min = float(rospy.get_param("~grasp_command_width_min", 0.010))
        self.grasp_command_width_max = float(rospy.get_param("~grasp_command_width_max", 0.078))
        self.grasp_width_margin_wide = float(rospy.get_param("~grasp_width_margin_wide", 0.001))
        self.grasp_width_margin_wide_threshold = float(
            rospy.get_param("~grasp_width_margin_wide_threshold", 0.068)
        )
        self.grasp_roll_variants = [
            str(v) for v in rospy.get_param("~grasp_roll_variants", ["x_down"])
        ]
        self.use_iterative_wrist_alignment = bool(rospy.get_param("~use_iterative_wrist_alignment", False))
        self.wrist_alignment_max_iters = int(rospy.get_param("~wrist_alignment_max_iters", 1))
        self.use_wrist_guided_insert_distance = bool(rospy.get_param("~use_wrist_guided_insert_distance", False))
        self.min_insert_distance = float(rospy.get_param("~min_insert_distance", 0.02))
        self.wrist_visibility_margin_px = int(rospy.get_param("~wrist_visibility_margin_px", 8))
        self.wrist_require_full_visibility = bool(rospy.get_param("~wrist_require_full_visibility", False))
        self.wrist_min_insert_fraction = float(rospy.get_param("~wrist_min_insert_fraction", 0.45))
        self.use_final_wrist_alignment = bool(rospy.get_param("~use_final_wrist_alignment", False))
        self.wrist_alignment_allow_vertical = bool(rospy.get_param("~wrist_alignment_allow_vertical", False))
        self.wrist_extra_insert_max = float(rospy.get_param("~wrist_extra_insert_max", 0.03))
        self.wrist_alignment_tol = float(rospy.get_param("~wrist_alignment_tol", 0.008))
        self.wrist_grasp_depth_margin = float(rospy.get_param("~wrist_grasp_depth_margin", 0.005))
        self.wrist_grasp_forward_min = float(rospy.get_param("~wrist_grasp_forward_min", 0.03))
        self.wrist_grasp_forward_max = float(rospy.get_param("~wrist_grasp_forward_max", 0.09))
        self.refresh_collision_before_pick = bool(rospy.get_param("~refresh_collision_before_pick", True))
        self.grasp_corridor_padding = float(rospy.get_param("~grasp_corridor_padding", 0.03))
        self.grasp_sub = rospy.Subscriber('/graspnet/grasp_pose_array', PoseArray, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber('/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)
        self.cmd_sub = rospy.Subscriber('/demo/command', String, self.ui_command_callback)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.debug_rgb_cb, queue_size=1)
        self.info_cam_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.debug_info_cb, queue_size=1)

        self.wrist_rgb = None
        self.wrist_depth = None
        rospy.Subscriber('/wrist_camera/color/image_raw', Image, self.wrist_rgb_cb)
        rospy.Subscriber('/wrist_camera/depth/image_raw', Image, self.wrist_depth_cb)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_model_states_cb, queue_size=1)

        self.status_pub = rospy.Publisher('/demo/task_status', String, queue_size=10)
        self.failure_reason_pub = rospy.Publisher('/demo/failure_reason', String, queue_size=10)
        self.cartesian_fraction_pub = rospy.Publisher('/demo/cartesian_plan_fraction', Float32, queue_size=50)
        self.active_grasp_target_pub = rospy.Publisher('/sam_perception/active_grasp_target', String, queue_size=1, latch=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.wait_for_service('/compute_ik')
        self.compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.refresh_background = rospy.ServiceProxy('/sam_perception/refresh_background', Empty, persistent=True)
        self.get_model_state = None
        self.get_model_properties = None
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=1.0)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState, persistent=True)
        except Exception:
            rospy.logwarn("⚠️ 未连接到 /gazebo/get_model_state，将使用货架 fallback 位姿。")
        try:
            rospy.wait_for_service('/gazebo/get_model_properties', timeout=1.0)
            self.get_model_properties = rospy.ServiceProxy('/gazebo/get_model_properties', GetModelProperties, persistent=True)
        except Exception:
            rospy.logwarn("⚠️ 未连接到 /gazebo/get_model_properties，将默认使用 link 作为物体链接名。")

        # 增加触发 VLM 的发布器
        self.pub_wrist_trigger = rospy.Publisher('/wrist_vlm/trigger', String, queue_size=1)
        rospy.loginfo("=" * 50)
        rospy.loginfo(">> 机械臂后台执行器已就绪！")
        rospy.loginfo(">> 等待调度器下发任务...")
        rospy.loginfo("=" * 50)
        self.add_fixed_shelf_collision_geometry()
        self.add_basket_collision_geometry()
        self.detach_grasped_object()

    def save_task_keyframe(self, label):
        if not self.save_task_keyframes or self.debug_rgb is None:
            return
        os.makedirs(self.figure_output_dir, exist_ok=True)
        filename = f"task_{self.task_figure_index:03d}_{label}.png"
        cv2.imwrite(os.path.join(self.figure_output_dir, filename), self.debug_rgb)

    def clear_failure_reason(self):
        self.last_failure_stage = ""
        self.last_failure_reason = ""

    def set_failure_reason(self, stage, reason):
        self.last_failure_stage = str(stage)
        self.last_failure_reason = str(reason)
        payload = json.dumps(
            {"stage": self.last_failure_stage, "reason": self.last_failure_reason},
            ensure_ascii=False,
        )
        self.failure_reason_pub.publish(payload)
        rospy.logwarn(f"⚠️ 失败阶段: {self.last_failure_stage}, 原因: {self.last_failure_reason}")

    def refresh_shelf_collision_space(self):
        try:
            self.refresh_background()
            rospy.loginfo("🧹 已刷新货架背景点云与碰撞空间。")
        except Exception as e:
            rospy.logwarn(f"⚠️ 刷新货架碰撞空间失败: {e}")

    def clear_active_grasp_target(self):
        self.active_grasp_target_pub.publish("")

    def pose_to_xyz(self, pose):
        return [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ]

    def publish_active_grasp_target(
        self,
        target_pose,
        robot_z,
        back_distance,
        width_hint=0.04,
        depth_hint=0.03,
        pre_grasp_pose=None,
        observation_pose=None,
        insert_distance=None,
    ):
        target = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z],
            dtype=np.float64,
        )
        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm < 1e-6:
            return
        robot_z = robot_z / norm

        corridor_back = float(back_distance + self.wrist_observation_backoff + 0.04)
        corridor_front = float(max(0.06, insert_distance if insert_distance is not None else (depth_hint + 0.04)))
        corridor_radius = float(np.clip((width_hint * 0.5) + self.grasp_corridor_padding, 0.05, 0.10))

        corridor_segments_world = []
        if observation_pose is not None and pre_grasp_pose is not None:
            corridor_segments_world.append(
                {
                    "start": self.pose_to_xyz(observation_pose),
                    "end": self.pose_to_xyz(pre_grasp_pose),
                }
            )
            corridor_segments_world.append(
                {
                    "start": self.pose_to_xyz(pre_grasp_pose),
                    "end": (target + robot_z * corridor_front).tolist(),
                }
            )
        else:
            corridor_segments_world.append(
                {
                    "start": (target - robot_z * corridor_back).tolist(),
                    "end": (target + robot_z * corridor_front).tolist(),
                }
            )

        payload = {
            "enabled": True,
            "corridor_start_world": (target - robot_z * corridor_back).tolist(),
            "corridor_end_world": (target + robot_z * corridor_front).tolist(),
            "corridor_radius_m": corridor_radius,
            "corridor_segments_world": corridor_segments_world,
        }
        self.active_grasp_target_pub.publish(json.dumps(payload, ensure_ascii=False))
        rospy.loginfo(
            "🛣️ 已发布当前抓取走廊: segments=%d, back=%.3f m, front=%.3f m, radius=%.3f m",
            len(corridor_segments_world),
            corridor_back,
            corridor_front,
            corridor_radius,
        )

    def add_world_box(self, name, center_xyz, size_xyz):
        pose = PoseStamped()
        pose.header.frame_id = self.robot.get_planning_frame()
        pose.pose.orientation.w = 1.0
        pose.pose.position.x = center_xyz[0]
        pose.pose.position.y = center_xyz[1]
        pose.pose.position.z = center_xyz[2]
        self.scene.add_box(name, pose, size=size_xyz)

    def get_shelf_collision_specs(self):
        # 直接对齐当前 Gazebo 模型 narrow_supermarket_shelf_enclosed_0 的碰撞几何。
        return [
            ("shelf_base", (0.0, 0.0, 0.10), (0.32, 0.60, 0.20)),
            ("shelf_backboard", (0.15, 0.0, 0.40), (0.02, 0.60, 0.80)),
            ("shelf_side_left", (0.0, 0.31, 0.40), (0.32, 0.02, 0.80)),
            ("shelf_side_right", (0.0, -0.31, 0.40), (0.32, 0.02, 0.80)),
            ("shelf_board_bottom", (0.0, 0.0, 0.21), (0.30, 0.60, 0.02)),
            ("shelf_board_middle", (0.0, 0.0, 0.41), (0.30, 0.60, 0.02)),
            ("shelf_board_top", (0.0, 0.0, 0.61), (0.30, 0.60, 0.02)),
        ]

    def get_shelf_world_pose(self):
        if self.get_model_state is not None:
            try:
                resp = self.get_model_state(self.shelf_model_name, "world")
                if getattr(resp, "success", False):
                    q = resp.pose.orientation
                    yaw = float(tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2])
                    return {
                        "x": float(resp.pose.position.x),
                        "y": float(resp.pose.position.y),
                        "z": float(resp.pose.position.z),
                        "yaw": yaw,
                    }
                rospy.logwarn(
                    f"⚠️ Gazebo 未返回货架 {self.shelf_model_name} 的有效位姿，改用 fallback 位姿。"
                )
            except Exception as e:
                rospy.logwarn(f"⚠️ 查询 Gazebo 货架位姿失败，改用 fallback 位姿: {e}")
        return dict(self.shelf_pose_fallback)

    def add_fixed_shelf_collision_geometry(self):
        if not self.use_fixed_shelf_collision:
            rospy.loginfo("📚 已关闭固定货架碰撞几何。")
            return

        shelf_pose = self.get_shelf_world_pose()
        transform = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        transform[:3, 3] = np.array(
            [shelf_pose["x"], shelf_pose["y"], shelf_pose["z"]],
            dtype=np.float64,
        )

        for name, local_center, size_xyz in self.get_shelf_collision_specs():
            self.scene.remove_world_object(name)
            local = np.array([local_center[0], local_center[1], local_center[2], 1.0], dtype=np.float64)
            world = transform @ local
            self.add_world_box(name, (float(world[0]), float(world[1]), float(world[2])), size_xyz)

        rospy.loginfo(
            "📚 已将固定货架碰撞几何加入 MoveIt 场景: model=%s x=%.3f y=%.3f z=%.3f yaw=%.3f",
            self.shelf_model_name,
            shelf_pose["x"],
            shelf_pose["y"],
            shelf_pose["z"],
            shelf_pose["yaw"],
        )

    def get_shelf_inward_axis_world(self):
        shelf_pose = self.get_shelf_world_pose()
        shelf_tf = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        inward = shelf_tf[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = np.linalg.norm(inward)
        if norm < 1e-6:
            return None
        return inward / norm

    def get_shelf_inner_regions_local(self):
        return [
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.235, "zmax": 0.385},
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.435, "zmax": 0.585},
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.635, "zmax": 0.775},
        ]

    def world_point_to_shelf_local(self, point_xyz):
        shelf_pose = self.get_shelf_world_pose()
        shelf_tf = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        shelf_tf[:3, 3] = np.array([shelf_pose["x"], shelf_pose["y"], shelf_pose["z"]], dtype=np.float64)
        world_to_shelf = np.linalg.inv(shelf_tf)
        point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
        point_local = world_to_shelf @ point_h
        return point_local[:3]

    def compute_shelf_cavity_clearance(self, pose):
        point_local = self.world_point_to_shelf_local(
            [pose.position.x, pose.position.y, pose.position.z]
        )
        best_clearance = None
        for region in self.get_shelf_inner_regions_local():
            if (
                region["xmin"] <= point_local[0] <= region["xmax"]
                and region["ymin"] <= point_local[1] <= region["ymax"]
                and region["zmin"] <= point_local[2] <= region["zmax"]
            ):
                clearance = min(
                    point_local[0] - region["xmin"],
                    region["xmax"] - point_local[0],
                    point_local[1] - region["ymin"],
                    region["ymax"] - point_local[1],
                    point_local[2] - region["zmin"],
                    region["zmax"] - point_local[2],
                )
                if best_clearance is None or clearance > best_clearance:
                    best_clearance = clearance
        return None if best_clearance is None else float(best_clearance)

    def get_downward_place_orientation(self):
        # 采用一个固定的末端朝下姿态作为备用放置姿态。
        q = tft.quaternion_from_euler(np.pi, 0.0, 0.0)
        pose = Pose()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose.orientation

    def get_place_transport_orientation(self, current_pose):
        mode = self.place_orientation_mode
        if mode == "downward":
            return copy.deepcopy(self.get_downward_place_orientation())
        return copy.deepcopy(current_pose.orientation)

    def orientation_distance_deg(self, orientation_a, orientation_b):
        qa = np.array(
            [orientation_a.x, orientation_a.y, orientation_a.z, orientation_a.w],
            dtype=np.float64,
        )
        qb = np.array(
            [orientation_b.x, orientation_b.y, orientation_b.z, orientation_b.w],
            dtype=np.float64,
        )
        if np.linalg.norm(qa) < 1e-8 or np.linalg.norm(qb) < 1e-8:
            return 180.0
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(dot)))

    def orientations_match(self, orientation_a, orientation_b, tol_deg=5.0):
        return self.orientation_distance_deg(orientation_a, orientation_b) <= float(tol_deg)

    def align_current_pose_orientation(self, target_orientation, description, retries=2):
        current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose = copy.deepcopy(current_pose)
        target_pose.orientation = copy.deepcopy(target_orientation)
        return self.execute_pose_goal(target_pose, description, retries=retries)

    def add_basket_collision_geometry(self):
        bx, by, bz = self.basket_x, self.basket_y, self.basket_z
        sx, sy, sz = self.basket_size
        t = self.basket_wall_thickness

        # 底板
        self.add_world_box("basket_base", (bx, by, bz + 0.005), (sx, sy, 0.01))
        # 前后壁
        self.add_world_box("basket_wall_front", (bx, by + (sy / 2 - t / 2), bz + 0.065), (sx, t, sz))
        self.add_world_box("basket_wall_back", (bx, by - (sy / 2 - t / 2), bz + 0.065), (sx, t, sz))
        # 左右壁
        self.add_world_box("basket_wall_left", (bx + (sx / 2 - t / 2), by, bz + 0.065), (t, sy, sz))
        self.add_world_box("basket_wall_right", (bx - (sx / 2 - t / 2), by, bz + 0.065), (t, sy, sz))
        rospy.loginfo(
            f"🧺 已将篮子碰撞体加入 MoveIt 场景: x={bx:.3f}, y={by:.3f}, z={bz:.3f}"
        )

    def get_next_basket_slot(self):
        if not self.use_basket_slot_grid:
            return self.basket_x, self.basket_y

        # 3 列 x 2 行的简单摆放网格，避免后续物体都堆在同一处。
        col = self.drop_counter % 3
        row = (self.drop_counter // 3) % 2
        x_offsets = [-0.07, 0.0, 0.07]
        y_offsets = [-0.04, 0.04]
        return self.basket_x + x_offsets[col], self.basket_y + y_offsets[row]

    def ui_command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ['home', 'stop', 'h']:
            rospy.logwarn("!!! 收到干预指令：紧急刹车并回 Home !!!")
            self.stop_requested = True
            self.arm.move_group.stop()
            self.is_running_task = False
            self.set_failure_reason("user_interrupt", "stop_command")
            self.arm.go_to_home()
            self.status_pub.publish("FAILED_BY_USER")
        elif cmd == 'all_done':
            rospy.loginfo("🏁 所有任务完成，回到 Home 位。")
            self.arm.go_to_home()
        elif cmd == 'quit':
            rospy.logwarn("收到退出指令，正在关闭后台服务...")
            rospy.signal_shutdown("UI 请求退出")

    def grasp_pose_callback(self, msg):
        if not self.is_running_task:
            self.grasp_pose_array_received = msg

    def grasp_info_callback(self, msg):
        if not self.is_running_task:
            self.grasp_infos = msg.data

    def debug_rgb_cb(self, msg):
        try:
            self.debug_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def debug_info_cb(self, msg):
        try:
            self.debug_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        except Exception:
            self.debug_K = None

    def wrist_rgb_cb(self, msg):
        try:
            self.wrist_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def wrist_depth_cb(self, msg):
        try:
            self.wrist_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            pass

    def gazebo_model_states_cb(self, msg):
        self.gazebo_model_states = msg

    def transform_pose(self, input_pose, target_frame="world"):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            return tf2_geometry_msgs.do_transform_pose(input_pose, transform)
        except Exception:
            return None

    def wait_for_scene_sync(self, object_name, object_is_known, object_is_attached, timeout=2.0):
        start = rospy.get_time()
        while (rospy.get_time() - start < timeout) and not rospy.is_shutdown():
            is_attached = len(self.scene.get_attached_objects([object_name])) > 0
            is_known = object_name in self.scene.get_known_object_names()
            if is_attached == object_is_attached and is_known == object_is_known:
                return True
            rospy.sleep(0.1)
        return False

    def estimate_grasped_object_size(self, width_hint=0.04, depth_hint=0.03):
        width_hint = float(width_hint) if width_hint is not None else 0.04
        depth_hint = float(depth_hint) if depth_hint is not None else 0.03

        # 用抓取宽度和深度估计一个保守但不过分夸张的商品碰撞盒。
        box_x = float(np.clip(depth_hint + 0.035, 0.04, 0.12))
        box_y = float(np.clip(width_hint + 0.02, 0.03, 0.09))
        box_z = float(np.clip(max(box_y * 1.8, 0.08), 0.08, 0.18))
        return (box_x, box_y, box_z)

    def compute_insert_extra_depth(self, width_hint=0.04, depth_hint=0.03):
        """根据抓取宽度和深度自适应计算插入时超越目标位姿的额外距离。"""
        width_hint = float(width_hint) if width_hint is not None else 0.04
        depth_hint = float(depth_hint) if depth_hint is not None else 0.03

        # 仅在抓取深度明显偏大时再额外往里补，避免默认插入过深。
        extra_depth = self.insert_extra_depth_min + max(0.0, depth_hint - 0.02) + self.insert_depth_margin
        return float(np.clip(extra_depth, self.insert_extra_depth_min, self.insert_extra_depth_max))

    def compute_command_grasp_width(self, predicted_width):
        predicted_width = float(predicted_width) if predicted_width is not None else 0.04
        lightly_shrunk = max(0.0, predicted_width - self.grasp_width_margin)
        command_width = float(
            np.clip(
                lightly_shrunk,
                self.grasp_command_width_min,
                self.grasp_command_width_max,
            )
        )
        return command_width, lightly_shrunk

    def build_level_grasp_quaternion(self, robot_z, roll_variant="y_down"):
        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm < 1e-6:
            return None
        robot_z = robot_z / norm
        world_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        if str(roll_variant).lower() == "x_down":
            robot_x = world_down
            robot_y = np.cross(robot_z, robot_x)
            if np.linalg.norm(robot_y) < 1e-6:
                return None
            robot_y = robot_y / np.linalg.norm(robot_y)
            robot_x = np.cross(robot_y, robot_z)
        else:
            robot_y = world_down
            robot_x = np.cross(robot_y, robot_z)
            if np.linalg.norm(robot_x) < 1e-6:
                return None
            robot_x = robot_x / np.linalg.norm(robot_x)
            robot_y = np.cross(robot_z, robot_x)

        new_rot_mat = np.eye(4)
        new_rot_mat[:3, 0] = robot_x / np.linalg.norm(robot_x)
        new_rot_mat[:3, 1] = robot_y / np.linalg.norm(robot_y)
        new_rot_mat[:3, 2] = robot_z
        return tft.quaternion_from_matrix(new_rot_mat)

    def get_gripper_touch_links(self):
        ee_link = self.arm.move_group.get_end_effector_link()
        if not ee_link:
            ee_link = "panda_hand"

        touch_links = {ee_link, "panda_hand", "panda_leftfinger", "panda_rightfinger"}
        try:
            touch_links.update(self.robot.get_link_names(group="panda_hand"))
        except Exception:
            pass
        return ee_link, sorted(touch_links)

    def attach_grasped_object(self, width_hint=0.04, depth_hint=0.03):
        if not self.enable_attached_collision_object:
            self.attached_object_size = None
            rospy.loginfo("📎 已禁用 MoveIt attached collision object，跳过附着碰撞盒。")
            return True

        ee_link, touch_links = self.get_gripper_touch_links()
        box_size = self.estimate_grasped_object_size(width_hint, depth_hint)

        self.scene.remove_attached_object(ee_link, name=self.attached_object_name)
        self.scene.remove_world_object(self.attached_object_name)
        rospy.sleep(0.2)

        box_pose = PoseStamped()
        box_pose.header.frame_id = ee_link
        box_pose.pose.orientation.w = 1.0
        # 将碰撞盒中心放在手爪前方，尽量贴近真实被夹持商品的位置。
        box_pose.pose.position.x = box_size[0] * 0.5 + 0.015
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.0

        self.scene.attach_box(
            ee_link,
            self.attached_object_name,
            box_pose,
            size=box_size,
            touch_links=touch_links
        )

        if self.wait_for_scene_sync(self.attached_object_name, object_is_known=False, object_is_attached=True):
            self.attached_object_size = box_size
            rospy.loginfo(
                f"📎 已将抓取物体附着到规划场景: link={ee_link}, size={tuple(round(v, 3) for v in box_size)}"
            )
            return True

        rospy.logwarn("⚠️ 抓取物体附着到规划场景失败，后续放置规划可能仍受影响。")
        self.attached_object_size = None
        return False

    def detach_grasped_object(self):
        if not self.enable_attached_collision_object:
            self.attached_object_size = None
            return True

        ee_link, _ = self.get_gripper_touch_links()
        attached_before = len(self.scene.get_attached_objects([self.attached_object_name])) > 0
        known_before = self.attached_object_name in self.scene.get_known_object_names()
        if not attached_before and not known_before:
            self.attached_object_size = None
            return True

        self.scene.remove_attached_object(ee_link, name=self.attached_object_name)
        self.scene.remove_world_object(self.attached_object_name)
        self.wait_for_scene_sync(self.attached_object_name, object_is_known=False, object_is_attached=False, timeout=1.0)
        self.attached_object_size = None
        return True

    def is_graspable_gazebo_model(self, name):
        ignored_prefixes = (
            "ground_plane",
            "depth_camera_model",
            "panda",
            "grasp_basket",
            "narrow_supermarket_shelf_enclosed",
        )
        return isinstance(name, str) and not name.startswith(ignored_prefixes)

    def get_gazebo_model_link_name(self, model_name):
        if self.get_model_properties is not None:
            try:
                resp = self.get_model_properties(model_name)
                body_names = list(getattr(resp, "body_names", []))
                if body_names:
                    return body_names[0]
            except Exception as e:
                rospy.logwarn(f"⚠️ 获取 Gazebo 模型 {model_name} 的 link 名失败，回退到默认 link: {e}")
        return "link"

    def find_nearest_gazebo_object(self, target_pose):
        if self.gazebo_model_states is None:
            return None, None

        target = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z], dtype=np.float64)
        ee_pose = self.arm.move_group.get_current_pose().pose
        ee_pos = np.array([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z], dtype=np.float64)
        best_name = None
        best_score = None
        for name, pose in zip(self.gazebo_model_states.name, self.gazebo_model_states.pose):
            if not self.is_graspable_gazebo_model(name):
                continue
            pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
            dist_target = float(np.linalg.norm(pos - target))
            dist_ee = float(np.linalg.norm(pos - ee_pos))
            score = dist_target + 0.35 * dist_ee
            if best_score is None or score < best_score:
                best_name = name
                best_score = score

        if best_name is None or best_score is None or best_score > self.gazebo_attach_distance_threshold:
            return None, None
        return best_name, self.get_gazebo_model_link_name(best_name)

    def attach_gazebo_object_if_possible(self, target_pose):
        if not self.enable_gazebo_attach:
            return False
        model_name, link_name = self.find_nearest_gazebo_object(target_pose)
        if model_name is None:
            rospy.logwarn("⚠️ 未找到足够接近当前目标位姿的 Gazebo 物体，跳过物理附着。")
            return False
        try:
            attacher = GazeboLinkAttacher(
                model_name_1="panda",
                link_name_1=self.gazebo_attach_link_name,
                model_name_2=model_name,
                link_name_2=link_name,
            )
            if attacher.attach():
                self.attached_gazebo_model = model_name
                self.attached_gazebo_link = link_name
                rospy.loginfo(f"🪝 已在 Gazebo 中附着物体: model={model_name}, link={link_name}")
                return True
        except Exception as e:
            rospy.logwarn(f"⚠️ Gazebo attach 调用失败: {e}")
        return False

    def detach_gazebo_object_if_needed(self):
        if not self.enable_gazebo_attach or self.attached_gazebo_model is None:
            return True
        try:
            attacher = GazeboLinkAttacher(
                model_name_1="panda",
                link_name_1=self.gazebo_attach_link_name,
                model_name_2=self.attached_gazebo_model,
                link_name_2=self.attached_gazebo_link or "link",
            )
            ok = attacher.detach()
            if ok:
                rospy.loginfo(f"🪝 已在 Gazebo 中释放物体: model={self.attached_gazebo_model}")
            else:
                rospy.logwarn(f"⚠️ Gazebo detach 返回失败: model={self.attached_gazebo_model}")
            self.attached_gazebo_model = None
            self.attached_gazebo_link = None
            return ok
        except Exception as e:
            rospy.logwarn(f"⚠️ Gazebo detach 调用失败: {e}")
            self.attached_gazebo_model = None
            self.attached_gazebo_link = None
            return False

    def execute_pose_goal(self, target_pose, description, retries=2):
        for attempt in range(1, retries + 1):
            if self.stop_requested:
                return False

            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.clear_pose_targets()
            self.arm.move_group.set_pose_target(target_pose)
            rospy.loginfo(f"{description} (第 {attempt}/{retries} 次尝试)")

            success = self.arm.move_group.go(wait=True)
            self.arm.move_group.stop()
            self.arm.move_group.clear_pose_targets()

            if success:
                return True

            rospy.logwarn(f"⚠️ {description} 失败，准备重试...")
            rospy.sleep(0.4)
        return False

    def execute_position_only_goal(self, x, y, z, description, retries=2):
        """仅约束位置，不约束姿态，让规划器自由选择末端朝向，适用于放置搬运阶段。"""
        for attempt in range(1, retries + 1):
            if self.stop_requested:
                return False
            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.clear_pose_targets()
            self.arm.move_group.set_position_target([x, y, z])
            rospy.loginfo(f"{description} [仅位置] (第 {attempt}/{retries} 次尝试) x={x:.3f} y={y:.3f} z={z:.3f}")
            success = self.arm.move_group.go(wait=True)
            self.arm.move_group.stop()
            self.arm.move_group.clear_pose_targets()
            if success:
                return True
            current_pose = self.arm.move_group.get_current_pose().pose
            pos_err = float(
                np.linalg.norm(
                    np.array(
                        [
                            current_pose.position.x - x,
                            current_pose.position.y - y,
                            current_pose.position.z - z,
                        ],
                        dtype=np.float64,
                    )
                )
            )
            if pos_err <= 0.02:
                rospy.logwarn(
                    f"⚠️ {description} [仅位置] 返回失败，但位置误差仅 {pos_err:.4f} m，按成功处理。"
                )
                return True
            rospy.logwarn(f"⚠️ {description} [仅位置] 失败，准备重试...")
            rospy.sleep(0.4)
        return False

    def move_to_current_xy_z_position_only(self, target_z, description):
        current_pose = self.arm.move_group.get_current_pose().pose
        return self.execute_position_only_goal(
            current_pose.position.x,
            current_pose.position.y,
            target_z,
            description,
            retries=2,
        )

    def execute_transport_goal(self, x, y, z, description, retries=2):
        current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose = copy.deepcopy(current_pose)
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        if self.use_pose_preserving_transport:
            if self.execute_pose_goal(target_pose, f"{description} [保姿态优先]", retries=retries):
                return True
            rospy.logwarn(f"⚠️ {description} 保姿态规划失败，回退到仅位置目标。")

        return self.execute_position_only_goal(x, y, z, description, retries=retries)

    def move_to_current_xy_z_transport_goal(self, target_z, description):
        current_pose = self.arm.move_group.get_current_pose().pose
        return self.execute_transport_goal(
            current_pose.position.x,
            current_pose.position.y,
            target_z,
            description,
            retries=2,
        )

    def move_horizontal_to_xy(
        self,
        target_x,
        target_y,
        description,
        orientation=None,
        allow_position_only_fallback=True,
    ):
        current_pose = self.arm.move_group.get_current_pose().pose
        dx = float(target_x - current_pose.position.x)
        dy = float(target_y - current_pose.position.y)
        horizontal_dist = float(np.linalg.norm([dx, dy]))
        if horizontal_dist < 0.003:
            rospy.loginfo(f"{description}：当前水平位置已满足，无需额外平移。")
            return True

        target_orientation = copy.deepcopy(orientation) if orientation is not None else copy.deepcopy(current_pose.orientation)
        orientation_locked = orientation is not None

        rospy.loginfo(
            f"{description}：水平平移 {horizontal_dist:.3f} m "
            f"(dx={dx:.3f}, dy={dy:.3f})"
        )
        if not orientation_locked or self.orientations_match(current_pose.orientation, target_orientation, tol_deg=2.0):
            if self.segmented_cartesian_move(
                dx=dx,
                dy=dy,
                dz=0.0,
                description=description,
                step_size=self.transport_step_size,
                min_step=0.01,
                avoid_collisions=True,
            ):
                return True
        else:
            rospy.loginfo(f"{description}：当前末端尚未对齐固定放置姿态，跳过笛卡尔平移，直接尝试位姿规划。")

        if orientation_locked:
            rospy.logwarn(f"⚠️ {description} 直线平移失败，改为保持固定放置姿态的位姿规划。")
        else:
            rospy.logwarn(f"⚠️ {description} 直线平移失败，改为保持当前姿态的位姿规划。")
        target_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.orientation = target_orientation
        pose_label = "固定放置姿态" if orientation_locked else "保当前姿态"
        if self.execute_pose_goal(target_pose, f"{description} [{pose_label}]", retries=2):
            return True

        if not allow_position_only_fallback:
            rospy.logwarn(f"🛑 {description} 固定姿态规划失败，且已禁止回退到仅位置目标。")
            return False

        rospy.logwarn(f"⚠️ {description} 保持姿态的位姿规划也失败，最后才回退到仅位置目标。")
        current_pose = self.arm.move_group.get_current_pose().pose
        return self.execute_position_only_goal(
            target_x,
            target_y,
            current_pose.position.z,
            description,
            retries=2,
        )

    def move_vertical_to_z(
        self,
        target_z,
        description,
        orientation=None,
        allow_position_only_fallback=True,
    ):
        current_pose = self.arm.move_group.get_current_pose().pose
        dz = target_z - current_pose.position.z
        if abs(dz) < 0.003:
            rospy.loginfo(f"{description}：当前高度已满足，无需额外竖直移动。")
            return True

        direction = "抬升" if dz > 0 else "下降"
        rospy.loginfo(f"{description}：{direction} {abs(dz):.3f} m")
        target_orientation = copy.deepcopy(orientation) if orientation is not None else copy.deepcopy(current_pose.orientation)
        orientation_locked = orientation is not None
        if not orientation_locked or self.orientations_match(current_pose.orientation, target_orientation, tol_deg=2.0):
            if self.segmented_cartesian_move(
                dx=0.0,
                dy=0.0,
                dz=dz,
                description=description,
                step_size=self.lift_step_size,
                min_step=0.005,
                avoid_collisions=True,
            ):
                return True
        else:
            rospy.loginfo(f"{description}：当前末端尚未对齐固定放置姿态，跳过笛卡尔升降，直接尝试位姿规划。")

        target_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose.position.z = target_z
        target_pose.orientation = target_orientation
        pose_label = "固定放置姿态" if orientation_locked else "保当前姿态"
        if self.execute_pose_goal(target_pose, f"{description} [{pose_label}]", retries=2):
            return True

        if not allow_position_only_fallback:
            rospy.logwarn(f"🛑 {description} 固定姿态规划失败，且已禁止回退到仅位置目标。")
            return False

        rospy.logwarn(f"⚠️ {description} 笛卡尔抬升/下降失败，回退到仅位置约束规划。")
        current_pose = self.arm.move_group.get_current_pose().pose
        return self.execute_position_only_goal(
            current_pose.position.x,
            current_pose.position.y,
            target_z,
            description,
            retries=2,
        )

    def get_ik_seeded(self, pose, seed_q, frame_id):
        """以 GTSP 预计算的关节角为种子求 IK，保证解与调度规划处于同一构型。"""
        req = PositionIKRequest()
        req.group_name = "panda_manipulator"
        robot_state = self.robot.get_current_state()
        js = JointState()
        js.name = self.arm.move_group.get_active_joints()
        js.position = list(seed_q)
        robot_state.joint_state = js
        req.robot_state = robot_state
        req.pose_stamped = PoseStamped()
        req.pose_stamped.header.frame_id = frame_id
        req.pose_stamped.pose = pose
        req.timeout = rospy.Duration(0.2)
        req.avoid_collisions = False
        try:
            res = self.compute_ik(req)
            if res.error_code.val == 1:
                return list(res.solution.joint_state.position[:7])
        except Exception as e:
            rospy.logwarn(f"seeded IK 失败，回退到 set_pose_target: {e}")
        return None

    def plan_cartesian_relative(self, dx=0.0, dy=0.0, dz=0.0,
                                eef_step=0.01, avoid_collisions=False):
        """只试算笛卡尔路径，不执行。"""
        start_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose = copy.deepcopy(start_pose)
        target_pose.position.x += dx
        target_pose.position.y += dy
        target_pose.position.z += dz

        plan, fraction = self.arm.move_group.compute_cartesian_path(
            [target_pose],
            eef_step,
            avoid_collisions
        )
        self.cartesian_fraction_pub.publish(Float32(data=float(fraction)))
        return plan, fraction

    def plan_cartesian_waypoints(self, waypoints, eef_step=0.01, avoid_collisions=False):
        """对一组 waypoints 做一次笛卡尔规划。"""
        if not waypoints:
            return None, 0.0
        plan, fraction = self.arm.move_group.compute_cartesian_path(
            waypoints,
            eef_step,
            avoid_collisions
        )
        self.cartesian_fraction_pub.publish(Float32(data=float(fraction)))
        return plan, fraction

    def execute_cartesian_plan(self, plan, jump_threshold=2.5):
        if plan is None or len(plan.joint_trajectory.points) == 0:
            return False
        points = plan.joint_trajectory.points
        for i in range(1, len(points)):
            prev = np.array(points[i-1].positions)
            curr = np.array(points[i].positions)
            max_jump = float(np.max(np.abs(curr - prev)))
            if max_jump > jump_threshold:
                rospy.logwarn(f"🛑 拒绝执行：关节跳变检测到第{i}点跳变 {max_jump:.3f} rad > 阈值 {jump_threshold} rad")
                return False
        success = self.arm.move_group.execute(plan, wait=True)
        self.arm.move_group.stop()
        if success:
            return True

        current_joints = np.array(self.arm.move_group.get_current_joint_values()[:len(points[-1].positions)], dtype=np.float64)
        target_joints = np.array(points[-1].positions, dtype=np.float64)
        max_joint_error = float(np.max(np.abs(current_joints - target_joints)))
        if max_joint_error <= 0.02:
            rospy.logwarn(
                f"⚠️ 笛卡尔执行返回失败，但末端关节已足够接近目标 (max_joint_error={max_joint_error:.4f} rad)，按成功处理。"
            )
            return True
        return False

    def build_linear_waypoints(self, dx, dy, dz, step_size):
        """从当前位置沿直线构建 waypoints 和对应累计位移。"""
        move_vec = np.array([dx, dy, dz], dtype=np.float64)
        total_dist = float(np.linalg.norm(move_vec))
        if total_dist < 1e-6:
            return [], []

        start_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        direction = move_vec / total_dist
        waypoints = []
        distances = []

        dist = min(step_size, total_dist)
        while dist < total_dist - 1e-9:
            pose = copy.deepcopy(start_pose)
            pose.position.x += float(direction[0] * dist)
            pose.position.y += float(direction[1] * dist)
            pose.position.z += float(direction[2] * dist)
            waypoints.append(pose)
            distances.append(dist)
            dist += step_size

        final_pose = copy.deepcopy(start_pose)
        final_pose.position.x += float(move_vec[0])
        final_pose.position.y += float(move_vec[1])
        final_pose.position.z += float(move_vec[2])
        waypoints.append(final_pose)
        distances.append(total_dist)
        return waypoints, distances

    def apply_safe_world_correction(self, dx, dy, dz):
        """腕部相机微调先试算，再按较安全的缩放比例执行。"""
        correction = np.array([dx, dy, dz], dtype=np.float64)
        if np.linalg.norm(correction) < 0.002:
            rospy.loginfo("🎯 目标居中完美，无需微调！")
            return True

        for scale in [1.0, 0.7, 0.4]:
            scaled = correction * scale
            plan, fraction = self.plan_cartesian_relative(
                dx=float(scaled[0]),
                dy=float(scaled[1]),
                dz=float(scaled[2]),
                avoid_collisions=False
            )
            if fraction >= 0.95:
                rospy.loginfo(
                    f"🔧 TF 刚体平移修正: dx={scaled[0]:.4f}, dy={scaled[1]:.4f}, "
                    f"dz={scaled[2]:.4f}, scale={scale:.1f}"
                )
                return self.execute_cartesian_plan(plan)

            rospy.logwarn(
                f"⚠️ 腕部修正 scale={scale:.1f} 只规划了 {fraction*100:.1f}% ，尝试缩小修正量。"
            )

        rospy.logwarn("⚠️ 腕部微调始终不可安全执行，保留当前预抓取位继续尝试。")
        return True

    def segmented_cartesian_move(self, dx, dy, dz, description,
                                 step_size=0.04, min_step=0.01,
                                 avoid_collisions=False):
        """将长距离直线运动拆成多段，但尽量把多段合成一条轨迹一次执行。"""
        move_vec = np.array([dx, dy, dz], dtype=np.float64)
        total_dist = float(np.linalg.norm(move_vec))
        if total_dist < 1e-6:
            return True

        direction = move_vec / total_dist
        moved = 0.0
        execution_idx = 0
        current_step = step_size

        while moved < total_dist - 1e-6:
            remaining = total_dist - moved
            remaining_vec = direction * remaining
            waypoints, distances = self.build_linear_waypoints(
                dx=float(remaining_vec[0]),
                dy=float(remaining_vec[1]),
                dz=float(remaining_vec[2]),
                step_size=current_step
            )

            if not waypoints:
                return True

            full_plan, full_fraction = self.plan_cartesian_waypoints(
                waypoints,
                avoid_collisions=avoid_collisions
            )
            if full_fraction >= 0.95:
                execution_idx += 1
                rospy.loginfo(
                    f"{description}：一次性执行剩余 {len(waypoints)} 段 "
                    f"(累计 {total_dist:.3f}/{total_dist:.3f} m)"
                )
                return self.execute_cartesian_plan(full_plan)

            # 用二分法找当前步长下“最长可连续执行前缀”
            lo, hi = 1, len(waypoints)
            best_n = 0
            best_plan = None
            while lo <= hi:
                mid = (lo + hi) // 2
                plan_mid, fraction_mid = self.plan_cartesian_waypoints(
                    waypoints[:mid],
                    avoid_collisions=avoid_collisions
                )
                if fraction_mid >= 0.95 and plan_mid is not None and len(plan_mid.joint_trajectory.points) > 0:
                    best_n = mid
                    best_plan = plan_mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_n > 0:
                execution_idx += 1
                chunk_dist = distances[best_n - 1]
                rospy.loginfo(
                    f"{description}：连续执行 {best_n} 段 "
                    f"(本次 {chunk_dist:.3f} m，累计 {moved + chunk_dist:.3f}/{total_dist:.3f} m)"
                )
                if not self.execute_cartesian_plan(best_plan):
                    return False
                moved += chunk_dist
                continue

            if current_step > min_step + 1e-9:
                next_step = max(current_step * 0.5, min_step)
                rospy.logwarn(
                    f"⚠️ {description} 当前步长 {current_step:.3f} m 无法形成有效连续轨迹，"
                    f"缩小到 {next_step:.3f} m 重试。"
                )
                current_step = next_step
                continue

            rospy.logwarn(f"🛑 {description} 失败：步长缩小后仍无法安全规划。")
            return False

        return True

    # ===================================================================================
    # 🧠 方案 B: VLM 语义微调 (纯相机局部坐标系版)
    # ===================================================================================
    def request_wrist_bbox(self):
        self.pub_wrist_trigger.publish("trigger")
        rospy.loginfo("等待 VLM 节点返回边框数据...")

        try:
            msg = rospy.wait_for_message('/wrist_vlm/bbox', Float32MultiArray, timeout=8.0)
            if len(msg.data) != 4:
                return None
            return list(map(int, msg.data))
        except rospy.ROSException:
            rospy.logwarn("VLM 节点超时未回复，放弃微调！")
            return None

    def expand_bbox(self, bbox, image_w, image_h, margin_px):
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, int(xmin) - margin_px)
        ymin = max(0, int(ymin) - margin_px)
        xmax = min(image_w, int(xmax) + margin_px)
        ymax = min(image_h, int(ymax) + margin_px)
        return [xmin, ymin, xmax, ymax]

    def wrist_bbox_well_visible(self, bbox):
        if bbox is None or self.wrist_rgb is None:
            return False
        h, w = self.wrist_rgb.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        m = int(self.wrist_visibility_margin_px)
        return xmin > m and ymin > m and xmax < (w - m) and ymax < (h - m)

    def calculate_world_correction(self, return_details=False, bbox_override=None):
        """
        全自动天地融合版：VLM 给范围 + 深度图抠边缘
        返回世界坐标系下的横向修正；当 return_details=True 时，同时返回手腕系前向距离。
        """
        def invalid_result():
            details = {
                "valid": False,
                "world_dx": 0.0,
                "world_dy": 0.0,
                "world_dz": 0.0,
                "hand_forward": None,
                "hand_lateral_x": None,
                "hand_lateral_y": None,
                "bbox": None,
                "target_pixel": None,
            }
            return details if return_details else (0.0, 0.0, 0.0)

        if self.wrist_depth is None or self.wrist_rgb is None:
            return invalid_result()

        h, w = self.wrist_depth.shape
        cx, cy = w // 2, h // 2  

        if bbox_override is None:
            bbox = self.request_wrist_bbox()
            if bbox is None:
                return invalid_result()
        else:
            bbox = list(map(int, bbox_override))

        xmin, ymin, xmax, ymax = self.expand_bbox(bbox, w, h, 0)

        # 2. X轴(左右)：通过深度图抠出物理边缘中点
        roi_depth = self.wrist_depth[ymin:ymax, xmin:xmax]
        valid_mask = (roi_depth > 0.05) & (roi_depth < 0.4)
        if not np.any(valid_mask):
            return invalid_result()
        
        front_depth = float(np.percentile(roi_depth[valid_mask], self.wrist_front_depth_percentile))
        target_mask = (roi_depth >= front_depth) & (roi_depth < front_depth + self.wrist_depth_band)
        
        local_y_indices, local_x_indices = np.where(target_mask)
        if len(local_x_indices) == 0:
            return invalid_result()
        
        global_x_indices = local_x_indices + xmin
        global_y_indices = local_y_indices + ymin
        target_x = int((np.min(global_x_indices) + np.max(global_x_indices)) / 2.0)
        target_y = int((np.min(global_y_indices) + np.max(global_y_indices)) / 2.0)
        tight_bbox = [
            int(np.min(global_x_indices)),
            int(np.min(global_y_indices)),
            int(np.max(global_x_indices)),
            int(np.max(global_y_indices)),
        ]

        # --- 渲染调试画面 ---
        img_cv = self.wrist_rgb.copy()
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2) 
        cv2.circle(img_cv, (cx, cy), 5, (255, 0, 0), -1)       
        cv2.circle(img_cv, (target_x, target_y), 5, (0, 255, 0), -1) 
        cv2.line(img_cv, (cx, cy), (target_x, target_y), (0, 255, 255), 2) 
        cv2.imshow("Auto VLM Eye-in-Hand", img_cv)
        cv2.waitKey(1)

        # =========================================================================
        # 🚀 核心工业级刚体逆解：绝对杜绝向量方向错误和视差补偿错误！
        # =========================================================================
        try:
            # 步骤A：构建物体在【相机坐标系】下的绝对 3D 点 (Pose)
            # 因为是点(Pose)而不是向量，所以在转换时物理高度差(0.04m)会被TF树完美计算在内！
            p_cam = PoseStamped()
            p_cam.header.frame_id = self.camera_frame
            p_cam.header.stamp = rospy.Time(0)
            fx, fy = 500.0, 500.0
            
            p_cam.pose.position.x = (target_x - cx) * front_depth / fx
            p_cam.pose.position.y = (target_y - cy) * front_depth / fy
            p_cam.pose.position.z = front_depth
            p_cam.pose.orientation.w = 1.0

            # 抓取夹爪的当前坐标系名称
            ee_link = self.arm.move_group.get_end_effector_link()
            if not ee_link: ee_link = "panda_hand"

            # 步骤B：问 TF 树，这个 3D 物体点在【夹爪坐标系】里在哪儿？
            trans_cam_to_hand = self.tf_buffer.lookup_transform(ee_link, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            p_hand = tf2_geometry_msgs.do_transform_pose(p_cam, trans_cam_to_hand)

            # 步骤C：得出夹爪需要移动的局部距离
            # 如果物体完美对准，它在夹爪坐标系下的 x 和 y 必定是 0。
            # 所以 p_hand 的 x 和 y，正是夹爪自身需要去消除的位移！
            vec_hand = Vector3Stamped()
            vec_hand.header.frame_id = ee_link
            vec_hand.header.stamp = rospy.Time(0)
            vec_hand.vector.x = p_hand.pose.position.x
            vec_hand.vector.y = p_hand.pose.position.y
            vec_hand.vector.z = 0.0 # 绝对不改变深度，深度交给 GraspNet 的初始规划

            # 步骤D：把夹爪的局部平移，转化为【世界坐标系】的位移
            trans_hand_to_world = self.tf_buffer.lookup_transform("world", ee_link, rospy.Time(0), rospy.Duration(1.0))
            vec_world = tf2_geometry_msgs.do_transform_vector3(vec_hand, trans_hand_to_world)

            details = {
                "valid": True,
                "world_dx": vec_world.vector.x,
                "world_dy": vec_world.vector.y,
                "world_dz": vec_world.vector.z,
                "hand_forward": float(p_hand.pose.position.z),
                "hand_lateral_x": float(p_hand.pose.position.x),
                "hand_lateral_y": float(p_hand.pose.position.y),
                "bbox": [xmin, ymin, xmax, ymax],
                "tight_bbox": tight_bbox,
                "front_depth": front_depth,
                "target_pixel": [target_x, target_y],
            }
            if return_details:
                return details
            return details["world_dx"], details["world_dy"], details["world_dz"]

        except Exception as e:
            rospy.logerr(f"TF 逆解空间映射失败: {e}")
            return invalid_result()
    # ===================================================================================

    def compute_wrist_grasp_forward_target(self, depth_hint=0.03):
        # panda_hand 原点到手指尖约 0.1034m（来自 Panda URDF）
        # desired_forward = 手指尖到物体前表面的余量，即手指刚好插入 depth_hint 时对应的 hand_forward
        PANDA_HAND_TO_FINGERTIP = 0.1034
        depth_hint = float(depth_hint) if depth_hint is not None else 0.03
        target = PANDA_HAND_TO_FINGERTIP - depth_hint + self.wrist_grasp_depth_margin
        return float(np.clip(target, self.wrist_grasp_forward_min, self.wrist_grasp_forward_max))

    def compute_insert_distance_from_wrist(self, chosen_depth, fallback_distance):
        fallback_distance = float(max(0.0, fallback_distance))
        if not self.use_wrist_guided_insert_distance:
            return fallback_distance

        measurement = self.calculate_world_correction(return_details=True)
        if not measurement["valid"] or measurement["hand_forward"] is None:
            rospy.logwarn("⚠️ 手腕相机未返回有效前向距离，回退到模型估计插入距离。")
            return fallback_distance

        desired_forward = self.compute_wrist_grasp_forward_target(depth_hint=chosen_depth)
        measured_forward = float(measurement["hand_forward"])
        insert_distance_from_measurement = max(0.0, measured_forward - desired_forward)
        # 手腕测距只允许在几何 fallback 基础上“增加”插入深度，
        # 不能把本来应该走到的包裹距离缩短，否则会出现未包裹住目标就闭爪。
        lower_bound = max(
            self.min_insert_distance,
            fallback_distance,
        )
        upper_bound = fallback_distance + self.wrist_extra_insert_max
        insert_distance = float(
            np.clip(
                max(fallback_distance, insert_distance_from_measurement),
                lower_bound,
                upper_bound,
            )
        )
        rospy.loginfo(
            f"[5/6] 手腕测距修正插入: measured_forward={measured_forward:.3f} m, "
            f"desired_forward={desired_forward:.3f} m, fallback={fallback_distance:.3f} m, "
            f"final_insert={insert_distance:.3f} m"
        )
        return insert_distance

    def compute_remaining_insert_distance(self, target_pose, robot_z, extra_depth):
        current_pose = self.arm.move_group.get_current_pose().pose
        delta = np.array(
            [
                target_pose.position.x - current_pose.position.x,
                target_pose.position.y - current_pose.position.y,
                target_pose.position.z - current_pose.position.z,
            ],
            dtype=np.float64,
        )
        remaining_to_target = max(0.0, float(np.dot(delta, np.array(robot_z, dtype=np.float64))))
        return remaining_to_target + float(extra_depth)

    def run_wrist_alignment_loop(self, stage_label, max_iters=2, require_visible=True):
        """
        仅使用手腕相机做横向校正和可见性确认，不使用前向距离决定闭爪时机。
        """
        max_iters = max(1, int(max_iters))
        had_usable_visibility = False
        for idx in range(max_iters):
            measurement = self.calculate_world_correction(return_details=True)
            if not measurement["valid"]:
                if had_usable_visibility:
                    rospy.logwarn(
                        f"{stage_label} 后续复核未返回有效测量，保留上一轮有效对齐结果继续执行。"
                    )
                    return True
                rospy.logwarn(f"{stage_label} 手腕相机未返回有效测量。")
                return False

            bbox = measurement["bbox"]
            visible_ok = self.wrist_bbox_well_visible(bbox)
            correction = np.array(
                [measurement["world_dx"], measurement["world_dy"], measurement["world_dz"]],
                dtype=np.float64,
            )
            if not self.wrist_alignment_allow_vertical:
                correction[2] = 0.0
            correction_norm = float(np.linalg.norm(correction))

            if visible_ok:
                had_usable_visibility = True
            elif require_visible:
                if had_usable_visibility:
                    rospy.logwarn(
                        f"{stage_label} 后续复核时目标未完整露出，当前 bbox={bbox}，"
                        "保留上一轮有效对齐结果继续执行。"
                    )
                    return True
                rospy.logwarn(
                    f"{stage_label} 目标在手腕视野中未完整露出，当前 bbox={bbox}。"
                )
                return False

            if correction_norm <= self.wrist_alignment_tol:
                rospy.loginfo(
                    f"{stage_label} 已满足对齐阈值，correction_norm={correction_norm:.4f} m"
                )
                return True

            if not self.apply_safe_world_correction(
                float(correction[0]),
                float(correction[1]),
                float(correction[2]),
            ):
                rospy.logwarn(f"{stage_label} 手腕横向校正失败。")
                return False
            rospy.sleep(0.25)

        rospy.logwarn(f"{stage_label} 达到最大校正次数后退出。")
        return True

    def wrist_guided_insert(self, robot_z, chosen_depth):
        """
        首次用手腕相机测量物体位置，动态计算需要插入的距离，然后一次性平滑执行。
        避免多步走走停停导致的抖动问题。
        """
        # 步骤1：请求 VLM 边框并取第一次测量
        bbox = self.request_wrist_bbox()
        if bbox is None:
            rospy.logwarn("⚠️ 初始手腕 VLM 检测失败，回退到模型估计插入。")
            return False, 0.0

        measurement = self.calculate_world_correction(return_details=True, bbox_override=bbox)
        if not measurement["valid"] or measurement["hand_forward"] is None:
            rospy.logwarn("⚠️ 手腕测距首次测量失败，回退到模型估计插入。")
            return False, 0.0

        first_forward = float(measurement["hand_forward"])

        # 步骤2：动态计算插入距离
        # 进入此函数时，手臂距 target_pose = pre_grasp_back_distance - wrist_observation_backoff
        # hand_forward 随手臂前进 1:1 减小，所以直接用首次测量值推算出需要走多远
        total_to_move = self.pre_grasp_back_distance - self.wrist_observation_backoff
        desired_forward = first_forward - total_to_move

        rospy.loginfo(
            f"[5/6] 手腕测距引导插入: first_forward={first_forward:.3f} m, "
            f"total_to_move={total_to_move:.3f} m, desired_forward={desired_forward:.3f} m"
        )

        if first_forward <= desired_forward + 1e-3:
            rospy.loginfo("🎯 目标已在抓取阈值内，直接闭爪。")
            return True, 0.0

        # 步骤3：一次性平滑执行全程插入
        if not self.segmented_cartesian_move(
            dx=robot_z[0] * total_to_move,
            dy=robot_z[1] * total_to_move,
            dz=robot_z[2] * total_to_move,
            description="[5/6] 手腕引导平滑插入",
            step_size=self.insert_step_size,
            min_step=0.01,
            avoid_collisions=False
        ):
            return False, 0.0

        return True, total_to_move

        rospy.logwarn("⚠️ 手腕测距插入达到最大深度仍未满足抓取阈值。")
        return False, inserted

    def pick_object(self):
        rospy.loginfo("[1/6] 正在等待 GTSP 下发的抓取姿态...")
        self.current_item_collision_size = None
        wait_count = 0
        while self.grasp_pose_array_received is None:
            rospy.sleep(0.5)
            wait_count += 1
            if wait_count > 60:
                rospy.logerr("等待超时！未收到候选抓取姿态。")
                self.set_failure_reason("observation_unreachable", "no_grasp_pose_received")
                return False

        planning_frame = self.robot.get_planning_frame()
        best_plan, best_pre_grasp_pose, best_target_pose, best_observation_pose = None, None, None, None
        chosen_width = 0.04
        chosen_depth = 0.03
        best_robot_z = None
        selected_back_distance = self.pre_grasp_back_distance

        self.clear_active_grasp_target()
        if self.refresh_collision_before_pick:
            self.refresh_shelf_collision_space()

        rospy.loginfo("🔍 开始强制水平修正与避障评估...")
        shelf_inward_axis = self.get_shelf_inward_axis_world() if self.force_shelf_normal_approach else None
        if self.force_shelf_normal_approach and shelf_inward_axis is not None:
            rospy.loginfo(
                "🧭 当前抓取将强制沿货架法向插入: axis=(%.3f, %.3f, %.3f)",
                shelf_inward_axis[0],
                shelf_inward_axis[1],
                shelf_inward_axis[2],
            )

        for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
            ps = PoseStamped()
            ps.header = self.grasp_pose_array_received.header
            ps.pose = raw_pose

            target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
            if target_pose_stamped is None:
                continue

            q_orig = [
                target_pose_stamped.pose.orientation.x, target_pose_stamped.pose.orientation.y,
                target_pose_stamped.pose.orientation.z, target_pose_stamped.pose.orientation.w
            ]
            mat_orig = tft.quaternion_matrix(q_orig)
            raw_grasp_x = mat_orig[:3, 0]

            flat_approach = np.array([raw_grasp_x[0], raw_grasp_x[1], 0.0])
            norm = np.linalg.norm(flat_approach)

            if norm < 0.001:
                continue

            robot_z = flat_approach / norm
            if shelf_inward_axis is not None:
                robot_z = shelf_inward_axis.copy()

            if target_pose_stamped.pose.position.z < 0.08:
                target_pose_stamped.pose.position.z = 0.08

            seed_q = list(self.grasp_infos[3:10]) if len(self.grasp_infos) >= 10 else None

            success = False
            for roll_variant in self.grasp_roll_variants:
                q_new = self.build_level_grasp_quaternion(robot_z, roll_variant=roll_variant)
                if q_new is None:
                    continue

                candidate_target_pose = copy.deepcopy(target_pose_stamped.pose)
                candidate_target_pose.orientation.x = q_new[0]
                candidate_target_pose.orientation.y = q_new[1]
                candidate_target_pose.orientation.z = q_new[2]
                candidate_target_pose.orientation.w = q_new[3]

                for back_distance in self.pre_grasp_back_distance_candidates:
                    pre_grasp_pose = copy.deepcopy(candidate_target_pose)
                    pre_grasp_pose.position.x -= robot_z[0] * back_distance
                    pre_grasp_pose.position.y -= robot_z[1] * back_distance
                    pre_grasp_pose.position.z -= robot_z[2] * back_distance

                    observation_pose = copy.deepcopy(pre_grasp_pose)
                    observation_pose.position.x -= robot_z[0] * self.wrist_observation_backoff
                    observation_pose.position.y -= robot_z[1] * self.wrist_observation_backoff
                    observation_pose.position.z -= robot_z[2] * self.wrist_observation_backoff

                    self.arm.move_group.set_start_state_to_current_state()
                    observation_q = self.get_ik_seeded(observation_pose, seed_q, planning_frame) if seed_q else None
                    if observation_q is not None:
                        rospy.loginfo(
                            f"✅ seeded IK 成功，使用关节角目标规划到观察位 "
                            f"(roll={roll_variant}, back_distance={back_distance:.3f} m)。"
                        )
                        self.arm.move_group.set_joint_value_target(observation_q)
                    else:
                        self.arm.move_group.set_pose_target(observation_pose)

                    plan_result = self.arm.move_group.plan()
                    if isinstance(plan_result, tuple):
                        success, plan = plan_result[0], plan_result[1]
                    else:
                        success, plan = len(plan_result.joint_trajectory.points) > 0, plan_result
                    self.arm.move_group.clear_pose_targets()

                    if success:
                        rospy.loginfo(f"🎉 成功锁定纯平行抓取姿态！(roll={roll_variant})")
                        best_plan = plan
                        best_pre_grasp_pose = pre_grasp_pose
                        best_target_pose = candidate_target_pose
                        best_observation_pose = observation_pose
                        best_robot_z = robot_z
                        selected_back_distance = back_distance
                        chosen_width = self.grasp_infos[i * 3] if len(self.grasp_infos) > i * 3 else 0.04
                        chosen_depth = self.grasp_infos[i * 3 + 2] if len(self.grasp_infos) > i * 3 + 2 else 0.03
                        break

                    rospy.logwarn(
                        f"⚠️ 姿态 {i+1} 在 roll={roll_variant}, back_distance={back_distance:.3f} m 时观察位不可达，尝试其他构型。"
                    )

                if success:
                    break

            if success:
                break

            rospy.logwarn(f"❌ 姿态 {i+1} 会撞货架，被淘汰！")

        if best_plan is None:
            rospy.logerr("所有候选姿态都不可达！")
            self.grasp_pose_array_received = None
            self.set_failure_reason("observation_unreachable", "all_candidates_unreachable")
            return False

        self.detach_grasped_object()
        self.gripper.open(width=0.08)
        self.task_figure_index += 1
        self.publish_active_grasp_target(
            best_target_pose,
            best_robot_z,
            selected_back_distance,
            width_hint=chosen_width,
            depth_hint=chosen_depth,
            pre_grasp_pose=best_pre_grasp_pose,
            observation_pose=best_observation_pose,
        )
        self.refresh_shelf_collision_space()

        rospy.loginfo("[4/6] 先移动到更靠后的观察位 (扩大手腕相机视野)...")
        if not self.arm.move_group.execute(best_plan, wait=True):
            rospy.logerr("移动失败！")
            self.set_failure_reason("observation_unreachable", "execute_observation_plan_failed")
            return False

        rospy.sleep(1.0) 
        self.save_task_keyframe("observe")
        
        # =========================================================================
        # 🌟🌟🌟 [步骤 4.5] VLM 语义微调 (TF 树空间刚体逆解版) 🌟🌟🌟
        # =========================================================================
        rospy.loginfo("[4.5/6] 📸 启动腕部相机，执行抓前对齐...")
        if self.use_iterative_wrist_alignment:
            if not self.run_wrist_alignment_loop(
                "[4.5/6] 腕部多轮对齐",
                max_iters=self.wrist_alignment_max_iters,
                require_visible=self.wrist_require_full_visibility,
            ):
                rospy.logwarn("🛑 腕部多轮对齐失败！")
                self.set_failure_reason("wrist_alignment_failed", "iterative_alignment_failed")
                return False
        else:
            fine_measurement = self.calculate_world_correction(return_details=True)
            fine_dx = np.clip(fine_measurement["world_dx"], -0.10, 0.10)
            fine_dy = np.clip(fine_measurement["world_dy"], -0.10, 0.10)
            fine_dz = np.clip(fine_measurement["world_dz"], -0.10, 0.10) if self.wrist_alignment_allow_vertical else 0.0
            if not self.apply_safe_world_correction(fine_dx, fine_dy, fine_dz):
                rospy.logwarn("🛑 腕部相机微调执行失败！")
                self.set_failure_reason("wrist_alignment_failed", "single_alignment_failed")
                return False
        rospy.sleep(0.5)
        # =========================================================================

        if self.wrist_observation_backoff > 1e-4:
            rospy.loginfo("[4.8/6] 从观察位推进到真正的预抓取位...")
            if not self.segmented_cartesian_move(
                dx=best_robot_z[0] * self.wrist_observation_backoff,
                dy=best_robot_z[1] * self.wrist_observation_backoff,
                dz=best_robot_z[2] * self.wrist_observation_backoff,
                description="[4.8/6] 靠近到预抓取位",
                step_size=min(0.02, self.wrist_observation_backoff),
                min_step=0.01,
                avoid_collisions=False
            ):
                rospy.logwarn("⚠️ 无法完全推进到预抓取位，将基于当前位置继续计算剩余插入距离。")
            rospy.sleep(0.3)

        if self.use_final_wrist_alignment:
            rospy.loginfo("[4.9/6] 在预抓取位做最后一次腕部对齐...")
            if not self.run_wrist_alignment_loop(
                "[4.9/6] 预抓取位终对齐",
                max_iters=1,
                require_visible=False,
            ):
                rospy.logwarn("🛑 预抓取位终对齐失败！")
                self.set_failure_reason("wrist_alignment_failed", "final_alignment_failed")
                return False
            rospy.sleep(0.2)
        
        rospy.loginfo(f"[5/6] 像抽屉一样水平滑入柜子...")
        extra_depth = self.compute_insert_extra_depth(
            width_hint=chosen_width,
            depth_hint=chosen_depth
        )
        total_insert_distance = self.compute_remaining_insert_distance(
            best_target_pose,
            best_robot_z,
            extra_depth,
        )
        total_insert_distance = self.compute_insert_distance_from_wrist(
            chosen_depth=chosen_depth,
            fallback_distance=total_insert_distance,
        )
        rospy.loginfo(
            f"[5/6] 插入距离规划: extra_depth={extra_depth:.3f} m, "
            f"final_total={total_insert_distance:.3f} m"
        )

        current_pre_insert_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        self.publish_active_grasp_target(
            best_target_pose,
            best_robot_z,
            selected_back_distance,
            width_hint=chosen_width,
            depth_hint=chosen_depth,
            pre_grasp_pose=current_pre_insert_pose,
            observation_pose=current_pre_insert_pose,
            insert_distance=total_insert_distance,
        )
        self.refresh_shelf_collision_space()

        in_dx = best_robot_z[0] * total_insert_distance
        in_dy = best_robot_z[1] * total_insert_distance
        in_dz = best_robot_z[2] * total_insert_distance

        if not self.segmented_cartesian_move(
            dx=in_dx,
            dy=in_dy,
            dz=in_dz,
            description="[5/6] 水平滑入柜子",
            step_size=self.insert_step_size,
            min_step=0.01,
            avoid_collisions=False
        ):
            rospy.logwarn("🛑 拦截：直线插入控制失败！")
            self.set_failure_reason("insert_failed", "cartesian_insert_failed")
            return False

        rospy.sleep(0.5)

        command_width, lightly_shrunk_width = self.compute_command_grasp_width(chosen_width)
        rospy.loginfo(
            f"[6/6] 闭合夹爪 (预测宽度: {chosen_width:.3f}m, "
            f"轻微收紧: {lightly_shrunk_width:.3f}m, 下发宽度: {command_width:.3f}m)"
        )
        if not self.gripper.close(width=command_width, force=self.grasp_force):
            rospy.logwarn("🛑 夹爪闭合动作失败，放弃附着碰撞体。")
            self.set_failure_reason("grasp_failed", "gripper_close_failed")
            return False
        self.last_grasp_width_hint = float(chosen_width)
        self.last_grasp_depth_hint = float(chosen_depth)
        self.current_item_collision_size = self.estimate_grasped_object_size(
            width_hint=chosen_width,
            depth_hint=chosen_depth,
        )
        attach_ok = self.attach_gazebo_object_if_possible(best_target_pose)
        if self.enable_gazebo_attach and getattr(self.gripper, "last_grasp_was_soft_success", False) and not attach_ok:
            rospy.logwarn("🛑 夹爪仅达到软成功，且 Gazebo 未找到可附着的目标物体，按未抓住处理。")
            self.set_failure_reason("grasp_failed", "soft_grasp_without_attach")
            return False
        self.attach_grasped_object(width_hint=chosen_width, depth_hint=chosen_depth)
        rospy.sleep(1.0)
        self.save_task_keyframe("grasped")

        rospy.loginfo(">>> 保持水平原路撤出...")
        if not self.segmented_cartesian_move(
            dx=-in_dx,
            dy=-in_dy,
            dz=-in_dz,
            description="保持水平原路撤出",
            step_size=self.retreat_step_size,
            min_step=0.01,
            avoid_collisions=False
        ):
            rospy.logwarn("🛑 撤出时发生异常，尝试强行回 Home！")
            self.set_failure_reason("retreat_failed", "cartesian_retreat_failed")
            return False

        rospy.sleep(0.5)

        # 额外后退确保完全离开货架，再抬升
        extra_retreat = self.pre_grasp_back_distance
        total_in = max(abs(in_dx), abs(in_dy), abs(in_dz))
        if total_in > 1e-6:
            rx = -in_dx / total_in * extra_retreat
            ry = -in_dy / total_in * extra_retreat
            rz = -in_dz / total_in * extra_retreat
            self.segmented_cartesian_move(
                dx=rx, dy=ry, dz=rz,
                description="额外后退离开货架",
                step_size=self.retreat_step_size,
                min_step=0.005,
                avoid_collisions=False
            )

        self.grasp_pose_array_received = None
        self.grasp_infos = []

        return True

    def place_object(self):
        if self.stop_requested:
            return False

        self.arm.move_group.stop()
        self.arm.move_group.clear_pose_targets()

        current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        basket_top_z = self.basket_z + self.basket_size[2]
        item_size = self.current_item_collision_size
        if item_size is None:
            item_size = self.attached_object_size
        if item_size is None:
            item_size = self.estimate_grasped_object_size(
                width_hint=self.last_grasp_width_hint,
                depth_hint=self.last_grasp_depth_hint,
            )
        item_half_height = 0.5 * float(item_size[2]) if item_size is not None else 0.05
        transport_clearance = max(self.place_hover_height, item_half_height + 0.015)
        release_z = basket_top_z + self.place_release_height_offset
        transit_z = basket_top_z + transport_clearance
        slot_x, slot_y = self.get_next_basket_slot()
        place_orientation = self.get_place_transport_orientation(current_pose)

        if self.place_orientation_mode == "downward":
            rospy.loginfo("🧭 放置阶段姿态模式: downward（末端切到固定朝下姿态后再搬运）。")
        else:
            rospy.loginfo("🧭 放置阶段姿态模式: preserve（保持抓取完成后的当前世界姿态）。")

        rospy.loginfo("[放置 1/3] 必要时调整到放置高度...")
        pre_align_orientation = current_pose.orientation if self.place_orientation_mode == "downward" else place_orientation
        if not self.move_vertical_to_z(
            transit_z,
            "放置前高度调整",
            orientation=pre_align_orientation,
            allow_position_only_fallback=self.place_allow_position_only_fallback,
        ):
            rospy.logwarn("⚠️ 无法调整到放置高度，放置中止。")
            self.set_failure_reason("place_transit_failed", "pre_place_height_adjust_failed")
            return False

        if self.place_orientation_mode == "downward":
            current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
            if not self.orientations_match(current_pose.orientation, place_orientation, tol_deg=3.0):
                rospy.loginfo("[放置 1.5/3] 切换到固定放置姿态...")
                if not self.align_current_pose_orientation(
                    place_orientation,
                    "切换到固定放置姿态",
                    retries=2,
                ):
                    rospy.logwarn("⚠️ 无法切换到固定放置姿态，放置中止。")
                    self.set_failure_reason("place_transit_failed", "align_place_orientation_failed")
                    return False

        rospy.loginfo(
            f"[放置 2/3] 平移到篮子上方: x={slot_x:.2f}, "
            f"y={slot_y:.2f}, z={transit_z:.2f}"
        )
        if not self.move_horizontal_to_xy(
            slot_x,
            slot_y,
            "移动到篮子上方安全位",
            orientation=place_orientation,
            allow_position_only_fallback=self.place_allow_position_only_fallback,
        ):
            rospy.logwarn("⚠️ 到达篮子上方失败，放置中止。")
            self.set_failure_reason("place_transit_failed", "move_over_basket_failed")
            return False

        if not self.simple_place_drop:
            rospy.loginfo("[放置 3/4] 垂直下降到释放高度...")
            if not self.move_vertical_to_z(
                release_z,
                "下放到释放高度",
                orientation=place_orientation,
                allow_position_only_fallback=self.place_allow_position_only_fallback,
            ):
                rospy.logwarn("⚠️ 下放到释放高度失败，放置中止。")
                self.set_failure_reason("place_drop_failed", "descend_to_release_failed")
                return False

        rospy.loginfo("[放置 3/3] 张开夹爪，释放物体。")
        detached_ok = self.detach_gazebo_object_if_needed()
        open_ok = self.gripper.open()
        if not open_ok and not detached_ok:
            rospy.logwarn("⚠️ 夹爪张开失败，且 Gazebo 物体未成功释放，放置中止。")
            self.set_failure_reason("place_drop_failed", "gripper_open_failed")
            return False
        if not open_ok and detached_ok:
            rospy.logwarn("⚠️ 夹爪张开返回失败，但 Gazebo 物体已释放，按已放置处理。")
        rospy.sleep(0.5)
        self.detach_grasped_object()
        self.clear_active_grasp_target()
        self.drop_counter += 1
        self.current_item_collision_size = None
        self.save_task_keyframe("placed")

        if self.refresh_collision_after_each_place:
            self.refresh_shelf_collision_space()

        self.arm.move_group.stop()
        self.arm.move_group.clear_pose_targets()
        return True

    def run_service(self):
        self.arm.go_to_home()
        while not rospy.is_shutdown():
            self.stop_requested = False
            self.clear_failure_reason()
            if self.grasp_pose_array_received is None:
                if self.show_grasp_debug and self.debug_rgb is not None: cv2.waitKey(1)
                rospy.sleep(0.1)
                continue

            rospy.loginfo(">>> 开始执行物理抓取序列...")
            self.is_running_task = True
            task_success = False
            task_start_wall = time.time()
            task_joint_cost = None

            if len(self.grasp_infos) >= 10:
                try:
                    current_q = np.array(self.arm.move_group.get_current_joint_values()[:7], dtype=np.float64)
                    target_q = np.array(self.grasp_infos[3:10], dtype=np.float64)
                    task_joint_cost = float(np.max(np.abs(target_q - current_q)))
                except Exception:
                    task_joint_cost = None

            try:
                if self.pick_object() and self.place_object():
                    rospy.loginfo("✅ 单次抓取-放置物理动作执行完毕！")
                    task_success = True
                else: rospy.logwarn("⚠️ 任务失败或中止。")
            except Exception as e:
                rospy.logerr(f"异常: {e}")

            self.grasp_pose_array_received = None
            self.grasp_infos = []
            self.is_running_task = False
            task_elapsed = time.time() - task_start_wall
            if task_joint_cost is not None:
                rospy.loginfo(
                    f"📊 本次任务统计: task_joint_cost={task_joint_cost:.4f} rad, "
                    f"task_time={task_elapsed:.3f} s"
                )
            else:
                rospy.loginfo(f"📊 本次任务统计: task_time={task_elapsed:.3f} s")
            if not task_success or self.stop_requested:
                rospy.loginfo(">>> 任务失败，回到 Home 保证安全。")
                self.detach_gazebo_object_if_needed()
                self.detach_grasped_object()
                self.clear_active_grasp_target()
                self.current_item_collision_size = None
                self.arm.go_to_home()
            rospy.loginfo(">>> 已复位，准备接收下一任务或等待调度器通知完成...")
            self.status_pub.publish("DONE" if task_success else "FAILED")

def main():
    rospy.init_node("pick_place_service", anonymous=True)
    connected = False
    rospy.loginfo("⏳ 正在尝试连接 MoveIt 动作服务器...")
    while not rospy.is_shutdown() and not connected:
        try:
            moveit_commander.roscpp_initialize(sys.argv)
            arm = ArmController("panda_manipulator")
            gripper = GripperActionController()
            demo = PickPlaceDemo(arm, gripper)
            connected = True
            rospy.loginfo("✅ 成功连接到 MoveIt！")
            demo.run_service()
        except RuntimeError: rospy.sleep(2.0)
        except Exception as e:
            rospy.logerr(f"💥 main循环发生致命崩溃！错误信息: {e}")
            break
        finally:
            if rospy.is_shutdown():
                try: cv2.destroyAllWindows()
                except Exception: pass

if __name__ == "__main__":
    main()
