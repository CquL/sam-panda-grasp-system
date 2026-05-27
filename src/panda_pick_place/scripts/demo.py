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
from std_msgs.msg import String, Float32, Header
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Vector3Stamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import GetModelState, GetModelProperties, GetLinkState
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionIKRequest
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
        self.latest_arm_joint_state = None
        self._task_joint_tracking_active = False
        self._task_joint_tracking_last = None
        self._task_joint_path_cost = 0.0
        self._task_joint_l1_path_cost = 0.0
        self.default_planning_time = float(rospy.get_param("~default_planning_time", 5.0))
        self.default_planning_attempts = int(rospy.get_param("~default_planning_attempts", 10))
        self.observation_planning_time = float(rospy.get_param("~observation_planning_time", 1.8))
        self.observation_planning_attempts = int(rospy.get_param("~observation_planning_attempts", 2))
        self.observation_lift_candidates = [
            float(v) for v in rospy.get_param("~observation_lift_candidates", [0.0, 0.02, 0.04, 0.06, 0.08])
        ]
        self.max_observation_search_time = float(rospy.get_param("~max_observation_search_time", 60.0))
        self.observation_ik_avoid_collisions = bool(
            rospy.get_param("~observation_ik_avoid_collisions", False)
        )
        self.enable_goal_state_validity_check = bool(
            rospy.get_param("~enable_goal_state_validity_check", True)
        )
        self.allow_relaxed_observation_orientation = bool(
            rospy.get_param("~allow_relaxed_observation_orientation", True)
        )
        self.observation_camera_axis_min_dot = float(
            rospy.get_param("~observation_camera_axis_min_dot", 0.97)
        )
        self.observation_ik_seed_trials = max(1, int(rospy.get_param("~observation_ik_seed_trials", 3)))
        self.observation_ik_timeout_sec = float(rospy.get_param("~observation_ik_timeout_sec", 0.5))
        self.use_gazebo_grasp_height_correction = bool(
            rospy.get_param("~use_gazebo_grasp_height_correction", True)
        )
        self.gazebo_grasp_height_correction_xy_threshold = float(
            rospy.get_param("~gazebo_grasp_height_correction_xy_threshold", 0.12)
        )
        self.gazebo_grasp_height_correction_max_delta = float(
            rospy.get_param("~gazebo_grasp_height_correction_max_delta", 0.14)
        )
        self.gazebo_grasp_height_correction_z_bias = float(
            rospy.get_param("~gazebo_grasp_height_correction_z_bias", 0.0)
        )
        self.observation_home_seed = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.last_observation_ik_seed = None
        self.last_place_ik_seed = None
        self.joint_limit_margin = float(rospy.get_param("~joint_limit_margin", 0.003))
        self.panda_joint_lower = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float64,
        )
        self.panda_joint_upper = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float64,
        )

        self.arm.move_group.set_planning_time(self.default_planning_time)
        self.arm.move_group.set_num_planning_attempts(self.default_planning_attempts)
        self.arm.move_group.set_goal_position_tolerance(0.01)
        self.arm.move_group.set_goal_orientation_tolerance(0.05)
        self.arm.move_group.set_goal_joint_tolerance(0.05)
        self.global_velocity_scaling = max(
            0.01, min(1.0, float(rospy.get_param("~global_velocity_scaling", 0.01)))
        )
        self.global_acceleration_scaling = max(
            0.01, min(1.0, float(rospy.get_param("~global_acceleration_scaling", 0.01)))
        )
        self.arm.move_group.set_max_velocity_scaling_factor(self.global_velocity_scaling)
        self.arm.move_group.set_max_acceleration_scaling_factor(self.global_acceleration_scaling)

        self.basket_x = float(rospy.get_param("~basket_x", 0.393296))
        self.basket_y = float(rospy.get_param("~basket_y", -0.315779))
        self.basket_z = float(rospy.get_param("~basket_z", 0.140719))
        self.basket_size = (0.36, 0.26, 0.16)
        self.basket_wall_thickness = 0.01
        basket_collision_param = rospy.get_param("~use_basket_collision_geometry", False)
        self.use_basket_collision_geometry = str(basket_collision_param).strip().lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        self.use_basket_slot_grid = bool(rospy.get_param("~use_basket_slot_grid", False))
        self.place_hover_height = float(rospy.get_param("~place_hover_height", 0.05))
        self.place_release_height_offset = float(rospy.get_param("~place_release_height_offset", 0.02))
        self.place_release_height_above_basket = float(
            rospy.get_param("~place_release_height_above_basket", 0.20)
        )
        self.place_orientation_mode = str(rospy.get_param("~place_orientation_mode", "preserve")).strip().lower()
        if self.place_orientation_mode not in ("preserve", "downward"):
            rospy.logwarn(
                f"⚠️ 未知的 place_orientation_mode={self.place_orientation_mode}，回退到 preserve。"
            )
            self.place_orientation_mode = "preserve"
        self.place_allow_position_only_fallback = bool(
            rospy.get_param("~place_allow_position_only_fallback", False)
        )
        self.place_path_orientation_tolerance_deg = float(
            rospy.get_param("~place_path_orientation_tolerance_deg", 4.0)
        )
        self.place_ik_timeout_sec = float(rospy.get_param("~place_ik_timeout_sec", 0.35))
        self.place_ik_seed_trials = max(1, int(rospy.get_param("~place_ik_seed_trials", 3)))
        self.plan_execution_start_tolerance = float(
            rospy.get_param("~plan_execution_start_tolerance", 0.01)
        )
        self.place_fallback_velocity_scaling = float(
            rospy.get_param("~place_fallback_velocity_scaling", 0.18)
        )
        self.place_fallback_acceleration_scaling = float(
            rospy.get_param("~place_fallback_acceleration_scaling", 0.12)
        )
        self.cartesian_retime = bool(rospy.get_param("~cartesian_retime", True))
        self.cartesian_velocity_scaling = float(rospy.get_param("~cartesian_velocity_scaling", 0.12))
        self.cartesian_acceleration_scaling = float(rospy.get_param("~cartesian_acceleration_scaling", 0.08))
        self.use_fixed_shelf_collision = bool(rospy.get_param("~use_fixed_shelf_collision", True))
        self.shelf_model_name = str(rospy.get_param("~shelf_model_name", "narrow_supermarket_shelf_enclosed_0"))
        self.shelf_pose_fallback = {
            "x": float(rospy.get_param("~shelf_pose_fallback_x", 0.737098)),
            "y": float(rospy.get_param("~shelf_pose_fallback_y", -0.148598)),
            "z": float(rospy.get_param("~shelf_pose_fallback_z", 0.205537)),
            "yaw": float(rospy.get_param("~shelf_pose_fallback_yaw", 0.0)),
        }
        self.force_shelf_normal_approach = bool(rospy.get_param("~force_shelf_normal_approach", True))
        (
            self.grasp_pose_approach_axis,
            self.grasp_pose_approach_sign,
            self.grasp_pose_approach_axis_label,
        ) = self.parse_grasp_axis_param(
            rospy.get_param("~grasp_pose_approach_axis", "+z"),
            default_axis="z",
            default_sign=1.0,
        )
        (
            self.grasp_pose_open_axis,
            self.grasp_pose_open_sign,
            self.grasp_pose_open_axis_label,
        ) = self.parse_grasp_axis_param(
            rospy.get_param("~grasp_pose_open_axis", "+x"),
            default_axis="x",
            default_sign=1.0,
        )
        self.use_grasp_pose_open_axis_orientation = bool(
            rospy.get_param("~use_grasp_pose_open_axis_orientation", False)
        )
        self.allow_raw_grasp_pose_orientation = bool(
            rospy.get_param("~allow_raw_grasp_pose_orientation", False)
        )
        self.enforce_horizontal_gripper_open_axis = bool(
            rospy.get_param("~enforce_horizontal_gripper_open_axis", True)
        )
        self.max_gripper_open_axis_vertical_component = float(
            rospy.get_param("~max_gripper_open_axis_vertical_component", 0.25)
        )
        self.execution_shelf_clearance_min = float(rospy.get_param("~execution_shelf_clearance_min", 0.025))
        self.simple_front_grasp_mode = bool(rospy.get_param("~simple_front_grasp_mode", True))
        self.simple_front_insert_extra_depth = float(rospy.get_param("~simple_front_insert_extra_depth", 0.010))
        self.simple_front_extra_retreat = float(rospy.get_param("~simple_front_extra_retreat", 0.080))
        self.grasp_transport_mode = str(rospy.get_param("~grasp_transport_mode", "force_only")).strip().lower()
        if self.grasp_transport_mode not in ("force_only", "moveit_attached", "gazebo_attach", "hybrid"):
            rospy.logwarn(
                f"⚠️ 未知 grasp_transport_mode={self.grasp_transport_mode}，回退到 force_only。"
            )
            self.grasp_transport_mode = "force_only"
        requested_enable_gazebo_attach = bool(rospy.get_param("~enable_gazebo_attach", False))
        self.gazebo_attach_distance_threshold = float(rospy.get_param("~gazebo_attach_distance_threshold", 0.18))
        self.gazebo_attach_link_name = str(rospy.get_param("~gazebo_attach_link_name", "panda_link8"))
        if self.grasp_transport_mode == "force_only":
            self.enable_gazebo_attach = False
            self.enable_attached_collision_object = False
            if requested_enable_gazebo_attach:
                rospy.logwarn("⚠️ 当前为 force_only 模式，忽略 enable_gazebo_attach=true。")
        elif self.grasp_transport_mode == "moveit_attached":
            self.enable_gazebo_attach = False
            if requested_enable_gazebo_attach:
                rospy.logwarn("⚠️ 当前为 moveit_attached 模式，忽略 enable_gazebo_attach=true。")
        elif self.grasp_transport_mode == "gazebo_attach":
            self.enable_gazebo_attach = requested_enable_gazebo_attach
            self.enable_attached_collision_object = False
            if not self.enable_gazebo_attach:
                rospy.logwarn("⚠️ 当前为 gazebo_attach 模式，但 enable_gazebo_attach=false，运行时不会执行 Gazebo 附着。")
        else:
            self.enable_gazebo_attach = requested_enable_gazebo_attach
        self.drop_counter = 0
        self.current_item_collision_size = None
        self.last_grasp_width_hint = 0.04
        self.last_grasp_depth_hint = 0.03
        self.gazebo_model_states = None
        self.attached_gazebo_model = None
        self.attached_gazebo_link = None
        self.functional_grasp_candidate_model = None
        self.functional_grasp_candidate_link = None
        self.functional_grasp_start_object_pos = None
        self.functional_grasp_start_ee_pos = None

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
        self.observation_back_distance_candidates = sorted(
            set([0.08, 0.10] + [float(v) for v in self.pre_grasp_back_distance_candidates])
        )
        self.wrist_observation_backoff = rospy.get_param("~wrist_observation_backoff", 0.04)
        self.wrist_depth_band = rospy.get_param("~wrist_depth_band", 0.05)
        self.wrist_front_depth_percentile = rospy.get_param("~wrist_front_depth_percentile", 20.0)
        self.wrist_min_valid_depth = float(rospy.get_param("~wrist_min_valid_depth", 0.02))
        self.wrist_max_valid_depth = float(rospy.get_param("~wrist_max_valid_depth", 0.50))
        self.wrist_alignment_pixel_mode = str(
            rospy.get_param("~wrist_alignment_pixel_mode", "bbox_center")
        ).strip().lower()
        if self.wrist_alignment_pixel_mode not in ("bbox_center", "tight_bbox_center", "front_surface"):
            rospy.logwarn(
                "⚠️ 未知 wrist_alignment_pixel_mode=%s，回退到 bbox_center。",
                self.wrist_alignment_pixel_mode,
            )
            self.wrist_alignment_pixel_mode = "bbox_center"
        self.wrist_default_fx = float(rospy.get_param("~wrist_default_fx", 500.0))
        self.wrist_default_fy = float(rospy.get_param("~wrist_default_fy", 500.0))
        self.wrist_alignment_target_hand_x = float(rospy.get_param("~wrist_alignment_target_hand_x", 0.0))
        self.wrist_alignment_target_hand_y = float(rospy.get_param("~wrist_alignment_target_hand_y", 0.0))
        self.wrist_alignment_target_hand_z = float(rospy.get_param("~wrist_alignment_target_hand_z", 0.0))
        self.wrist_alignment_max_step_xy = float(rospy.get_param("~wrist_alignment_max_step_xy", 0.03))
        self.wrist_alignment_max_step_z = float(rospy.get_param("~wrist_alignment_max_step_z", 0.015))
        self.wrist_vlm_timeout_sec = float(rospy.get_param("~wrist_vlm_timeout_sec", 4.0))
        self.require_wrist_alignment = bool(rospy.get_param("~require_wrist_alignment", False))
        self.wrist_alignment_before_pregrasp = bool(
            rospy.get_param("~wrist_alignment_before_pregrasp", True)
        )
        self._wrist_vlm_request_seq = 0
        self.enable_wrist_tf_guard = bool(rospy.get_param("~enable_wrist_tf_guard", True))
        self.wrist_tf_guard_trans_tol = float(rospy.get_param("~wrist_tf_guard_trans_tol", 0.002))
        self.wrist_tf_guard_rot_tol_deg = float(rospy.get_param("~wrist_tf_guard_rot_tol_deg", 0.8))
        self.insert_extra_depth_min = rospy.get_param("~insert_extra_depth_min", 0.06)
        self.insert_depth_margin = rospy.get_param("~insert_depth_margin", 0.035)
        self.insert_extra_depth_max = rospy.get_param("~insert_extra_depth_max", 0.09)
        self.insert_distance_over_back_cap = float(
            rospy.get_param("~insert_distance_over_back_cap", 0.08)
        )
        self.insert_distance_absolute_max = float(
            rospy.get_param("~insert_distance_absolute_max", 0.20)
        )
        self.use_pose_preserving_transport = rospy.get_param("~use_pose_preserving_transport", False)
        self.simple_place_drop = bool(rospy.get_param("~simple_place_drop", True))
        self.fast_drop_when_inside_basket = bool(rospy.get_param("~fast_drop_when_inside_basket", True))
        self.fast_drop_inner_margin_xy = float(rospy.get_param("~fast_drop_inner_margin_xy", 0.015))
        self.fast_drop_min_height_above_top = float(rospy.get_param("~fast_drop_min_height_above_top", 0.02))
        self.refresh_collision_after_each_place = bool(
            rospy.get_param("~refresh_collision_after_each_place", False)
        )
        self.transport_step_size = float(rospy.get_param("~transport_step_size", 0.025))
        self.insert_step_size = float(rospy.get_param("~insert_step_size", 0.025))
        self.retreat_step_size = float(rospy.get_param("~retreat_step_size", 0.025))
        self.lift_step_size = float(rospy.get_param("~lift_step_size", 0.012))
        self.post_grasp_lift_distance = float(rospy.get_param("~post_grasp_lift_distance", 0.015))
        self.cartesian_eef_step = float(rospy.get_param("~cartesian_eef_step", 0.005))
        self.grasp_force = float(rospy.get_param("~grasp_force", 50.0))
        self.grasp_width_margin = float(rospy.get_param("~grasp_width_margin", 0.008))
        self.grasp_width_margin_ratio = float(rospy.get_param("~grasp_width_margin_ratio", 0.060))
        self.grasp_command_width_min = float(rospy.get_param("~grasp_command_width_min", 0.008))
        self.grasp_command_width_max = float(rospy.get_param("~grasp_command_width_max", 0.078))
        self.grasp_width_margin_wide = float(rospy.get_param("~grasp_width_margin_wide", 0.012))
        self.grasp_width_margin_wide_threshold = float(
            rospy.get_param("~grasp_width_margin_wide_threshold", 0.070)
        )
        self.last_grasp_width_margin_used = self.grasp_width_margin
        self.grasp_probe_retreat_distance = float(rospy.get_param("~grasp_probe_retreat_distance", 0.0))
        self.grasp_probe_wait_sec = float(rospy.get_param("~grasp_probe_wait_sec", 0.10))
        self.enable_grasp_resqueeze_retry = bool(rospy.get_param("~enable_grasp_resqueeze_retry", True))
        self.grasp_resqueeze_extra_margin = float(rospy.get_param("~grasp_resqueeze_extra_margin", 0.006))
        self.grasp_resqueeze_force_boost = float(rospy.get_param("~grasp_resqueeze_force_boost", 10.0))
        self.grasp_empty_close_threshold = float(rospy.get_param("~grasp_empty_close_threshold", 0.010))
        self.force_grasp_verify_after_close = bool(
            rospy.get_param("~force_grasp_verify_after_close", True)
        )
        self.force_grasp_verify_after_retreat = bool(
            rospy.get_param("~force_grasp_verify_after_retreat", True)
        )
        self.force_grasp_verify_wait_sec = float(
            rospy.get_param("~force_grasp_verify_wait_sec", 0.25)
        )
        self.gazebo_functional_grasp_verify = bool(
            rospy.get_param("~gazebo_functional_grasp_verify", True)
        )
        self.gazebo_functional_grasp_max_object_to_ee = float(
            rospy.get_param("~gazebo_functional_grasp_max_object_to_ee", 0.14)
        )
        self.gazebo_functional_grasp_min_object_motion = float(
            rospy.get_param("~gazebo_functional_grasp_min_object_motion", 0.015)
        )
        self.gazebo_functional_grasp_min_follow_ratio = float(
            rospy.get_param("~gazebo_functional_grasp_min_follow_ratio", 0.35)
        )
        self.gazebo_functional_grasp_probe_min_object_motion = float(
            rospy.get_param("~gazebo_functional_grasp_probe_min_object_motion", 0.006)
        )
        self.gazebo_functional_grasp_probe_min_follow_ratio = float(
            rospy.get_param("~gazebo_functional_grasp_probe_min_follow_ratio", 0.30)
        )
        self.grasp_roll_variants = [
            str(v) for v in rospy.get_param("~grasp_roll_variants", ["x_down", "x_up"])
        ]
        self.use_iterative_wrist_alignment = bool(rospy.get_param("~use_iterative_wrist_alignment", True))
        self.wrist_alignment_max_iters = int(rospy.get_param("~wrist_alignment_max_iters", 3))
        self.use_wrist_guided_insert_distance = bool(rospy.get_param("~use_wrist_guided_insert_distance", True))
        self.min_insert_distance = float(rospy.get_param("~min_insert_distance", 0.02))
        self.wrist_visibility_margin_px = int(rospy.get_param("~wrist_visibility_margin_px", 8))
        self.wrist_require_full_visibility = bool(rospy.get_param("~wrist_require_full_visibility", False))
        self.wrist_visibility_fallback_max_insert = float(
            rospy.get_param("~wrist_visibility_fallback_max_insert", 0.09)
        )
        self.wrist_min_insert_fraction = float(rospy.get_param("~wrist_min_insert_fraction", 0.60))
        self.use_final_wrist_alignment = bool(rospy.get_param("~use_final_wrist_alignment", False))
        self.wrist_alignment_allow_vertical = bool(rospy.get_param("~wrist_alignment_allow_vertical", True))
        self.wrist_alignment_use_camera_plane_tf = bool(
            rospy.get_param("~wrist_alignment_use_camera_plane_tf", True)
        )
        self.final_preinsert_world_z_lift = float(
            rospy.get_param("~final_preinsert_world_z_lift", 0.0005)
        )
        self.pregrasp_retain_observation_lift = bool(
            rospy.get_param("~pregrasp_retain_observation_lift", True)
        )
        self.pregrasp_retained_lift_max = float(rospy.get_param("~pregrasp_retained_lift_max", 0.06))
        self.wrist_extra_insert_max = float(rospy.get_param("~wrist_extra_insert_max", 0.03))
        self.wrist_insert_allow_shorten = bool(rospy.get_param("~wrist_insert_allow_shorten", False))
        self.wrist_insert_shorten_max = float(rospy.get_param("~wrist_insert_shorten_max", 0.08))
        self.wrist_insert_depth_scale = float(rospy.get_param("~wrist_insert_depth_scale", 0.35))
        self.wrist_insert_depth_cap = float(rospy.get_param("~wrist_insert_depth_cap", 0.015))
        self.wrist_alignment_tol = float(rospy.get_param("~wrist_alignment_tol", 0.008))
        self.wrist_grasp_depth_margin = float(rospy.get_param("~wrist_grasp_depth_margin", 0.005))
        self.wrist_grasp_forward_min = float(rospy.get_param("~wrist_grasp_forward_min", 0.075))
        self.wrist_grasp_forward_max = float(rospy.get_param("~wrist_grasp_forward_max", 0.110))
        self.use_wrist_grasp_refinement = bool(rospy.get_param("~use_wrist_grasp_refinement", False))
        self.wrist_object_cloud_topic = str(
            rospy.get_param("~wrist_object_cloud_topic", "/sam_perception/wrist_object_cloud")
        )
        self.wrist_grasp_pose_topic = str(
            rospy.get_param("~wrist_grasp_pose_topic", "/graspnet/wrist_grasp_pose_array")
        )
        self.wrist_grasp_info_topic = str(
            rospy.get_param("~wrist_grasp_info_topic", "/graspnet/wrist_grasp_info")
        )
        self.wrist_grasp_timeout_sec = float(rospy.get_param("~wrist_grasp_timeout_sec", 6.0))
        self.wrist_grasp_bbox_margin_px = int(rospy.get_param("~wrist_grasp_bbox_margin_px", 8))
        self.wrist_grasp_depth_band = float(rospy.get_param("~wrist_grasp_depth_band", self.wrist_depth_band))
        self.wrist_grasp_min_points = int(rospy.get_param("~wrist_grasp_min_points", 80))
        self.wrist_grasp_cloud_max_points = int(rospy.get_param("~wrist_grasp_cloud_max_points", 12000))
        self.wrist_grasp_min_approach_alignment = float(
            rospy.get_param("~wrist_grasp_min_approach_alignment", 0.25)
        )
        self.refresh_collision_before_pick = bool(rospy.get_param("~refresh_collision_before_pick", True))
        self.grasp_corridor_padding = float(rospy.get_param("~grasp_corridor_padding", 0.03))
        self.enable_direct_grasp_fallback = bool(rospy.get_param("~enable_direct_grasp_fallback", True))
        self.direct_grasp_back_distance_candidates = [
            float(v) for v in rospy.get_param("~direct_grasp_back_distance_candidates", [0.04, 0.06, 0.08])
        ]
        self.direct_grasp_planning_time = float(rospy.get_param("~direct_grasp_planning_time", 3.0))
        self.direct_grasp_planning_attempts = int(rospy.get_param("~direct_grasp_planning_attempts", 5))
        self.direct_grasp_retreat_distance = float(rospy.get_param("~direct_grasp_retreat_distance", 0.08))
        self.direct_grasp_require_width_hold = bool(rospy.get_param("~direct_grasp_require_width_hold", True))
        self.grasp_sub = rospy.Subscriber('/graspnet/grasp_pose_array', PoseArray, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber('/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)
        self.wrist_grasp_pose_array_received = None
        self.wrist_grasp_infos = []
        self.wrist_grasp_sub = rospy.Subscriber(
            self.wrist_grasp_pose_topic, PoseArray, self.wrist_grasp_pose_callback, queue_size=1
        )
        self.wrist_grasp_info_sub = rospy.Subscriber(
            self.wrist_grasp_info_topic, Float32MultiArray, self.wrist_grasp_info_callback, queue_size=1
        )
        self.cmd_sub = rospy.Subscriber('/demo/command', String, self.ui_command_callback)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.debug_rgb_cb, queue_size=1)
        self.info_cam_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.debug_info_cb, queue_size=1)

        self.wrist_rgb = None
        self.wrist_depth = None
        self.wrist_K = None
        self.wrist_cam_in_ee_baseline = None
        rospy.Subscriber('/wrist_camera/color/image_raw', Image, self.wrist_rgb_cb)
        rospy.Subscriber('/wrist_camera/depth/image_raw', Image, self.wrist_depth_cb)
        rospy.Subscriber('/wrist_camera/color/camera_info', CameraInfo, self.wrist_info_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_model_states_cb, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.joint_states_cb, queue_size=50)

        self.status_pub = rospy.Publisher('/demo/task_status', String, queue_size=10)
        self.task_metrics_pub = rospy.Publisher('/demo/task_metrics', String, queue_size=20)
        self.failure_reason_pub = rospy.Publisher('/demo/failure_reason', String, queue_size=10)
        self.cartesian_fraction_pub = rospy.Publisher('/demo/cartesian_plan_fraction', Float32, queue_size=50)
        self.active_grasp_target_pub = rospy.Publisher('/sam_perception/active_grasp_target', String, queue_size=1, latch=True)
        self.wrist_object_cloud_pub = rospy.Publisher(self.wrist_object_cloud_topic, PointCloud2, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.wait_for_service('/compute_ik')
        self.compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        self.check_state_validity = None
        if self.enable_goal_state_validity_check:
            try:
                rospy.wait_for_service('/check_state_validity', timeout=2.0)
                self.check_state_validity = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
                rospy.loginfo("✅ MoveIt 状态有效性检查服务就绪！")
            except Exception as exc:
                rospy.logwarn("⚠️ /check_state_validity 不可用，跳过候选关节态碰撞预筛: %s", exc)
        self.refresh_background = rospy.ServiceProxy('/sam_perception/refresh_background', Empty, persistent=True)
        self.get_model_state = None
        self.get_model_properties = None
        self.get_link_state = None
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
        try:
            rospy.wait_for_service('/gazebo/get_link_state', timeout=1.0)
            self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState, persistent=True)
        except Exception:
            rospy.logwarn("⚠️ 未连接到 /gazebo/get_link_state，将无法用模型 link 中心修正抓取高度。")

        # 增加触发 VLM 的发布器
        self.pub_wrist_trigger = rospy.Publisher('/wrist_vlm/trigger', String, queue_size=1)
        rospy.loginfo("=" * 50)
        rospy.loginfo(">> 机械臂后台执行器已就绪！")
        rospy.loginfo(">> 等待调度器下发任务...")
        rospy.loginfo(
            "🐢 运动速度限制: global_vel=%.3f, global_acc=%.3f, cartesian_vel=%.3f, cartesian_acc=%.3f, place_fallback_vel=%.3f, place_fallback_acc=%.3f",
            self.global_velocity_scaling,
            self.global_acceleration_scaling,
            self.cartesian_velocity_scaling,
            self.cartesian_acceleration_scaling,
            self.place_fallback_velocity_scaling,
            self.place_fallback_acceleration_scaling,
        )
        rospy.loginfo(
            "🤏 闭爪宽度策略: normal_margin=%.3f m, wide_margin=%.3f m, ratio=%.3f, wide_threshold=%.3f m",
            self.grasp_width_margin,
            self.grasp_width_margin_wide,
            self.grasp_width_margin_ratio,
            self.grasp_width_margin_wide_threshold,
        )
        rospy.loginfo(
            "🪛 抓后探测策略: probe_retreat=%.3f m, probe_wait=%.2f s, resqueeze_retry=%s, "
            "resqueeze_extra_margin=%.3f m, resqueeze_force_boost=%.1f N, "
            "empty_close_threshold=%.3f m, post_grasp_lift=%.3f m",
            self.grasp_probe_retreat_distance,
            self.grasp_probe_wait_sec,
            str(self.enable_grasp_resqueeze_retry).lower(),
            self.grasp_resqueeze_extra_margin,
            self.grasp_resqueeze_force_boost,
            self.grasp_empty_close_threshold,
            self.post_grasp_lift_distance,
        )
        rospy.loginfo(
            "📸 腕部对齐策略: pixel_mode=%s, iterative=%s, max_iters=%d, allow_vertical=%s, "
            "max_step_xy=%.3f m, max_step_z=%.3f m, vlm_timeout=%.1f s, "
            "required=%s, before_pregrasp=%s, final=%s, camera_plane_tf=%s, final_z_lift=%.4f m",
            self.wrist_alignment_pixel_mode,
            str(self.use_iterative_wrist_alignment).lower(),
            self.wrist_alignment_max_iters,
            str(self.wrist_alignment_allow_vertical).lower(),
            self.wrist_alignment_max_step_xy,
            self.wrist_alignment_max_step_z,
            self.wrist_vlm_timeout_sec,
            str(self.require_wrist_alignment).lower(),
            str(self.wrist_alignment_before_pregrasp).lower(),
            str(self.use_final_wrist_alignment).lower(),
            str(self.wrist_alignment_use_camera_plane_tf).lower(),
            self.final_preinsert_world_z_lift,
        )
        rospy.loginfo(
            "👁️ 观察位搜索策略: roll_variants=%s, back_candidates=%s, lifts=%s, ik_timeout=%.2f s, relaxed_obs_orientation=%s, camera_axis_min_dot=%.2f",
            str(self.grasp_roll_variants),
            str([round(v, 3) for v in self.observation_back_distance_candidates]),
            str([round(v, 3) for v in self.observation_lift_candidates]),
            self.observation_ik_timeout_sec,
            str(self.allow_relaxed_observation_orientation).lower(),
            self.observation_camera_axis_min_dot,
        )
        rospy.loginfo(
            "🧯 工程兜底直接抓取: enabled=%s, close_pose_back_candidates=%s, planning=%.1fs/%d attempts, retreat=%.3f m, require_width_hold=%s",
            str(self.enable_direct_grasp_fallback).lower(),
            str([round(v, 3) for v in self.direct_grasp_back_distance_candidates]),
            self.direct_grasp_planning_time,
            self.direct_grasp_planning_attempts,
            self.direct_grasp_retreat_distance,
            str(self.direct_grasp_require_width_hold).lower(),
        )
        rospy.loginfo(
            "📐 Gazebo目标高度修正: enabled=%s, xy_threshold=%.3f m, max_delta=%.3f m, z_bias=%.3f m",
            str(self.use_gazebo_grasp_height_correction).lower(),
            self.gazebo_grasp_height_correction_xy_threshold,
            self.gazebo_grasp_height_correction_max_delta,
            self.gazebo_grasp_height_correction_z_bias,
        )
        rospy.loginfo(
            "🧭 Grasp pose轴映射: approach=%s -> Panda TCP +Z, open=%s -> Panda TCP +Y, "
            "use_open_axis=%s, allow_raw_orientation=%s, horizontal_open_axis=%s(max_z=%.2f)",
            self.grasp_pose_approach_axis_label,
            self.grasp_pose_open_axis_label,
            str(self.use_grasp_pose_open_axis_orientation).lower(),
            str(self.allow_raw_grasp_pose_orientation).lower(),
            str(self.enforce_horizontal_gripper_open_axis).lower(),
            self.max_gripper_open_axis_vertical_component,
        )
        rospy.loginfo(
            "📏 插入深度策略: wrist_guided=%s, allow_shorten=%s, shorten_max=%.3f m, "
            "min_fraction=%.2f, vis_fallback_cap=%.3f m, over_back_cap=%.3f m, "
            "absolute_max=%.3f m, depth_scale=%.2f, depth_cap=%.3f m, forward_window=[%.3f, %.3f] m",
            str(self.use_wrist_guided_insert_distance).lower(),
            str(self.wrist_insert_allow_shorten).lower(),
            self.wrist_insert_shorten_max,
            self.wrist_min_insert_fraction,
            self.wrist_visibility_fallback_max_insert,
            self.insert_distance_over_back_cap,
            self.insert_distance_absolute_max,
            self.wrist_insert_depth_scale,
            self.wrist_insert_depth_cap,
            self.wrist_grasp_forward_min,
            self.wrist_grasp_forward_max,
        )
        rospy.loginfo(
            "🧷 固定前插执行模式: enabled=%s, extra_insert=%.3f m, extra_retreat=%.3f m",
            str(self.simple_front_grasp_mode).lower(),
            self.simple_front_insert_extra_depth,
            self.simple_front_extra_retreat,
        )
        rospy.loginfo(
            "🧪 腕部二次GraspNet: enabled=%s, cloud=%s, poses=%s, timeout=%.1f s, "
            "bbox_margin=%d px, depth_band=%.3f m, min_points=%d",
            str(self.use_wrist_grasp_refinement).lower(),
            self.wrist_object_cloud_topic,
            self.wrist_grasp_pose_topic,
            self.wrist_grasp_timeout_sec,
            self.wrist_grasp_bbox_margin_px,
            self.wrist_grasp_depth_band,
            self.wrist_grasp_min_points,
        )
        rospy.loginfo(
            "🧲 抓取运输策略: mode=%s, gazebo_attach=%s, moveit_attached_collision=%s",
            self.grasp_transport_mode,
            str(self.enable_gazebo_attach).lower(),
            str(self.enable_attached_collision_object).lower(),
        )
        rospy.loginfo(
            "🤲 纯力抓取校验: after_close=%s, after_retreat=%s, wait=%.2f s",
            str(self.force_grasp_verify_after_close).lower(),
            str(self.force_grasp_verify_after_retreat).lower(),
            self.force_grasp_verify_wait_sec,
        )
        rospy.loginfo(
            "🧪 Gazebo 功能性抓取校验: enabled=%s, max_object_to_ee=%.3f m, "
            "min_object_motion=%.3f m, min_follow_ratio=%.2f, "
            "probe_min_object_motion=%.3f m, probe_min_follow_ratio=%.2f",
            str(self.gazebo_functional_grasp_verify).lower(),
            self.gazebo_functional_grasp_max_object_to_ee,
            self.gazebo_functional_grasp_min_object_motion,
            self.gazebo_functional_grasp_min_follow_ratio,
            self.gazebo_functional_grasp_probe_min_object_motion,
            self.gazebo_functional_grasp_probe_min_follow_ratio,
        )
        rospy.loginfo(
            "🧺 篮子目标: x=%.3f, y=%.3f, z=%.3f, size=(%.3f, %.3f, %.3f), collision_geometry=%s",
            self.basket_x,
            self.basket_y,
            self.basket_z,
            self.basket_size[0],
            self.basket_size[1],
            self.basket_size[2],
            str(self.use_basket_collision_geometry).lower(),
        )
        rospy.loginfo("=" * 50)
        self.add_fixed_shelf_collision_geometry()
        if self.use_basket_collision_geometry:
            self.add_basket_collision_geometry()
        else:
            self.remove_basket_collision_geometry()
        self.detach_grasped_object()

    def save_task_keyframe(self, label):
        if not self.save_task_keyframes or self.debug_rgb is None:
            return
        os.makedirs(self.figure_output_dir, exist_ok=True)
        filename = f"task_{self.task_figure_index:03d}_{label}.png"
        cv2.imwrite(os.path.join(self.figure_output_dir, filename), self.debug_rgb)

    def log_gripper_width_snapshot(self, label):
        getter = getattr(self.gripper, "get_current_gripper_width", None)
        if getter is None:
            return
        try:
            width = getter()
        except Exception as exc:
            rospy.logwarn(f"⚠️ 读取夹爪宽度失败（{label}）: {exc}")
            return
        if width is None:
            rospy.loginfo(f"🤏 [{label}] 当前夹爪宽度未知（尚未收到 joint_states）。")
            return
        rospy.loginfo(f"🤏 [{label}] 当前夹爪宽度约 {float(width):.3f} m")

    def clear_functional_grasp_candidate(self):
        self.functional_grasp_candidate_model = None
        self.functional_grasp_candidate_link = None
        self.functional_grasp_start_object_pos = None
        self.functional_grasp_start_ee_pos = None

    def get_current_ee_position(self):
        try:
            pose = self.arm.move_group.get_current_pose().pose
            return np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ],
                dtype=np.float64,
            )
        except Exception:
            return None

    def get_gazebo_model_position(self, model_name):
        if self.gazebo_model_states is None or not model_name:
            return None
        try:
            idx = list(self.gazebo_model_states.name).index(model_name)
        except Exception:
            return None
        try:
            pose = self.gazebo_model_states.pose[idx]
            return np.array(
                [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ],
                dtype=np.float64,
            )
        except Exception:
            return None

    def begin_gazebo_functional_grasp_tracking(self, target_pose):
        self.clear_functional_grasp_candidate()
        if not self.gazebo_functional_grasp_verify or self.grasp_transport_mode != "force_only":
            return False

        model_name, link_name = self.find_nearest_gazebo_object(target_pose)
        if model_name is None:
            rospy.logwarn("⚠️ 未找到可用于 Gazebo 功能性抓取校验的目标物体。")
            return False

        object_pos = self.get_gazebo_model_position(model_name)
        ee_pos = self.get_current_ee_position()
        if object_pos is None or ee_pos is None:
            rospy.logwarn("⚠️ Gazebo 功能性抓取校验初始化失败：无法读取物体或末端位置。")
            return False

        self.functional_grasp_candidate_model = model_name
        self.functional_grasp_candidate_link = link_name
        self.functional_grasp_start_object_pos = object_pos
        self.functional_grasp_start_ee_pos = ee_pos
        rospy.loginfo(
            "🧪 已锁定 Gazebo 功能性抓取校验目标: model=%s, link=%s",
            model_name,
            link_name,
        )
        return True

    def can_attempt_gazebo_functional_recovery(self, requested_width):
        if (
            not self.gazebo_functional_grasp_verify
            or self.grasp_transport_mode != "force_only"
            or self.functional_grasp_candidate_model is None
        ):
            return False
        observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
        if observed_width is None:
            return False
        recoverable_upper = min(
            float(self.grasp_command_width_max) - 1e-3,
            float(requested_width) + 0.015,
        )
        return float(observed_width) <= recoverable_upper

    def verify_gazebo_functional_grasp(
        self,
        stage_label,
        min_object_motion=None,
        min_follow_ratio=None,
    ):
        if (
            not self.gazebo_functional_grasp_verify
            or self.grasp_transport_mode != "force_only"
            or self.functional_grasp_candidate_model is None
        ):
            return None

        object_pos = self.get_gazebo_model_position(self.functional_grasp_candidate_model)
        ee_pos = self.get_current_ee_position()
        if (
            object_pos is None
            or ee_pos is None
            or self.functional_grasp_start_object_pos is None
            or self.functional_grasp_start_ee_pos is None
        ):
            rospy.logwarn(f"⚠️ [{stage_label}] Gazebo 功能性抓取校验失败：位姿数据不可用。")
            return None

        object_motion = float(np.linalg.norm(object_pos - self.functional_grasp_start_object_pos))
        ee_motion = float(np.linalg.norm(ee_pos - self.functional_grasp_start_ee_pos))
        object_to_ee = float(np.linalg.norm(object_pos - ee_pos))
        min_object_motion_threshold = (
            float(self.gazebo_functional_grasp_min_object_motion)
            if min_object_motion is None
            else float(min_object_motion)
        )
        min_follow_ratio_threshold = (
            float(self.gazebo_functional_grasp_min_follow_ratio)
            if min_follow_ratio is None
            else float(min_follow_ratio)
        )
        min_required_motion = max(
            min_object_motion_threshold,
            min(0.05, 0.20 * ee_motion),
        )
        follow_ratio = object_motion / max(ee_motion, 1e-6) if ee_motion > 1e-6 else 0.0
        success = (
            object_to_ee <= float(self.gazebo_functional_grasp_max_object_to_ee)
            and object_motion >= min_required_motion
            and (ee_motion < 0.01 or follow_ratio >= min_follow_ratio_threshold)
        )
        log_fn = rospy.loginfo if success else rospy.logwarn
        log_fn(
            "🧪 [%s] Gazebo 功能性抓取校验: model=%s, object_to_ee=%.3f m, "
            "object_motion=%.3f m, ee_motion=%.3f m, follow_ratio=%.2f, "
            "min_required_motion=%.3f m, min_follow_ratio=%.2f",
            stage_label,
            self.functional_grasp_candidate_model,
            object_to_ee,
            object_motion,
            ee_motion,
            follow_ratio,
            min_required_motion,
            min_follow_ratio_threshold,
        )
        return bool(success)

    def verify_force_grasp_hold(self, label, requested_width, wait_sec=None):
        if self.grasp_transport_mode != "force_only":
            return True

        wait_time = self.force_grasp_verify_wait_sec if wait_sec is None else float(wait_sec)
        if wait_time > 1e-4:
            rospy.sleep(wait_time)

        evaluator = getattr(self.gripper, "evaluate_soft_grasp", None)
        if evaluator is None:
            rospy.loginfo(f"🤏 [{label}] 当前夹爪控制器未提供持有校验接口，跳过纯力抓取校验。")
            return True

        observed_width = None
        getter = getattr(self.gripper, "get_current_gripper_width", None)
        if getter is not None:
            try:
                observed_width = getter()
            except Exception as exc:
                rospy.logwarn(f"⚠️ [{label}] 读取夹爪宽度失败，改用最近观测值: {exc}")
        if observed_width is None:
            observed_width = getattr(self.gripper, "last_observed_gripper_width", None)

        diag = evaluator(requested_width, observed_width=observed_width)
        current_width = diag.get("current_width")
        if current_width is None:
            rospy.logwarn(f"⚠️ [{label}] 纯力抓取校验跳过：当前夹爪宽度未知。")
            return True

        rospy.loginfo(
            "🤏 [%s] 纯力抓取持有校验: observed=%.3f m, requested=%.3f m, "
            "accept_window=[%.3f, %.3f] m",
            label,
            float(current_width),
            float(requested_width),
            float(diag.get("lower_bound", requested_width)),
            float(diag.get("upper_bound", requested_width)),
        )
        if bool(diag.get("success", False)):
            rospy.loginfo(f"✅ [{label}] 纯力抓取持有校验通过。")
            return True

        rospy.logwarn(
            "⚠️ [%s] 纯力抓取持有校验失败: observed=%.3f m, occupied=%s, within_window=%s",
            label,
            float(current_width),
            str(bool(diag.get("occupied", False))).lower(),
            str(bool(diag.get("within_window", False))).lower(),
        )
        return False

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

    def estimate_task_joint_cost(self):
        if self.grasp_pose_array_received is None:
            return None
        pose_count = int(len(self.grasp_pose_array_received.poses))
        if pose_count <= 0:
            return None
        try:
            _pose_infos, seed_q = self.parse_grasp_info_payload(pose_count)
            if seed_q is None or len(seed_q) < 7:
                return None
            current_q = np.array(self.arm.move_group.get_current_joint_values()[:7], dtype=np.float64)
            target_q = np.array(seed_q[:7], dtype=np.float64)
            return float(np.max(np.abs(target_q - current_q)))
        except Exception:
            return None

    def joint_states_cb(self, msg):
        if msg is None or not getattr(msg, "name", None):
            return
        name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        joints = []
        try:
            for joint_i in range(1, 8):
                idx = name_to_idx.get(f"panda_joint{joint_i}")
                if idx is None:
                    return
                joints.append(float(msg.position[idx]))
        except Exception:
            return

        current = np.array(joints, dtype=np.float64)
        self.latest_arm_joint_state = current
        if not self._task_joint_tracking_active:
            return

        if self._task_joint_tracking_last is None:
            self._task_joint_tracking_last = current.copy()
            return

        delta = np.abs(current - self._task_joint_tracking_last)
        if np.any(delta > 1.5):
            # Ignore impossible jumps from stale/sim-time reset joint messages.
            self._task_joint_tracking_last = current.copy()
            return
        self._task_joint_path_cost += float(np.max(delta))
        self._task_joint_l1_path_cost += float(np.sum(delta))
        self._task_joint_tracking_last = current.copy()

    def start_task_joint_cost_tracking(self):
        self._task_joint_path_cost = 0.0
        self._task_joint_l1_path_cost = 0.0
        if self.latest_arm_joint_state is not None:
            self._task_joint_tracking_last = self.latest_arm_joint_state.copy()
        else:
            try:
                self._task_joint_tracking_last = np.array(
                    self.arm.move_group.get_current_joint_values()[:7],
                    dtype=np.float64,
                )
            except Exception:
                self._task_joint_tracking_last = None
        self._task_joint_tracking_active = True

    def stop_task_joint_cost_tracking(self):
        self._task_joint_tracking_active = False
        return float(self._task_joint_path_cost), float(self._task_joint_l1_path_cost)

    def publish_task_metrics(
        self,
        status,
        task_joint_cost,
        task_elapsed,
        estimated_seed_joint_cost=None,
        task_joint_l1_cost=None,
    ):
        payload = {
            "status": str(status),
            "task_time_sec": float(task_elapsed),
            "task_joint_cost_rad": None if task_joint_cost is None else float(task_joint_cost),
            "estimated_seed_joint_cost_rad": (
                None if estimated_seed_joint_cost is None else float(estimated_seed_joint_cost)
            ),
            "task_joint_l1_cost_rad": None if task_joint_l1_cost is None else float(task_joint_l1_cost),
            "failure_stage": self.last_failure_stage,
            "failure_reason": self.last_failure_reason,
        }
        try:
            self.task_metrics_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception:
            pass

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

    def get_current_joint_seed(self):
        try:
            current = [float(v) for v in self.arm.move_group.get_current_joint_values()[:7]]
        except Exception:
            return None
        return current if len(current) >= 7 else None

    def _append_unique_joint_seed(self, seeds, candidate):
        if candidate is None:
            return
        seed = [float(v) for v in candidate[:7]]
        if len(seed) < 7:
            return
        if any(np.allclose(np.array(seed), np.array(existing), atol=1e-4) for existing in seeds):
            return
        seeds.append(seed)

    def _build_place_ik_seeds(self, primary_seed=None):
        seeds = []
        self._append_unique_joint_seed(seeds, primary_seed)
        self._append_unique_joint_seed(seeds, self.last_place_ik_seed)
        self._append_unique_joint_seed(seeds, self.get_current_joint_seed())
        self._append_unique_joint_seed(seeds, self.last_observation_ik_seed)
        self._append_unique_joint_seed(seeds, self.observation_home_seed)
        if not seeds:
            return [None]
        return seeds[: self.place_ik_seed_trials]

    def build_orientation_path_constraints(self, target_orientation, tolerance_deg=None):
        tol_deg = (
            self.place_path_orientation_tolerance_deg
            if tolerance_deg is None
            else float(tolerance_deg)
        )
        tol_rad = float(np.deg2rad(max(0.1, tol_deg)))
        constraints = Constraints()
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.robot.get_planning_frame()
        eef_link = self.arm.move_group.get_end_effector_link() or getattr(self.arm, "eef_link", "") or "panda_hand"
        orientation_constraint.link_name = eef_link
        orientation_constraint.orientation = copy.deepcopy(target_orientation)
        orientation_constraint.absolute_x_axis_tolerance = tol_rad
        orientation_constraint.absolute_y_axis_tolerance = tol_rad
        orientation_constraint.absolute_z_axis_tolerance = tol_rad
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    def get_place_ik_collision_checked(self, pose, frame_id, primary_seed=None, timeout_sec=None):
        ik_timeout = float(timeout_sec) if timeout_sec is not None else float(self.place_ik_timeout_sec)
        for seed in self._build_place_ik_seeds(primary_seed=primary_seed):
            req = PositionIKRequest()
            req.group_name = "panda_manipulator"
            req.robot_state = self.robot.get_current_state()
            self._apply_seed_to_robot_state(req.robot_state, seed)
            req.pose_stamped = PoseStamped()
            req.pose_stamped.header.frame_id = frame_id
            req.pose_stamped.pose = pose
            req.timeout = rospy.Duration(ik_timeout)
            req.avoid_collisions = True
            try:
                res = self.compute_ik(req)
                if res.error_code.val == 1:
                    q = list(res.solution.joint_state.position[:7])
                    self.last_place_ik_seed = q
                    return q
            except Exception:
                continue
        return None

    def _plan_has_points(self, plan):
        return plan is not None and hasattr(plan, "joint_trajectory") and len(plan.joint_trajectory.points) > 0

    def _unpack_plan_result(self, plan_result):
        if isinstance(plan_result, tuple):
            success = bool(plan_result[0]) if len(plan_result) > 0 else False
            plan = plan_result[1] if len(plan_result) > 1 else None
        else:
            plan = plan_result
            success = self._plan_has_points(plan)
        if not success and self._plan_has_points(plan):
            success = True
        return success, plan

    def _get_plan_start_deviation(self, plan):
        if not self._plan_has_points(plan):
            return None
        current_seed = self.get_current_joint_seed()
        if current_seed is None:
            return None
        start_positions = np.array(plan.joint_trajectory.points[0].positions, dtype=np.float64)
        current_positions = np.array(current_seed[: len(start_positions)], dtype=np.float64)
        if start_positions.size == 0 or start_positions.shape != current_positions.shape:
            return None
        return float(np.max(np.abs(current_positions - start_positions)))

    def _ensure_strictly_increasing_time(self, plan, min_dt=1e-3):
        if not self._plan_has_points(plan):
            return plan
        adjusted = False
        prev_t = None
        for idx, point in enumerate(plan.joint_trajectory.points):
            t_sec = max(0.0, float(point.time_from_start.to_sec()))
            if idx == 0:
                if t_sec < 1e-6:
                    point.time_from_start = rospy.Duration.from_sec(0.0)
                prev_t = float(point.time_from_start.to_sec())
                continue
            min_allowed = float(prev_t) + float(min_dt)
            if t_sec <= min_allowed:
                point.time_from_start = rospy.Duration.from_sec(min_allowed)
                adjusted = True
                prev_t = min_allowed
            else:
                prev_t = t_sec
        if adjusted:
            rospy.logwarn("⚠️ 检测到非严格递增的轨迹时间戳，已在本地修正后再执行。")
        return plan

    def retime_plan_from_current_state(self, plan, velocity_scaling=None, acceleration_scaling=None):
        if not self._plan_has_points(plan):
            return plan
        vel = self.place_fallback_velocity_scaling if velocity_scaling is None else float(velocity_scaling)
        acc = self.place_fallback_acceleration_scaling if acceleration_scaling is None else float(acceleration_scaling)
        try:
            retimed = self.arm.move_group.retime_trajectory(
                self.robot.get_current_state(),
                plan,
                velocity_scaling_factor=max(0.01, min(1.0, vel)),
                acceleration_scaling_factor=max(0.01, min(1.0, acc)),
            )
            if self._plan_has_points(retimed):
                plan = retimed
        except Exception as e:
            rospy.logwarn(f"⚠️ 轨迹重定时失败，继续尝试原轨迹: {e}")
        return self._ensure_strictly_increasing_time(plan)

    def execute_planned_trajectory(self, plan, description, start_tolerance=None, retime=False):
        if not self._plan_has_points(plan):
            return False

        max_start_dev = (
            self.plan_execution_start_tolerance
            if start_tolerance is None
            else float(start_tolerance)
        )
        plan_to_execute = copy.deepcopy(plan)
        if retime:
            plan_to_execute = self.retime_plan_from_current_state(plan_to_execute)
        else:
            plan_to_execute = self._ensure_strictly_increasing_time(plan_to_execute)

        start_dev = self._get_plan_start_deviation(plan_to_execute)
        if start_dev is not None and start_dev > max_start_dev:
            rospy.logwarn(
                f"⚠️ {description} 轨迹起点与当前关节偏差过大 "
                f"({start_dev:.4f} rad > {max_start_dev:.4f} rad)，放弃执行并等待重规划。"
            )
            return False

        success = self.arm.move_group.execute(plan_to_execute, wait=True)
        self.arm.move_group.stop()
        if success:
            return True

        current_joints = np.array(
            self.arm.move_group.get_current_joint_values()[: len(plan_to_execute.joint_trajectory.points[-1].positions)],
            dtype=np.float64,
        )
        target_joints = np.array(plan_to_execute.joint_trajectory.points[-1].positions, dtype=np.float64)
        max_joint_error = float(np.max(np.abs(current_joints - target_joints)))
        if max_joint_error <= 0.02:
            rospy.logwarn(
                f"⚠️ {description} 执行返回失败，但末端关节已足够接近目标 "
                f"(max_joint_error={max_joint_error:.4f} rad)，按成功处理。"
            )
            return True
        return False

    def align_current_pose_orientation(self, target_orientation, description, retries=2):
        current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose = copy.deepcopy(current_pose)
        target_pose.orientation = copy.deepcopy(target_orientation)
        return self.execute_pose_goal(
            target_pose,
            description,
            retries=retries,
            orientation_path_lock=copy.deepcopy(target_orientation),
            preferred_seed=self.get_current_joint_seed(),
        )

    def add_basket_collision_geometry(self):
        bx, by, bz = self.basket_x, self.basket_y, self.basket_z
        sx, sy, sz = self.basket_size
        t = self.basket_wall_thickness

        self.remove_basket_collision_geometry(log=False)
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

    def get_basket_collision_object_names(self):
        return [
            "basket_base",
            "basket_wall_front",
            "basket_wall_back",
            "basket_wall_left",
            "basket_wall_right",
        ]

    def remove_basket_collision_geometry(self, log=True):
        for name in self.get_basket_collision_object_names():
            self.scene.remove_world_object(name)
        for name in self.get_basket_collision_object_names():
            self.wait_for_scene_sync(name, object_is_known=False, object_is_attached=False, timeout=0.5)
        if log:
            rospy.loginfo(
                "🧺 已从 MoveIt 碰撞场景移除篮子几何，仅保留篮子位姿用于放置目标。"
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

    def is_pose_inside_basket_xy(self, pose, margin_xy=None):
        margin = self.fast_drop_inner_margin_xy if margin_xy is None else float(margin_xy)
        half_x = 0.5 * float(self.basket_size[0]) - margin
        half_y = 0.5 * float(self.basket_size[1]) - margin
        if half_x <= 0.0 or half_y <= 0.0:
            return False
        dx = float(pose.position.x) - float(self.basket_x)
        dy = float(pose.position.y) - float(self.basket_y)
        return abs(dx) <= half_x and abs(dy) <= half_y

    def try_fast_drop_if_ready(self, pose, basket_top_z, stage_label):
        if not self.fast_drop_when_inside_basket:
            return False, True

        inside_xy = self.is_pose_inside_basket_xy(pose)
        min_allowed_z = float(basket_top_z + self.fast_drop_min_height_above_top)
        current_z = float(pose.position.z)
        if inside_xy and current_z >= min_allowed_z:
            rospy.loginfo(
                "⚡ [%s] 当前末端已在篮子范围上方且高度满足阈值，执行快速放置：直接完全张开夹爪释放。",
                stage_label,
            )
            if not self.release_grasped_item():
                return True, False
            if self.refresh_collision_after_each_place:
                self.refresh_shelf_collision_space()
            self.arm.move_group.stop()
            self.arm.move_group.clear_pose_targets()
            return True, True

        if inside_xy:
            rospy.logwarn(
                "⚠️ [%s] 当前已进入篮子 XY 范围，但高度不足（z=%.3f < %.3f），转常规放置流程。",
                stage_label,
                current_z,
                min_allowed_z,
            )
            return False, True

        half_x = 0.5 * float(self.basket_size[0]) - self.fast_drop_inner_margin_xy
        half_y = 0.5 * float(self.basket_size[1]) - self.fast_drop_inner_margin_xy
        rospy.loginfo(
            "ℹ️ [%s] 快速放置未触发：末端尚未进入篮子范围。"
            "current_xy=(%.3f,%.3f), basket_center=(%.3f,%.3f), inner_half=(%.3f,%.3f)",
            stage_label,
            float(pose.position.x),
            float(pose.position.y),
            float(self.basket_x),
            float(self.basket_y),
            float(half_x),
            float(half_y),
        )
        return False, True

    def release_grasped_item(self):
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
        self.clear_functional_grasp_candidate()
        self.clear_active_grasp_target()
        self.drop_counter += 1
        self.current_item_collision_size = None
        self.save_task_keyframe("placed")
        return True

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

    def wrist_grasp_pose_callback(self, msg):
        self.wrist_grasp_pose_array_received = msg

    def wrist_grasp_info_callback(self, msg):
        self.wrist_grasp_infos = msg.data

    def parse_grasp_info_payload(self, num_poses):
        """
        解析 grasp_info：
        1) 旧格式：单姿态 [width, score, depth, seed_q(7)]
        2) 新格式：多姿态 [pose1(w,s,d), pose2(w,s,d), ... , seed_q(7)]
        3) 兼容仅多姿态 [pose1(w,s,d), pose2(w,s,d), ...]
        """
        n = max(0, int(num_poses))
        data = list(self.grasp_infos) if self.grasp_infos is not None else []
        per_pose_infos = [[0.04, 0.0, 0.03] for _ in range(n)]
        seed_q = None

        if n <= 0:
            return per_pose_infos, seed_q

        # 新格式（优先）：3*n + 7
        if len(data) >= (3 * n + 7):
            for i in range(n):
                base = 3 * i
                per_pose_infos[i] = [
                    float(data[base]),
                    float(data[base + 1]),
                    float(data[base + 2]),
                ]
            seed_q = [float(v) for v in data[3 * n : 3 * n + 7]]
            return per_pose_infos, seed_q

        # 兼容：仅 3*n
        if len(data) >= (3 * n):
            for i in range(n):
                base = 3 * i
                per_pose_infos[i] = [
                    float(data[base]),
                    float(data[base + 1]),
                    float(data[base + 2]),
                ]
            return per_pose_infos, seed_q

        # 旧单姿态格式兼容
        if n == 1 and len(data) >= 3:
            per_pose_infos[0] = [float(data[0]), float(data[1]), float(data[2])]
            if len(data) >= 10:
                seed_q = [float(v) for v in data[3:10]]
            return per_pose_infos, seed_q

        return per_pose_infos, seed_q

    def parse_wrist_grasp_info_payload(self, num_poses):
        n = max(0, int(num_poses))
        data = list(self.wrist_grasp_infos) if self.wrist_grasp_infos is not None else []
        per_pose_infos = [[0.04, 0.0, 0.03] for _ in range(n)]
        if n <= 0 or len(data) == 0:
            return per_pose_infos

        stride = int(len(data) / n) if len(data) % n == 0 else 0
        if stride >= 4:
            for i in range(n):
                base = i * stride
                # grasp_from_sam raw format: [object_id, width, score, depth]
                per_pose_infos[i] = [
                    float(data[base + 1]),
                    float(data[base + 2]),
                    float(data[base + 3]),
                ]
            return per_pose_infos
        if stride >= 3:
            for i in range(n):
                base = i * stride
                per_pose_infos[i] = [
                    float(data[base]),
                    float(data[base + 1]),
                    float(data[base + 2]),
                ]
        return per_pose_infos

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

    def wrist_info_cb(self, msg):
        try:
            self.wrist_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        except Exception:
            self.wrist_K = None

    def quaternion_distance_deg(self, qa, qb):
        qa = np.array(qa, dtype=np.float64)
        qb = np.array(qb, dtype=np.float64)
        if np.linalg.norm(qa) < 1e-9 or np.linalg.norm(qb) < 1e-9:
            return 180.0
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(dot)))

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
        margin = self.grasp_width_margin
        if predicted_width >= self.grasp_width_margin_wide_threshold:
            margin = max(self.grasp_width_margin, self.grasp_width_margin_wide)
        margin += max(0.0, predicted_width) * max(0.0, self.grasp_width_margin_ratio)
        self.last_grasp_width_margin_used = float(margin)
        lightly_shrunk = max(0.0, predicted_width - self.last_grasp_width_margin_used)
        command_width = float(
            np.clip(
                lightly_shrunk,
                self.grasp_command_width_min,
                self.grasp_command_width_max,
            )
        )
        return command_width, lightly_shrunk

    def compute_grasp_probe_distance(self, executed_insert_distance):
        executed_insert_distance = float(executed_insert_distance)
        if executed_insert_distance <= 1e-6:
            return 0.0
        probe_distance = min(max(0.0, self.grasp_probe_retreat_distance), executed_insert_distance)
        if executed_insert_distance - probe_distance < 0.010:
            probe_distance = max(0.0, executed_insert_distance - 0.010)
        return float(max(0.0, probe_distance))

    def compute_resqueeze_width(self, current_command_width):
        base_width = float(current_command_width)
        observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
        if observed_width is not None:
            base_width = min(base_width, float(observed_width))
        target_width = base_width - max(0.0, self.grasp_resqueeze_extra_margin)
        return float(np.clip(target_width, self.grasp_command_width_min, self.grasp_command_width_max))

    def parse_grasp_axis_param(self, value, default_axis="z", default_sign=1.0):
        text = str(value).strip().lower()
        sign = float(default_sign)
        if text.startswith("-"):
            sign = -1.0
            text = text[1:]
        elif text.startswith("+"):
            sign = 1.0
            text = text[1:]

        aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "0": "x",
            "1": "y",
            "2": "z",
        }
        axis = aliases.get(text)
        if axis is None:
            axis = str(default_axis).strip().lower()
            if axis not in ("x", "y", "z"):
                axis = "z"
            sign = float(default_sign)
            rospy.logwarn(
                "⚠️ 未知抓取姿态轴参数 %r，回退到 %s%s。",
                value,
                "+" if sign >= 0.0 else "-",
                axis,
            )
        label = ("%s%s" % ("+" if sign >= 0.0 else "-", axis.upper()))
        return axis, sign, label

    def get_grasp_pose_axis_vector(self, rot_mat, axis_name, axis_sign=1.0):
        axis_index = {"x": 0, "y": 1, "z": 2}.get(str(axis_name).lower())
        if axis_index is None:
            return None
        vec = float(axis_sign) * np.array(rot_mat[:3, axis_index], dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    def get_configured_grasp_pose_axes(self, rot_mat):
        approach_axis = self.get_grasp_pose_axis_vector(
            rot_mat,
            self.grasp_pose_approach_axis,
            self.grasp_pose_approach_sign,
        )
        open_axis = self.get_grasp_pose_axis_vector(
            rot_mat,
            self.grasp_pose_open_axis,
            self.grasp_pose_open_sign,
        )
        return approach_axis, open_axis

    def build_grasp_quaternion_from_approach_open_axis(self, robot_z, gripper_open_axis):
        robot_z = np.array(robot_z, dtype=np.float64)
        z_norm = np.linalg.norm(robot_z)
        if z_norm < 1e-6:
            return None
        robot_z = robot_z / z_norm

        robot_y = np.array(gripper_open_axis, dtype=np.float64)
        robot_y = robot_y - np.dot(robot_y, robot_z) * robot_z
        y_norm = np.linalg.norm(robot_y)
        if y_norm < 1e-6:
            return None
        robot_y = robot_y / y_norm

        robot_x = np.cross(robot_y, robot_z)
        x_norm = np.linalg.norm(robot_x)
        if x_norm < 1e-6:
            return None
        robot_x = robot_x / x_norm
        robot_y = np.cross(robot_z, robot_x)

        new_rot_mat = np.eye(4)
        new_rot_mat[:3, 0] = robot_x / np.linalg.norm(robot_x)
        new_rot_mat[:3, 1] = robot_y / np.linalg.norm(robot_y)
        new_rot_mat[:3, 2] = robot_z
        return tft.quaternion_from_matrix(new_rot_mat)

    def tcp_z_axis_from_quaternion(self, q_candidate):
        rot_mat = tft.quaternion_matrix(q_candidate)
        tcp_z = np.array(rot_mat[:3, 2], dtype=np.float64)
        norm = np.linalg.norm(tcp_z)
        if norm < 1e-6:
            return None
        return tcp_z / norm

    def tcp_z_axis_matches_insert_axis(self, q_candidate, robot_z, label=""):
        desired = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(desired)
        if norm < 1e-6:
            return False
        desired = desired / norm
        tcp_z = self.tcp_z_axis_from_quaternion(q_candidate)
        if tcp_z is None:
            return False
        alignment = float(np.dot(tcp_z, desired))
        allowed = alignment >= float(self.observation_camera_axis_min_dot)
        if not allowed:
            rospy.logdebug(
                "reject orientation %s: TCP +Z/camera axis alignment %.3f < %.3f",
                str(label),
                alignment,
                self.observation_camera_axis_min_dot,
            )
        return allowed

    def gripper_open_axis_is_allowed(self, q_candidate, label=""):
        if not self.enforce_horizontal_gripper_open_axis:
            return True
        rot_mat = tft.quaternion_matrix(q_candidate)
        open_axis = np.array(rot_mat[:3, 1], dtype=np.float64)
        norm = np.linalg.norm(open_axis)
        if norm < 1e-6:
            return False
        open_axis = open_axis / norm
        vertical_component = abs(float(open_axis[2]))
        allowed = vertical_component <= float(self.max_gripper_open_axis_vertical_component)
        if not allowed:
            rospy.logdebug(
                "reject grasp orientation %s: gripper open axis vertical component %.3f > %.3f",
                str(label),
                vertical_component,
                self.max_gripper_open_axis_vertical_component,
            )
        return allowed

    def build_grasp_orientation_options(self, robot_z, raw_open_axis, q_orig, include_raw=False):
        options = []
        if self.use_grasp_pose_open_axis_orientation and raw_open_axis is not None:
            q_axis = self.build_grasp_quaternion_from_approach_open_axis(robot_z, raw_open_axis)
            if q_axis is not None and self.gripper_open_axis_is_allowed(q_axis, "graspnet_axes"):
                options.append(("graspnet_axes", q_axis))
        if (
            include_raw
            and self.allow_raw_grasp_pose_orientation
            and self.tcp_z_axis_matches_insert_axis(q_orig, robot_z, "raw_grasp_pose")
            and self.gripper_open_axis_is_allowed(q_orig, "raw_grasp_pose")
        ):
            options.append(("raw_grasp_pose", q_orig))
        for roll_variant in self.grasp_roll_variants:
            q_level = self.build_level_grasp_quaternion(robot_z, roll_variant=roll_variant)
            if q_level is not None and self.gripper_open_axis_is_allowed(q_level, roll_variant):
                options.append((str(roll_variant), q_level))
        return options

    def build_level_grasp_quaternion(self, robot_z, roll_variant="y_down"):
        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm < 1e-6:
            return None
        robot_z = robot_z / norm
        world_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        variant = str(roll_variant).lower()

        if variant == "x_down":
            robot_x = world_down
            robot_y = np.cross(robot_z, robot_x)
            if np.linalg.norm(robot_y) < 1e-6:
                return None
            robot_y = robot_y / np.linalg.norm(robot_y)
            robot_x = np.cross(robot_y, robot_z)
        elif variant == "x_up":
            robot_x = -world_down
            robot_y = np.cross(robot_z, robot_x)
            if np.linalg.norm(robot_y) < 1e-6:
                return None
            robot_y = robot_y / np.linalg.norm(robot_y)
            robot_x = np.cross(robot_y, robot_z)
        elif variant in ("y_left", "y_pos", "open_y_pos", "open_left"):
            robot_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            robot_y = robot_y - np.dot(robot_y, robot_z) * robot_z
            if np.linalg.norm(robot_y) < 1e-6:
                return None
            robot_y = robot_y / np.linalg.norm(robot_y)
            robot_x = np.cross(robot_y, robot_z)
        elif variant in ("y_right", "y_neg", "open_y_neg", "open_right"):
            robot_y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            robot_y = robot_y - np.dot(robot_y, robot_z) * robot_z
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

    def get_gazebo_link_world_pose(self, model_name, link_name=None):
        if self.get_link_state is None or not model_name:
            return None
        link = link_name or self.get_gazebo_model_link_name(model_name)
        scoped_link = link if "::" in str(link) else f"{model_name}::{link}"
        try:
            resp = self.get_link_state(scoped_link, "world")
            if getattr(resp, "success", False):
                return resp.link_state.pose
        except Exception as e:
            rospy.logdebug("读取 Gazebo link 位姿失败: model=%s, link=%s, err=%s", model_name, scoped_link, str(e))
        return None

    def correct_target_height_from_gazebo(self, target_pose, label=""):
        if (
            not self.use_gazebo_grasp_height_correction
            or self.gazebo_model_states is None
            or self.get_link_state is None
        ):
            return target_pose

        target = np.array(
            [target_pose.position.x, target_pose.position.y, target_pose.position.z],
            dtype=np.float64,
        )
        best_name = None
        best_xy_dist = None
        best_pose = None
        for name, model_pose in zip(self.gazebo_model_states.name, self.gazebo_model_states.pose):
            if not self.is_graspable_gazebo_model(name):
                continue
            model_xy = np.array([model_pose.position.x, model_pose.position.y], dtype=np.float64)
            xy_dist = float(np.linalg.norm(model_xy - target[:2]))
            if xy_dist > float(self.gazebo_grasp_height_correction_xy_threshold):
                continue
            if best_xy_dist is None or xy_dist < best_xy_dist:
                best_name = name
                best_xy_dist = xy_dist
                best_pose = model_pose

        if best_name is None:
            return target_pose

        link_name = self.get_gazebo_model_link_name(best_name)
        link_pose = self.get_gazebo_link_world_pose(best_name, link_name)
        if link_pose is None:
            return target_pose

        gz_center_z = float(link_pose.position.z) + float(self.gazebo_grasp_height_correction_z_bias)
        vision_z = float(target_pose.position.z)
        dz = gz_center_z - vision_z
        if abs(dz) < 1e-4:
            return target_pose

        # 非对称 delta 上限：向下修正 (dz<0) 是安全的（进入工作空间），允许较大修正；
        # 向上修正 (dz>0) 风险高（可能超出工作空间），使用较紧的上限。
        if dz > 0:
            max_allowed = float(self.gazebo_grasp_height_correction_max_delta)
        else:
            max_allowed = float(rospy.get_param(
                "~gazebo_grasp_height_correction_max_delta_down",
                self.gazebo_grasp_height_correction_max_delta * 2.5,
            ))
        if abs(dz) > max_allowed:
            rospy.logwarn(
                "⚠️ Gazebo目标高度修正被拒绝: model=%s, raw_z=%.3f, gz_center_z=%.3f, dz=%+.3f 超过上限 %.3f (方向=%s)",
                best_name,
                vision_z,
                gz_center_z,
                dz,
                max_allowed,
                "up" if dz > 0 else "down",
            )
            return target_pose

        # 向下修正也加一个最低安全帽，避免修正到桌面以下
        gazebo_min_z = float(rospy.get_param("~gazebo_height_correction_min_z", 0.15))
        blend_ratio = float(rospy.get_param("~gazebo_height_correction_blend_ratio", 0.5))
        blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))
        if blend_ratio >= 1.0:
            corrected_z = gz_center_z
        elif blend_ratio <= 0.0:
            corrected_z = vision_z
        else:
            corrected_z = vision_z * (1.0 - blend_ratio) + gz_center_z * blend_ratio

        # Workspace 安全帽：修正后 Z 必须在合理范围内
        gazebo_max_z = float(rospy.get_param("~gazebo_height_correction_max_z", 0.78))
        if corrected_z > gazebo_max_z:
            rospy.logwarn(
                "⚠️ Gazebo高度修正后 z=%.3f 超过 workspace 上限 %.3f, 回退到视觉估计 z=%.3f",
                corrected_z,
                gazebo_max_z,
                vision_z,
            )
            return target_pose
        if corrected_z < gazebo_min_z:
            rospy.logwarn(
                "⚠️ Gazebo高度修正后 z=%.3f 低于安全下限 %.3f, 回退到视觉估计 z=%.3f",
                corrected_z,
                gazebo_min_z,
                vision_z,
            )
            return target_pose

        corrected_pose = copy.deepcopy(target_pose)
        corrected_pose.position.z = corrected_z
        rospy.loginfo(
            "📐 Gazebo目标高度修正%s: model=%s, link=%s, xy_dist=%.3f, z %.3f -> %.3f (dz=%+.3f, blend=%.2f)",
            f"({label})" if label else "",
            best_name,
            link_name,
            float(best_xy_dist if best_xy_dist is not None else 0.0),
            vision_z,
            corrected_z,
            dz,
            blend_ratio,
        )
        return corrected_pose

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

    def execute_pose_goal(
        self,
        target_pose,
        description,
        retries=2,
        orientation_path_lock=None,
        preferred_seed=None,
    ):
        for attempt in range(1, retries + 1):
            if self.stop_requested:
                return False

            constraints = (
                self.build_orientation_path_constraints(orientation_path_lock)
                if orientation_path_lock is not None
                else None
            )

            success = False
            try:
                rospy.sleep(0.12)
                self.arm.move_group.set_start_state_to_current_state()
                self.arm.move_group.clear_pose_targets()
                if constraints is not None:
                    self.arm.move_group.set_path_constraints(constraints)

                planning_frame = self.robot.get_planning_frame()
                joint_target = self.get_place_ik_collision_checked(
                    target_pose,
                    planning_frame,
                    primary_seed=preferred_seed,
                )
                if joint_target is not None:
                    self.arm.move_group.set_joint_value_target(joint_target)
                    rospy.loginfo(
                        f"{description} [seeded IK{' + path lock' if constraints is not None else ''}] "
                        f"(第 {attempt}/{retries} 次尝试)"
                    )
                else:
                    self.arm.move_group.set_pose_target(target_pose)
                    rospy.loginfo(
                        f"{description} [{'path lock' if constraints is not None else 'pose target'}] "
                        f"(第 {attempt}/{retries} 次尝试)"
                    )

                plan_result = self.arm.move_group.plan()
                plan_success, plan = self._unpack_plan_result(plan_result)
                if plan_success:
                    success = self.execute_planned_trajectory(
                        plan,
                        description,
                        retime=True,
                    )
                else:
                    rospy.logwarn(f"⚠️ {description} 未生成有效规划轨迹。")
            finally:
                self.arm.move_group.stop()
                self.arm.move_group.clear_pose_targets()
                self.arm.move_group.clear_path_constraints()

            if success:
                current_seed = self.get_current_joint_seed()
                if current_seed is not None:
                    self.last_place_ik_seed = current_seed
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
            if self.execute_pose_goal(
                target_pose,
                f"{description} [保姿态优先]",
                retries=retries,
                orientation_path_lock=copy.deepcopy(current_pose.orientation),
                preferred_seed=self.get_current_joint_seed(),
            ):
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
        if self.execute_pose_goal(
            target_pose,
            f"{description} [{pose_label}]",
            retries=2,
            orientation_path_lock=copy.deepcopy(target_orientation) if orientation_locked else None,
            preferred_seed=self.get_current_joint_seed(),
        ):
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
        if self.execute_pose_goal(
            target_pose,
            f"{description} [{pose_label}]",
            retries=2,
            orientation_path_lock=copy.deepcopy(target_orientation) if orientation_locked else None,
            preferred_seed=self.get_current_joint_seed(),
        ):
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

    def _apply_seed_to_robot_state(self, robot_state, seed_q):
        if seed_q is None:
            return
        seed = [float(v) for v in seed_q[:7]]
        if len(seed) < 7:
            return
        joint_state = robot_state.joint_state
        if joint_state is None or not getattr(joint_state, "name", None):
            return
        name_to_idx = {name: idx for idx, name in enumerate(joint_state.name)}
        positions = list(joint_state.position)
        for joint_i in range(7):
            joint_name = f"panda_joint{joint_i + 1}"
            idx = name_to_idx.get(joint_name)
            if idx is None or idx >= len(positions):
                continue
            positions[idx] = seed[joint_i]
        joint_state.position = positions

    def _build_observation_ik_seeds(self, primary_seed=None):
        seeds = []
        for candidate in [primary_seed, self.last_observation_ik_seed]:
            if candidate is None:
                continue
            c = [float(v) for v in candidate[:7]]
            if len(c) < 7:
                continue
            if not any(np.allclose(np.array(c), np.array(s), atol=1e-4) for s in seeds):
                seeds.append(c)
        try:
            current_seed = [float(v) for v in self.arm.move_group.get_current_joint_values()[:7]]
            if len(current_seed) >= 7 and not any(np.allclose(np.array(current_seed), np.array(s), atol=1e-4) for s in seeds):
                seeds.append(current_seed)
        except Exception:
            pass
        if not any(np.allclose(np.array(self.observation_home_seed), np.array(s), atol=1e-4) for s in seeds):
            seeds.append(list(self.observation_home_seed))
        if not seeds:
            return [None]
        return seeds[: self.observation_ik_seed_trials]

    def get_ik_seeded(self, pose, seed_q, frame_id):
        """以 GTSP 预计算的关节角为首选种子求 IK，提高观察位可达率。"""
        for seed in self._build_observation_ik_seeds(primary_seed=seed_q):
            req = PositionIKRequest()
            req.group_name = "panda_manipulator"
            req.robot_state = self.robot.get_current_state()
            self._apply_seed_to_robot_state(req.robot_state, seed)
            req.pose_stamped = PoseStamped()
            req.pose_stamped.header.frame_id = frame_id
            req.pose_stamped.pose = pose
            req.timeout = rospy.Duration(float(self.observation_ik_timeout_sec))
            req.avoid_collisions = bool(self.observation_ik_avoid_collisions)
            try:
                res = self.compute_ik(req)
                if res.error_code.val == 1:
                    q = list(res.solution.joint_state.position[:7])
                    self.last_observation_ik_seed = q
                    return q
            except Exception:
                continue
        return None

    def get_ik_collision_checked(self, pose, frame_id, timeout_sec=0.35):
        """无种子 IK，带碰撞检测，避免 set_pose_target 长时间超时。"""
        ik_timeout = float(timeout_sec) if timeout_sec is not None else float(self.observation_ik_timeout_sec)
        for seed in self._build_observation_ik_seeds(primary_seed=None):
            req = PositionIKRequest()
            req.group_name = "panda_manipulator"
            req.robot_state = self.robot.get_current_state()
            self._apply_seed_to_robot_state(req.robot_state, seed)
            req.pose_stamped = PoseStamped()
            req.pose_stamped.header.frame_id = frame_id
            req.pose_stamped.pose = pose
            req.timeout = rospy.Duration(ik_timeout)
            req.avoid_collisions = bool(self.observation_ik_avoid_collisions)
            try:
                res = self.compute_ik(req)
                if res.error_code.val == 1:
                    q = list(res.solution.joint_state.position[:7])
                    self.last_observation_ik_seed = q
                    return q
            except Exception:
                continue
        return None

    def apply_quaternion_to_pose(self, pose, q):
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])

    def pose_quaternion_list(self, pose):
        return [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ]

    def plan_observation_pose(self, observation_pose, seed_q, planning_frame):
        observation_q = self.get_ik_seeded(observation_pose, seed_q, planning_frame) if seed_q else None
        if observation_q is None:
            observation_q = self.get_ik_collision_checked(observation_pose, planning_frame)
        if observation_q is None:
            return None, None, "ik_failed"
        if not self.joint_state_is_valid(observation_q, label="观察位"):
            return observation_q, None, "state_collision"

        try:
            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.set_joint_value_target(observation_q)
            plan_result = self.arm.move_group.plan()
            success, plan = self._unpack_plan_result(plan_result)
        finally:
            self.arm.move_group.clear_pose_targets()
        if not success:
            return observation_q, None, "planning_failed"
        return observation_q, plan, "success"

    def plan_observation_pose_with_relaxed_orientation(
        self,
        observation_pose,
        final_orientation_label,
        robot_z,
        raw_open_axis,
        q_orig,
        seed_q,
        planning_frame,
    ):
        _, plan, status = self.plan_observation_pose(observation_pose, seed_q, planning_frame)
        if plan is not None:
            return copy.deepcopy(observation_pose), plan, str(final_orientation_label), status

        if not self.allow_relaxed_observation_orientation:
            return None, None, str(final_orientation_label), status

        final_q = self.pose_quaternion_list(observation_pose)
        observation_options = self.build_grasp_orientation_options(
            robot_z,
            raw_open_axis,
            q_orig,
            include_raw=True,
        )
        for observation_label, q_observe in observation_options:
            if self.quaternion_distance_deg(final_q, q_observe) < 1.0:
                continue
            if not self.tcp_z_axis_matches_insert_axis(q_observe, robot_z, observation_label):
                continue
            relaxed_pose = copy.deepcopy(observation_pose)
            self.apply_quaternion_to_pose(relaxed_pose, q_observe)
            _, relaxed_plan, relaxed_status = self.plan_observation_pose(relaxed_pose, seed_q, planning_frame)
            if relaxed_plan is not None:
                label = f"{observation_label}->final:{final_orientation_label}"
                return relaxed_pose, relaxed_plan, label, "success"
            status = relaxed_status

        return None, None, str(final_orientation_label), status

    def build_lift_retained_grasp_poses(self, target_pose, pre_grasp_pose, obs_lift):
        stored_target_pose = copy.deepcopy(target_pose)
        stored_pre_grasp_pose = copy.deepcopy(pre_grasp_pose)
        retained_lift = 0.0
        if self.pregrasp_retain_observation_lift:
            retained_lift = float(np.clip(float(obs_lift), 0.0, max(0.0, self.pregrasp_retained_lift_max)))
            if retained_lift > 1e-4:
                stored_target_pose.position.z += retained_lift
                stored_pre_grasp_pose.position.z += retained_lift
        return stored_target_pose, stored_pre_grasp_pose, retained_lift

    def normalize_current_joint_state_if_needed(self):
        """当当前关节稍微越界时，先拉回软限位，减少后续规划的 start-state 异常。"""
        try:
            current = np.array(self.arm.move_group.get_current_joint_values()[:7], dtype=np.float64)
        except Exception:
            return
        if current.shape[0] < 7:
            return

        lower = self.panda_joint_lower + self.joint_limit_margin
        upper = self.panda_joint_upper - self.joint_limit_margin
        clamped = np.clip(current, lower, upper)
        max_delta = float(np.max(np.abs(clamped - current)))
        if max_delta < 1e-6:
            return

        rospy.logwarn(
            f"⚠️ 当前关节接近/超出软限位，先执行小幅回正 (max_delta={max_delta:.5f} rad)。"
        )
        self.arm.move_group.set_start_state_to_current_state()
        self.arm.move_group.set_joint_value_target(clamped.tolist())
        ok = self.arm.move_group.go(wait=True)
        self.arm.move_group.stop()
        self.arm.move_group.clear_pose_targets()
        if ok:
            rospy.loginfo("✅ 关节状态已回正到软限位内。")
        else:
            rospy.logwarn("⚠️ 关节状态回正执行失败，继续尝试原流程。")

    def log_workspace_diagnostics(self, pose_infos, planning_frame):
        """当所有候选观察位都不可达时，打印详细的 workspace 诊断信息。"""
        try:
            current_joints = self.arm.move_group.get_current_joint_values()[:7]
        except Exception:
            current_joints = None
        base_link = self.robot.get_planning_frame()

        rospy.logwarn("=" * 60)
        rospy.logwarn("🔍 Workspace 诊断 —— 候选位姿与机器人当前状态")
        if current_joints is not None:
            rospy.logwarn(
                "  当前关节 (rad): [%s]",
                ", ".join(f"{v:.3f}" for v in current_joints),
            )
            j4 = float(current_joints[3])
            rospy.logwarn(
                "  joint4=%.3f rad (%.1f°), 上限=-0.070 rad, 距离上限=%.3f rad",
                j4,
                float(np.degrees(j4)),
                float(self.panda_joint_upper[3] - j4),
            )
            try:
                ee_pose = self.arm.move_group.get_current_pose().pose
                ee_x, ee_y, ee_z = ee_pose.position.x, ee_pose.position.y, ee_pose.position.z
                rospy.logwarn("  当前末端位姿: (%.3f, %.3f, %.3f) in %s", ee_x, ee_y, ee_z, base_link)
            except Exception:
                pass

        grasp_header = self.grasp_pose_array_received.header if self.grasp_pose_array_received is not None else None
        source_frame = grasp_header.frame_id if grasp_header is not None else "unknown"
        rospy.logwarn("  候选位姿 source frame: %s", source_frame)
        rospy.logwarn("  候选数量: %d", len(pose_infos) if pose_infos else 0)

        shelf_pose = self.get_shelf_world_pose()
        rospy.logwarn(
            "  货架位姿: (%.3f, %.3f, %.3f) yaw=%.3f",
            shelf_pose["x"],
            shelf_pose["y"],
            shelf_pose["z"],
            shelf_pose["yaw"],
        )

        # --- 点云 vs Gazebo 偏差诊断 ---
        for i, raw_pose in enumerate(self.grasp_pose_array_received.poses[:3]):
            ps = PoseStamped()
            ps.header = grasp_header
            ps.pose = raw_pose
            target_world = self.transform_pose(ps, target_frame=planning_frame)
            if target_world is None:
                continue
            tp = target_world.pose.position

            # 找最近的 Gazebo 可抓取模型
            nearest_name, nearest_dist, nearest_pos = None, float("inf"), None
            if self.gazebo_model_states is not None:
                for model_state in self.gazebo_model_states:
                    if not self.is_graspable_gazebo_model(model_state.name):
                        continue
                    gz_pos = model_state.pose.position
                    dist = float(np.linalg.norm(
                        np.array([tp.x, tp.y, tp.z])
                        - np.array([gz_pos.x, gz_pos.y, gz_pos.z])
                    ))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_name = model_state.name
                        nearest_pos = (gz_pos.x, gz_pos.y, gz_pos.z)

            if nearest_name is not None:
                rospy.logwarn(
                    "  候选 %d (world): pos=(%.3f,%.3f,%.3f) | "
                    "最近Gazebo: %s pos=(%.3f,%.3f,%.3f) dist=%.3f m",
                    i + 1,
                    tp.x, tp.y, tp.z,
                    nearest_name,
                    nearest_pos[0], nearest_pos[1], nearest_pos[2],
                    nearest_dist,
                )
            else:
                rospy.logwarn(
                    "  候选 %d (world): pos=(%.3f, %.3f, %.3f) | 无可抓取Gazebo模型",
                    i + 1, tp.x, tp.y, tp.z,
                )
            clearance = self.compute_shelf_cavity_clearance(target_world.pose)
            if clearance is not None:
                rospy.logwarn("    货架腔体余量: %.3f m", clearance)
            else:
                rospy.logwarn("    ⚠️ 该位姿不在货架腔体范围内！")

        rospy.logwarn("=" * 60)

    def joint_state_is_valid(self, joint_values, label="候选位", group_name=None):
        """用 MoveIt 状态有效性服务提前剔除整臂碰撞的 IK 解。"""
        if (
            not self.enable_goal_state_validity_check
            or self.check_state_validity is None
            or joint_values is None
        ):
            return True

        try:
            robot_state = self.robot.get_current_state()
            self._apply_seed_to_robot_state(robot_state, joint_values)
            group = group_name or getattr(self.arm, "group_name", "panda_manipulator")
            res = self.check_state_validity(robot_state, group, Constraints())
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "⚠️ MoveIt 状态有效性检查调用失败，暂不拦截候选: %s",
                str(exc),
            )
            return True

        if getattr(res, "valid", False):
            return True

        contacts = []
        for contact in list(getattr(res, "contacts", []))[:3]:
            body_1 = getattr(contact, "contact_body_1", "")
            body_2 = getattr(contact, "contact_body_2", "")
            if body_1 or body_2:
                contacts.append(f"{body_1}<->{body_2}")
        suffix = f": {', '.join(contacts)}" if contacts else "（无接触详情）"
        rospy.logwarn_throttle(
            2.0,
            "⚠️ %s 的 IK 解被整臂碰撞预筛拒绝%s",
            str(label),
            suffix,
        )
        return False

    def plan_cartesian_relative(self, dx=0.0, dy=0.0, dz=0.0,
                                eef_step=None, avoid_collisions=False):
        """只试算笛卡尔路径，不执行。"""
        start_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        target_pose = copy.deepcopy(start_pose)
        target_pose.position.x += dx
        target_pose.position.y += dy
        target_pose.position.z += dz
        step = self.cartesian_eef_step if eef_step is None else float(eef_step)

        plan, fraction = self.arm.move_group.compute_cartesian_path(
            [target_pose],
            step,
            avoid_collisions
        )
        self.cartesian_fraction_pub.publish(Float32(data=float(fraction)))
        return plan, fraction

    def plan_cartesian_waypoints(self, waypoints, eef_step=None, avoid_collisions=False):
        """对一组 waypoints 做一次笛卡尔规划。"""
        if not waypoints:
            return None, 0.0
        step = self.cartesian_eef_step if eef_step is None else float(eef_step)
        plan, fraction = self.arm.move_group.compute_cartesian_path(
            waypoints,
            step,
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
        if self.cartesian_retime:
            plan = self.retime_plan_from_current_state(
                plan,
                velocity_scaling=self.cartesian_velocity_scaling,
                acceleration_scaling=self.cartesian_acceleration_scaling,
            )
            return self.execute_planned_trajectory(plan, "笛卡尔轨迹", retime=False)
        return self.execute_planned_trajectory(plan, "笛卡尔轨迹", retime=False)

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
        self._wrist_vlm_request_seq += 1
        request_id = int(self._wrist_vlm_request_seq)
        result = {"done": False, "bbox": None, "failure": False}

        def bbox_cb(msg):
            data = list(msg.data)
            if len(data) == 5:
                try:
                    msg_request_id = int(round(float(data[0])))
                except Exception:
                    return
                if msg_request_id != request_id:
                    rospy.logwarn(
                        "⚠️ 忽略过期 wrist VLM bbox: got_request=%s expected=%s",
                        str(msg_request_id),
                        str(request_id),
                    )
                    return
                result["done"] = True
                result["bbox"] = list(map(int, data[1:5]))
                return
            if len(data) == 1:
                try:
                    msg_request_id = int(round(float(data[0])))
                except Exception:
                    return
                if msg_request_id == request_id:
                    result["done"] = True
                    result["failure"] = True
                return
            if len(data) == 4:
                # Legacy wrist_vlm_node compatibility. New nodes publish [request_id, xmin, ymin, xmax, ymax].
                result["done"] = True
                result["bbox"] = list(map(int, data))

        sub = rospy.Subscriber('/wrist_vlm/bbox', Float32MultiArray, bbox_cb, queue_size=1)
        self.pub_wrist_trigger.publish(f"trigger:{request_id}")
        rospy.loginfo("等待 VLM 节点返回边框数据... request_id=%d", request_id)

        deadline = time.time() + max(0.1, float(self.wrist_vlm_timeout_sec))
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                if result["done"]:
                    break
                rospy.sleep(0.02)
        finally:
            try:
                sub.unregister()
            except Exception:
                pass

        if result["bbox"] is not None:
            return result["bbox"]
        if result["failure"]:
            rospy.logwarn("VLM 节点返回空框，放弃本轮微调。")
        else:
            rospy.logwarn("VLM 节点 %.1fs 内未回复，放弃本轮微调。", self.wrist_vlm_timeout_sec)
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

    def build_wrist_object_cloud_msg(self, bbox=None):
        if self.wrist_depth is None:
            rospy.logwarn("⚠️ 腕部二次 GraspNet 无深度图，跳过。")
            return None, None

        h, w = self.wrist_depth.shape
        if self.wrist_K is not None:
            fx = float(self.wrist_K[0, 0])
            fy = float(self.wrist_K[1, 1])
            cx = float(self.wrist_K[0, 2])
            cy = float(self.wrist_K[1, 2])
        else:
            fx, fy = float(self.wrist_default_fx), float(self.wrist_default_fy)
            cx, cy = float(w) * 0.5, float(h) * 0.5

        if bbox is None:
            bbox = self.request_wrist_bbox()
        if bbox is None:
            rospy.logwarn("⚠️ 腕部二次 GraspNet 无目标 bbox，跳过。")
            return None, None

        xmin, ymin, xmax, ymax = self.expand_bbox(
            list(map(int, bbox)),
            w,
            h,
            int(self.wrist_grasp_bbox_margin_px),
        )
        if xmax <= xmin + 2 or ymax <= ymin + 2:
            rospy.logwarn("⚠️ 腕部二次 GraspNet bbox 无效: %s", str([xmin, ymin, xmax, ymax]))
            return None, None

        roi_depth = self.wrist_depth[ymin:ymax, xmin:xmax]
        valid_mask = (
            np.isfinite(roi_depth)
            & (roi_depth > self.wrist_min_valid_depth)
            & (roi_depth < self.wrist_max_valid_depth)
        )
        if not np.any(valid_mask):
            rospy.logwarn("⚠️ 腕部二次 GraspNet bbox 内无有效深度点。")
            return None, None

        front_depth = float(np.percentile(roi_depth[valid_mask], self.wrist_front_depth_percentile))
        depth_band = max(0.005, float(self.wrist_grasp_depth_band))
        object_mask = valid_mask & (roi_depth >= front_depth - 0.003) & (roi_depth <= front_depth + depth_band)
        local_y, local_x = np.where(object_mask)
        point_count = int(len(local_x))
        if point_count < int(self.wrist_grasp_min_points):
            rospy.logwarn(
                "⚠️ 腕部二次 GraspNet 点数不足: points=%d, min=%d, bbox=%s, front_depth=%.3f",
                point_count,
                self.wrist_grasp_min_points,
                str([xmin, ymin, xmax, ymax]),
                front_depth,
            )
            return None, None

        global_x = local_x.astype(np.float64) + float(xmin)
        global_y = local_y.astype(np.float64) + float(ymin)
        z = roi_depth[local_y, local_x].astype(np.float64)
        x = (global_x - cx) * z / fx
        y = (global_y - cy) * z / fy
        points = np.stack([x, y, z], axis=1).astype(np.float32)

        max_points = max(0, int(self.wrist_grasp_cloud_max_points))
        if max_points > 0 and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("object_id", 12, PointField.UINT32, 1),
        ]
        rows = [(float(p[0]), float(p[1]), float(p[2]), 0) for p in points]
        header = Header(stamp=rospy.Time.now(), frame_id=self.camera_frame)
        cloud_msg = point_cloud2.create_cloud(header, fields, rows)
        meta = {
            "bbox": [xmin, ymin, xmax, ymax],
            "points": int(len(points)),
            "front_depth": front_depth,
            "depth_band": depth_band,
        }
        return cloud_msg, meta

    def refine_grasp_with_wrist_graspnet(self, planning_frame, selected_back_distance, bbox_hint=None, seed_q=None):
        if not self.use_wrist_grasp_refinement:
            return None

        cloud_msg, meta = self.build_wrist_object_cloud_msg(bbox_hint)
        if cloud_msg is None:
            return None

        self.wrist_grasp_pose_array_received = None
        self.wrist_grasp_infos = []
        rospy.loginfo(
            "🧪 发布腕部单物体点云给二次 GraspNet: points=%d, bbox=%s, front_depth=%.3f, depth_band=%.3f",
            meta["points"],
            str(meta["bbox"]),
            meta["front_depth"],
            meta["depth_band"],
        )
        self.wrist_object_cloud_pub.publish(cloud_msg)
        rospy.sleep(0.05)
        self.wrist_object_cloud_pub.publish(cloud_msg)

        deadline = time.time() + max(0.1, float(self.wrist_grasp_timeout_sec))
        pose_msg = None
        while not rospy.is_shutdown() and time.time() < deadline:
            if (
                self.wrist_grasp_pose_array_received is not None
                and len(self.wrist_grasp_pose_array_received.poses) > 0
            ):
                pose_msg = self.wrist_grasp_pose_array_received
                break
            rospy.sleep(0.05)

        if pose_msg is None:
            rospy.logwarn("⚠️ 腕部二次 GraspNet 在 %.1f s 内无候选，保留原抓取姿态。", self.wrist_grasp_timeout_sec)
            return None

        pose_infos = self.parse_wrist_grasp_info_payload(len(pose_msg.poses))
        shelf_inward_axis = self.get_shelf_inward_axis_world() if self.force_shelf_normal_approach else None
        current_observation_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        back_distance = max(0.04, float(selected_back_distance))
        best = None
        best_score = -1e9

        for i, raw_pose in enumerate(pose_msg.poses):
            ps = PoseStamped()
            ps.header = pose_msg.header
            ps.pose = raw_pose
            target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
            if target_pose_stamped is None:
                continue

            q_orig = [
                target_pose_stamped.pose.orientation.x,
                target_pose_stamped.pose.orientation.y,
                target_pose_stamped.pose.orientation.z,
                target_pose_stamped.pose.orientation.w,
            ]
            mat_orig = tft.quaternion_matrix(q_orig)
            raw_approach_axis, raw_open_axis = self.get_configured_grasp_pose_axes(mat_orig)
            if raw_approach_axis is None:
                continue
            flat_approach = np.array([raw_approach_axis[0], raw_approach_axis[1], 0.0], dtype=np.float64)
            flat_norm = np.linalg.norm(flat_approach)
            if flat_norm < 0.001:
                continue
            flat_approach = flat_approach / flat_norm

            alignment = 1.0
            robot_z = flat_approach
            if shelf_inward_axis is not None:
                alignment_signed = float(np.dot(flat_approach, shelf_inward_axis))
                alignment = abs(alignment_signed)
                if alignment < self.wrist_grasp_min_approach_alignment:
                    rospy.logwarn(
                        "⚠️ 腕部 GraspNet 候选 %d approach 与货架法向不一致，跳过: alignment=%.3f",
                        i + 1,
                        alignment,
                    )
                    continue
                robot_z = shelf_inward_axis.copy()
                if alignment_signed < 0.0:
                    robot_z = -robot_z

            if target_pose_stamped.pose.position.z < 0.08:
                target_pose_stamped.pose.position.z = 0.08

            orientation_options = self.build_grasp_orientation_options(
                robot_z,
                raw_open_axis,
                q_orig,
                include_raw=False,
            )
            if not orientation_options:
                continue

            width, score, depth = pose_infos[i] if i < len(pose_infos) else [0.04, 0.0, 0.03]
            if float(width) > float(self.grasp_command_width_max) + 1e-4:
                continue

            for orientation_label, q_new in orientation_options:
                candidate_target_pose = copy.deepcopy(target_pose_stamped.pose)
                candidate_target_pose.orientation.x = q_new[0]
                candidate_target_pose.orientation.y = q_new[1]
                candidate_target_pose.orientation.z = q_new[2]
                candidate_target_pose.orientation.w = q_new[3]

                pre_grasp_pose = copy.deepcopy(candidate_target_pose)
                pre_grasp_pose.position.x -= robot_z[0] * back_distance
                pre_grasp_pose.position.y -= robot_z[1] * back_distance
                pre_grasp_pose.position.z -= robot_z[2] * back_distance

                pre_q = self.get_ik_seeded(pre_grasp_pose, seed_q, planning_frame) if seed_q else None
                if pre_q is None:
                    pre_q = self.get_ik_collision_checked(pre_grasp_pose, planning_frame)
                if pre_q is None:
                    continue

                rank_score = float(score) + 0.20 * float(alignment)
                if rank_score > best_score:
                    best_score = rank_score
                    best = {
                        "target_pose": candidate_target_pose,
                        "pre_grasp_pose": pre_grasp_pose,
                        "observation_pose": current_observation_pose,
                        "robot_z": np.array(robot_z, dtype=np.float64),
                        "back_distance": back_distance,
                        "width": float(width),
                        "depth": float(depth),
                        "score": float(score),
                        "alignment": float(alignment),
                        "mode": orientation_label,
                        "index": i + 1,
                    }

        if best is None:
            rospy.logwarn("⚠️ 腕部二次 GraspNet 没有通过 IK/轴约束的候选，保留原抓取姿态。")
            return None

        rospy.loginfo(
            "✅ 腕部二次 GraspNet 覆盖最终抓取: idx=%d, mode=%s, width=%.3f, depth=%.3f, "
            "score=%.3f, alignment=%.3f, back=%.3f, insert_axis=(%.3f,%.3f,%.3f)",
            best["index"],
            best["mode"],
            best["width"],
            best["depth"],
            best["score"],
            best["alignment"],
            best["back_distance"],
            best["robot_z"][0],
            best["robot_z"][1],
            best["robot_z"][2],
        )
        return best

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
                "frame_id": None,
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
        if self.wrist_K is not None:
            fx = float(self.wrist_K[0, 0])
            fy = float(self.wrist_K[1, 1])
            cx = float(self.wrist_K[0, 2])
            cy = float(self.wrist_K[1, 2])
        else:
            fx, fy = float(self.wrist_default_fx), float(self.wrist_default_fy)
            cx, cy = float(w) * 0.5, float(h) * 0.5

        if bbox_override is None:
            bbox = self.request_wrist_bbox()
            if bbox is None:
                return invalid_result()
        else:
            bbox = list(map(int, bbox_override))

        xmin, ymin, xmax, ymax = self.expand_bbox(bbox, w, h, 0)

        # 2. X轴(左右)：通过深度图抠出物理边缘中点
        roi_depth = self.wrist_depth[ymin:ymax, xmin:xmax]
        valid_mask = (roi_depth > self.wrist_min_valid_depth) & (roi_depth < self.wrist_max_valid_depth)
        if not np.any(valid_mask):
            return invalid_result()
        
        front_depth = float(np.percentile(roi_depth[valid_mask], self.wrist_front_depth_percentile))
        target_mask = (roi_depth >= front_depth) & (roi_depth < front_depth + self.wrist_depth_band)
        
        local_y_indices, local_x_indices = np.where(target_mask)
        if len(local_x_indices) == 0:
            return invalid_result()

        target_mask_u8 = (target_mask.astype(np.uint8) * 255)
        dist_map = cv2.distanceTransform(target_mask_u8, cv2.DIST_L2, 5)
        if dist_map is not None and dist_map.size and float(np.max(dist_map)) > 0.5:
            local_target_y, local_target_x = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        else:
            local_target_x = int(np.median(local_x_indices))
            local_target_y = int(np.median(local_y_indices))

        global_x_indices = local_x_indices + xmin
        global_y_indices = local_y_indices + ymin
        front_surface_x = int(local_target_x + xmin)
        front_surface_y = int(local_target_y + ymin)
        tight_bbox = [
            int(np.min(global_x_indices)),
            int(np.min(global_y_indices)),
            int(np.max(global_x_indices)),
            int(np.max(global_y_indices)),
        ]
        bbox_center_x = int(round(0.5 * (xmin + xmax)))
        bbox_center_y = int(round(0.5 * (ymin + ymax)))
        tight_bbox_center_x = int(round(0.5 * (tight_bbox[0] + tight_bbox[2])))
        tight_bbox_center_y = int(round(0.5 * (tight_bbox[1] + tight_bbox[3])))

        if self.wrist_alignment_pixel_mode == "front_surface":
            target_x, target_y = front_surface_x, front_surface_y
        elif self.wrist_alignment_pixel_mode == "tight_bbox_center":
            target_x, target_y = tight_bbox_center_x, tight_bbox_center_y
        else:
            target_x, target_y = bbox_center_x, bbox_center_y

        target_x = int(np.clip(target_x, 0, w - 1))
        target_y = int(np.clip(target_y, 0, h - 1))

        # --- 渲染调试画面 ---
        img_cv = self.wrist_rgb.copy()
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2) 
        cv2.circle(img_cv, (int(round(cx)), int(round(cy))), 5, (255, 0, 0), -1)       
        cv2.circle(img_cv, (target_x, target_y), 5, (0, 255, 0), -1) 
        if self.wrist_alignment_pixel_mode != "front_surface":
            cv2.circle(img_cv, (front_surface_x, front_surface_y), 3, (0, 165, 255), -1)
        cv2.line(img_cv, (int(round(cx)), int(round(cy))), (target_x, target_y), (0, 255, 255), 2) 
        cv2.imshow("Auto VLM Eye-in-Hand", img_cv)
        cv2.waitKey(1)

        # =========================================================================
        # 🚀 核心工业级刚体逆解：绝对杜绝向量方向错误和视差补偿错误！
        # =========================================================================
        try:
            # 步骤A：构建物体在【相机坐标系】下的绝对 3D 点 (Pose)
            # 用点位姿走 TF，能自动带入当前相机安装位姿（平移+旋转）补偿。
            p_cam = PoseStamped()
            p_cam.header.frame_id = self.camera_frame
            p_cam.header.stamp = rospy.Time(0)
            
            p_cam.pose.position.x = (target_x - cx) * front_depth / fx
            p_cam.pose.position.y = (target_y - cy) * front_depth / fy
            p_cam.pose.position.z = front_depth
            p_cam.pose.orientation.w = 1.0

            # 抓取夹爪的当前坐标系名称
            ee_link = self.arm.move_group.get_end_effector_link()
            if not ee_link: ee_link = "panda_hand"

            # 步骤B：问 TF 树，这个 3D 物体点在【夹爪坐标系】里在哪儿？
            trans_cam_to_hand = self.tf_buffer.lookup_transform(ee_link, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))

            # 固定安装位姿一致性检查：若偏离基线，说明 TF 树状态异常，跳过本次修正以防误动作。
            if self.enable_wrist_tf_guard:
                cur_t = np.array(
                    [
                        float(trans_cam_to_hand.transform.translation.x),
                        float(trans_cam_to_hand.transform.translation.y),
                        float(trans_cam_to_hand.transform.translation.z),
                    ],
                    dtype=np.float64,
                )
                cur_q = np.array(
                    [
                        float(trans_cam_to_hand.transform.rotation.x),
                        float(trans_cam_to_hand.transform.rotation.y),
                        float(trans_cam_to_hand.transform.rotation.z),
                        float(trans_cam_to_hand.transform.rotation.w),
                    ],
                    dtype=np.float64,
                )
                if self.wrist_cam_in_ee_baseline is None:
                    self.wrist_cam_in_ee_baseline = (cur_t.copy(), cur_q.copy())
                else:
                    base_t, base_q = self.wrist_cam_in_ee_baseline
                    trans_err = float(np.linalg.norm(cur_t - base_t))
                    rot_err_deg = self.quaternion_distance_deg(cur_q, base_q)
                    if trans_err > self.wrist_tf_guard_trans_tol or rot_err_deg > self.wrist_tf_guard_rot_tol_deg:
                        rospy.logwarn(
                            "⚠️ 检测到 wrist->ee 静态外参异常漂移，跳过本轮手眼修正: "
                            f"trans_err={trans_err:.4f} m, rot_err={rot_err_deg:.3f} deg"
                        )
                        return invalid_result()

            p_hand = tf2_geometry_msgs.do_transform_pose(p_cam, trans_cam_to_hand)

            # 步骤C：得出夹爪需要移动的局部距离
            # 如果物体完美对准，它在夹爪坐标系下的 x 和 y 必定是 0。
            # 所以 p_hand 的 x 和 y，正是夹爪自身需要去消除的位移！
            target_in_hand = np.array(
                [
                    self.wrist_alignment_target_hand_x,
                    self.wrist_alignment_target_hand_y,
                    self.wrist_alignment_target_hand_z,
                ],
                dtype=np.float64,
            )
            object_in_hand = np.array(
                [
                    float(p_hand.pose.position.x),
                    float(p_hand.pose.position.y),
                    float(p_hand.pose.position.z),
                ],
                dtype=np.float64,
            )
            hand_error = object_in_hand - target_in_hand

            # 步骤D：把图像平面误差转化为【世界坐标系】的位移。
            # 默认用 camera x/y 像素偏差走 TF；camera z 固定为 0，避免把前向深度混进微调。
            correction_frame = self.robot.get_planning_frame()
            if not correction_frame:
                correction_frame = "world"
            correction_mode = "hand_lateral"
            if self.wrist_alignment_use_camera_plane_tf:
                vec_cam = Vector3Stamped()
                vec_cam.header.frame_id = self.camera_frame
                vec_cam.header.stamp = rospy.Time(0)
                vec_cam.vector.x = float((target_x - cx) * front_depth / fx)
                vec_cam.vector.y = float((target_y - cy) * front_depth / fy)
                vec_cam.vector.z = 0.0
                try:
                    trans_cam_to_frame = self.tf_buffer.lookup_transform(
                        correction_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.4)
                    )
                except Exception:
                    correction_frame = "world"
                    trans_cam_to_frame = self.tf_buffer.lookup_transform(
                        correction_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.8)
                    )
                vec_world = tf2_geometry_msgs.do_transform_vector3(vec_cam, trans_cam_to_frame)
                correction_mode = "camera_plane_tf"
            else:
                vec_hand = Vector3Stamped()
                vec_hand.header.frame_id = ee_link
                vec_hand.header.stamp = rospy.Time(0)
                vec_hand.vector.x = float(hand_error[0])
                vec_hand.vector.y = float(hand_error[1])
                vec_hand.vector.z = 0.0
                try:
                    trans_hand_to_frame = self.tf_buffer.lookup_transform(
                        correction_frame, ee_link, rospy.Time(0), rospy.Duration(0.4)
                    )
                except Exception:
                    correction_frame = "world"
                    trans_hand_to_frame = self.tf_buffer.lookup_transform(
                        correction_frame, ee_link, rospy.Time(0), rospy.Duration(0.8)
                    )
                vec_world = tf2_geometry_msgs.do_transform_vector3(vec_hand, trans_hand_to_frame)
            rospy.loginfo(
                "🔧 VLM图像平面对齐: mode=%s, pixel_err=(%.1f, %.1f), depth=%.3f m, "
                "world_delta=(%.4f, %.4f, %.4f)",
                correction_mode,
                float(target_x - cx),
                float(target_y - cy),
                front_depth,
                float(vec_world.vector.x),
                float(vec_world.vector.y),
                float(vec_world.vector.z),
            )
            world_dx = float(vec_world.vector.x)
            world_dy = float(vec_world.vector.y)
            world_dz = float(vec_world.vector.z)

            details = {
                "valid": True,
                "world_dx": world_dx,
                "world_dy": world_dy,
                "world_dz": world_dz,
                "frame_id": correction_frame,
                "hand_forward": float(p_hand.pose.position.z),
                "hand_lateral_x": float(hand_error[0]),
                "hand_lateral_y": float(hand_error[1]),
                "hand_object_x": float(object_in_hand[0]),
                "hand_object_y": float(object_in_hand[1]),
                "hand_object_z": float(object_in_hand[2]),
                "hand_target_x": float(target_in_hand[0]),
                "hand_target_y": float(target_in_hand[1]),
                "hand_target_z": float(target_in_hand[2]),
                "bbox": [xmin, ymin, xmax, ymax],
                "tight_bbox": tight_bbox,
                "front_depth": front_depth,
                "target_pixel": [target_x, target_y],
                "bbox_center_pixel": [bbox_center_x, bbox_center_y],
                "front_surface_pixel": [front_surface_x, front_surface_y],
                "alignment_pixel_mode": self.wrist_alignment_pixel_mode,
                "correction_mode": correction_mode,
                "camera_fx": float(fx),
                "camera_fy": float(fy),
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
        # 这里不再默认把整个 depth_hint 都兑现成前向插入。
        # 对货架里的杯子/带把手物体，完整兑现会导致明显插得过深。
        # 改成“前表面停靠 + 小量包裹”策略：只保留 depth_hint 的一部分，并且有上限。
        PANDA_HAND_TO_FINGERTIP = 0.1034
        depth_hint = float(depth_hint) if depth_hint is not None else 0.03
        effective_depth = np.clip(
            max(0.0, depth_hint) * max(0.0, self.wrist_insert_depth_scale),
            0.0,
            max(0.0, self.wrist_insert_depth_cap),
        )
        target = PANDA_HAND_TO_FINGERTIP - effective_depth + self.wrist_grasp_depth_margin
        return float(np.clip(target, self.wrist_grasp_forward_min, self.wrist_grasp_forward_max))

    def compute_insert_distance_from_wrist(self, chosen_depth, fallback_distance):
        fallback_distance = float(max(0.0, fallback_distance))
        if not self.use_wrist_guided_insert_distance:
            return fallback_distance

        if self.wrist_insert_allow_shorten:
            conservative_fallback = min(
                fallback_distance,
                max(self.min_insert_distance, self.wrist_visibility_fallback_max_insert),
            )
        else:
            conservative_fallback = fallback_distance

        measurement = self.calculate_world_correction(return_details=True)
        if not measurement["valid"] or measurement["hand_forward"] is None:
            rospy.logwarn(
                "⚠️ 手腕相机未返回有效前向距离，回退到保守插入距离 %.3f m（模型估计 %.3f m）。",
                conservative_fallback,
                fallback_distance,
            )
            return conservative_fallback

        visibility_bbox = measurement.get("tight_bbox") or measurement.get("bbox")
        if visibility_bbox is not None and not self.wrist_bbox_well_visible(visibility_bbox):
            rospy.logwarn(
                "⚠️ 手腕目标前表面未完整露出，bbox=%s，禁止缩短插入距离，"
                "回退到保守插入距离 %.3f m（模型估计 %.3f m）。",
                visibility_bbox,
                conservative_fallback,
                fallback_distance,
            )
            return conservative_fallback

        desired_forward = self.compute_wrist_grasp_forward_target(depth_hint=chosen_depth)
        measured_forward = float(measurement["hand_forward"])
        if measured_forward <= 0.0:
            rospy.logwarn(
                "⚠️ 手腕测得前向距离异常(measured_forward=%.3f m)，禁止缩短插入距离，"
                "回退到保守插入距离 %.3f m（模型估计 %.3f m）。",
                measured_forward,
                conservative_fallback,
                fallback_distance,
            )
            return conservative_fallback
        insert_distance_from_measurement = max(0.0, measured_forward - desired_forward)
        if self.wrist_insert_allow_shorten:
            lower_bound = max(
                self.min_insert_distance,
                fallback_distance * max(0.0, self.wrist_min_insert_fraction),
                fallback_distance - max(0.0, self.wrist_insert_shorten_max),
            )
        else:
            lower_bound = max(
                self.min_insert_distance,
                fallback_distance,
            )
        upper_bound = fallback_distance + self.wrist_extra_insert_max
        insert_distance = float(
            np.clip(
                insert_distance_from_measurement,
                lower_bound,
                upper_bound,
            )
        )
        rospy.loginfo(
            f"[5/6] 手腕测距修正插入: measured_forward={measured_forward:.3f} m, "
            f"desired_forward={desired_forward:.3f} m, fallback={fallback_distance:.3f} m, "
            f"measured_insert={insert_distance_from_measurement:.3f} m, "
            f"clamp=[{lower_bound:.3f}, {upper_bound:.3f}] m, "
            f"final_insert={insert_distance:.3f} m"
        )
        return insert_distance

    def cap_final_insert_distance(self, insert_distance, selected_back_distance):
        insert_distance = float(max(0.0, insert_distance))
        caps = []
        if self.insert_distance_over_back_cap > 1e-6 and selected_back_distance is not None:
            caps.append(
                max(
                    self.min_insert_distance,
                    float(selected_back_distance) + float(self.insert_distance_over_back_cap),
                )
            )
        if self.insert_distance_absolute_max > 1e-6:
            caps.append(max(self.min_insert_distance, float(self.insert_distance_absolute_max)))
        if not caps:
            return insert_distance

        capped_distance = min(insert_distance, min(caps))
        if capped_distance + 1e-4 < insert_distance:
            rospy.logwarn(
                "⚠️ 插入距离超过上限，执行限幅: %.3f -> %.3f m "
                "(selected_back=%.3f m, over_back_cap=%.3f m, absolute_max=%.3f m)",
                insert_distance,
                capped_distance,
                float(selected_back_distance) if selected_back_distance is not None else -1.0,
                self.insert_distance_over_back_cap,
                self.insert_distance_absolute_max,
            )
        return float(capped_distance)

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

        def correction_from_measurement(measurement):
            correction = np.array(
                [measurement["world_dx"], measurement["world_dy"], measurement["world_dz"]],
                dtype=np.float64,
            )
            if not self.wrist_alignment_allow_vertical:
                correction[2] = 0.0
            correction[0] = float(np.clip(correction[0], -self.wrist_alignment_max_step_xy, self.wrist_alignment_max_step_xy))
            correction[1] = float(np.clip(correction[1], -self.wrist_alignment_max_step_xy, self.wrist_alignment_max_step_xy))
            correction[2] = float(np.clip(correction[2], -self.wrist_alignment_max_step_z, self.wrist_alignment_max_step_z))
            return correction

        for idx in range(max_iters):
            measurement = self.calculate_world_correction(return_details=True)
            if not measurement["valid"]:
                if had_usable_visibility:
                    rospy.logwarn(
                        f"{stage_label} 后续复核未返回有效测量，禁止继续前插抓取。"
                    )
                    return False
                rospy.logwarn(f"{stage_label} 手腕相机未返回有效测量。")
                return False

            bbox = measurement["bbox"]
            visible_ok = self.wrist_bbox_well_visible(bbox)
            correction = correction_from_measurement(measurement)
            correction_norm = float(np.linalg.norm(correction))

            rospy.loginfo(
                f"{stage_label} 手眼对齐误差: "
                f"obj_hand=({measurement.get('hand_object_x', 0.0):.3f},"
                f"{measurement.get('hand_object_y', 0.0):.3f},"
                f"{measurement.get('hand_object_z', 0.0):.3f}) m, "
                f"target_hand=({measurement.get('hand_target_x', 0.0):.3f},"
                f"{measurement.get('hand_target_y', 0.0):.3f},"
                f"{measurement.get('hand_target_z', 0.0):.3f}) m, "
                f"corr_{measurement.get('frame_id', 'world')}="
                f"({correction[0]:.3f},{correction[1]:.3f},{correction[2]:.3f}) m"
            )

            if visible_ok:
                had_usable_visibility = True
            elif require_visible:
                if had_usable_visibility:
                    rospy.logwarn(
                        f"{stage_label} 后续复核时目标未完整露出，当前 bbox={bbox}，"
                        "禁止继续前插抓取。"
                    )
                    return False
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

        measurement = self.calculate_world_correction(return_details=True)
        if not measurement["valid"]:
            rospy.logwarn(f"{stage_label} 最大校正次数后复核失败，禁止继续前插抓取。")
            return False
        correction = correction_from_measurement(measurement)
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm <= self.wrist_alignment_tol:
            rospy.loginfo(
                f"{stage_label} 最大校正次数后复核通过，correction_norm={correction_norm:.4f} m"
            )
            return True

        rospy.logwarn(
            f"{stage_label} 达到最大校正次数仍未对准，correction_norm={correction_norm:.4f} m > "
            f"{self.wrist_alignment_tol:.4f} m，禁止继续前插抓取。"
        )
        return False

    def run_wrist_alignment_stage(self, stage_label):
        """
        执行一次腕部相机对齐阶段。
        返回 (ok_to_continue, bbox)，当 require_wrist_alignment=false 时，测量失败只会跳过微调。
        """
        if self.use_iterative_wrist_alignment:
            if not self.run_wrist_alignment_loop(
                stage_label,
                max_iters=self.wrist_alignment_max_iters,
                require_visible=self.wrist_require_full_visibility,
            ):
                if self.require_wrist_alignment:
                    rospy.logwarn("🛑 腕部多轮对齐失败！")
                    self.set_failure_reason("wrist_alignment_failed", "iterative_alignment_failed")
                    return False, None
                rospy.logwarn("⚠️ 腕部多轮对齐失败，本轮跳过微调并按固定轴继续执行。")
            return True, None

        fine_measurement = self.calculate_world_correction(return_details=True)
        if not fine_measurement["valid"]:
            if self.require_wrist_alignment:
                rospy.logwarn("🛑 腕部相机微调未返回有效测量，禁止继续前插抓取。")
                self.set_failure_reason("wrist_alignment_failed", "single_alignment_no_measurement")
                return False, None
            rospy.logwarn("⚠️ 腕部相机微调未返回有效测量，本轮跳过微调并按固定轴继续执行。")
            return True, None

        wrist_refine_bbox = fine_measurement.get("bbox")
        fine_dx = np.clip(
            fine_measurement["world_dx"],
            -self.wrist_alignment_max_step_xy,
            self.wrist_alignment_max_step_xy,
        )
        fine_dy = np.clip(
            fine_measurement["world_dy"],
            -self.wrist_alignment_max_step_xy,
            self.wrist_alignment_max_step_xy,
        )
        fine_dz = (
            np.clip(
                fine_measurement["world_dz"],
                -self.wrist_alignment_max_step_z,
                self.wrist_alignment_max_step_z,
            )
            if self.wrist_alignment_allow_vertical
            else 0.0
        )
        if not self.apply_safe_world_correction(fine_dx, fine_dy, fine_dz):
            if self.require_wrist_alignment:
                rospy.logwarn("🛑 腕部相机微调执行失败！")
                self.set_failure_reason("wrist_alignment_failed", "single_alignment_failed")
                return False, wrist_refine_bbox
            rospy.logwarn("⚠️ 腕部相机微调执行失败，本轮保留当前位姿继续执行。")
        return True, wrist_refine_bbox

    def apply_final_preinsert_world_z_lift(self):
        """在最终前插前，对末端做一个世界坐标系 Z 向补偿。"""
        lift = float(getattr(self, "final_preinsert_world_z_lift", 0.0))
        if abs(lift) <= 1e-6:
            return True

        current_z = float(self.arm.move_group.get_current_pose().pose.position.z)
        rospy.loginfo(
            "[4.95/6] 最终前插前世界Z补偿: dz=%+.4f m, ee_z %.4f -> %.4f",
            lift,
            current_z,
            current_z + lift,
        )
        step = max(0.0005, min(0.005, abs(lift)))
        min_step = max(0.0001, min(0.0005, abs(lift)))
        if not self.segmented_cartesian_move(
            dx=0.0,
            dy=0.0,
            dz=lift,
            description="[4.95/6] 最终世界Z补偿",
            step_size=step,
            min_step=min_step,
            avoid_collisions=False,
        ):
            rospy.logwarn("🛑 最终前插前世界Z补偿失败。")
            self.set_failure_reason("preinsert_z_lift_failed", "final_world_z_lift_failed")
            return False
        rospy.sleep(0.1)
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

    def plan_direct_grasp_fallback(self, planning_frame, pose_infos, seed_q_global):
        """工程兜底：观察位全失败时，直接规划到最终抓取位准备闭爪。"""
        if not self.enable_direct_grasp_fallback:
            return None
        if self.grasp_pose_array_received is None or len(self.grasp_pose_array_received.poses) == 0:
            return None

        rospy.logwarn(
            "⚠️ 观察位搜索失败，启动工程兜底直接抓取：直接规划到候选抓取位，到位后闭爪。"
        )
        shelf_inward_axis = self.get_shelf_inward_axis_world() if self.force_shelf_normal_approach else None

        self.arm.move_group.set_planning_time(self.direct_grasp_planning_time)
        self.arm.move_group.set_num_planning_attempts(self.direct_grasp_planning_attempts)
        try:
            for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
                ps = PoseStamped()
                ps.header = self.grasp_pose_array_received.header
                ps.pose = raw_pose

                target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
                if target_pose_stamped is None:
                    continue

                q_orig = [
                    target_pose_stamped.pose.orientation.x,
                    target_pose_stamped.pose.orientation.y,
                    target_pose_stamped.pose.orientation.z,
                    target_pose_stamped.pose.orientation.w,
                ]
                mat_orig = tft.quaternion_matrix(q_orig)
                raw_approach_axis, raw_open_axis = self.get_configured_grasp_pose_axes(mat_orig)

                robot_z = None
                flat_approach = None
                if raw_approach_axis is not None:
                    flat_approach = np.array([raw_approach_axis[0], raw_approach_axis[1], 0.0], dtype=np.float64)
                    norm = np.linalg.norm(flat_approach)
                    if norm >= 0.001:
                        robot_z = flat_approach / norm

                if shelf_inward_axis is not None:
                    robot_z = shelf_inward_axis.copy()
                    if flat_approach is not None and np.linalg.norm(flat_approach) >= 0.001:
                        if float(np.dot(robot_z, flat_approach)) < 0.0:
                            robot_z = -robot_z

                if robot_z is None:
                    robot_z = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                robot_z_norm = np.linalg.norm(robot_z)
                if robot_z_norm < 1e-6:
                    continue
                robot_z = robot_z / robot_z_norm

                if target_pose_stamped.pose.position.z < 0.08:
                    target_pose_stamped.pose.position.z = 0.08
                target_pose_stamped.pose = self.correct_target_height_from_gazebo(
                    target_pose_stamped.pose,
                    label=f"直接兜底姿态{i+1}",
                )

                orientation_options = []

                def append_orientation(label, quat):
                    if quat is None or len(quat) < 4:
                        return
                    for _existing_label, existing_quat in orientation_options:
                        if self.quaternion_distance_deg(existing_quat, quat) < 1.0:
                            return
                    orientation_options.append((label, list(quat)))

                append_orientation("direct_raw", q_orig)
                for option_label, option_quat in self.build_grasp_orientation_options(
                    robot_z,
                    raw_open_axis,
                    q_orig,
                    include_raw=True,
                ):
                    append_orientation(option_label, option_quat)

                if not orientation_options:
                    continue

                for back_distance in self.direct_grasp_back_distance_candidates:
                    back_distance = float(max(0.0, back_distance))
                    for orientation_label, q_candidate in orientation_options:
                        final_pose = copy.deepcopy(target_pose_stamped.pose)
                        self.apply_quaternion_to_pose(final_pose, q_candidate)

                        approach_pose = copy.deepcopy(final_pose)
                        approach_pose.position.x -= robot_z[0] * back_distance
                        approach_pose.position.y -= robot_z[1] * back_distance
                        approach_pose.position.z -= robot_z[2] * back_distance

                        direct_q = None
                        if seed_q_global is not None:
                            direct_q = self.get_ik_seeded(approach_pose, list(seed_q_global), planning_frame)
                        if direct_q is None:
                            direct_q = self.get_ik_collision_checked(
                                approach_pose,
                                planning_frame,
                                timeout_sec=max(0.35, self.observation_ik_timeout_sec),
                            )
                        if direct_q is None:
                            rospy.logdebug(
                                "直接兜底 IK 失败: pose=%d, mode=%s, back=%.3f",
                                i + 1,
                                orientation_label,
                                back_distance,
                            )
                            continue
                        if not self.joint_state_is_valid(
                            direct_q,
                            label=f"直接兜底姿态{i+1}/{orientation_label}/back={back_distance:.3f}",
                        ):
                            continue

                        try:
                            self.arm.move_group.set_start_state_to_current_state()
                            self.arm.move_group.clear_pose_targets()
                            self.arm.move_group.set_joint_value_target(direct_q)
                            plan_result = self.arm.move_group.plan()
                            plan_success, plan = self._unpack_plan_result(plan_result)
                        finally:
                            self.arm.move_group.clear_pose_targets()

                        if not plan_success:
                            rospy.logdebug(
                                "直接兜底规划失败: pose=%d, mode=%s, back=%.3f",
                                i + 1,
                                orientation_label,
                                back_distance,
                            )
                            continue

                        if i < len(pose_infos):
                            chosen_width = float(pose_infos[i][0])
                            chosen_depth = float(pose_infos[i][2])
                        else:
                            chosen_width = 0.04
                            chosen_depth = 0.03

                        rospy.logwarn(
                            "✅ 工程兜底预抓取位规划成功: 姿态 %d, mode=%s, back=%.3f m, "
                            "width=%.3f m, depth=%.3f m",
                            i + 1,
                            orientation_label,
                            back_distance,
                            chosen_width,
                            chosen_depth,
                        )
                        rospy.logwarn(
                            "📍 工程兜底位姿诊断: target=(%.3f, %.3f, %.3f), "
                            "pregrasp=(%.3f, %.3f, %.3f), insert_axis=(%.3f, %.3f, %.3f)",
                            final_pose.position.x,
                            final_pose.position.y,
                            final_pose.position.z,
                            approach_pose.position.x,
                            approach_pose.position.y,
                            approach_pose.position.z,
                            robot_z[0],
                            robot_z[1],
                            robot_z[2],
                        )
                        return {
                            "plan": plan,
                            "target_pose": final_pose,
                            "pre_grasp_pose": copy.deepcopy(approach_pose),
                            "observation_pose": copy.deepcopy(approach_pose),
                            "robot_z": robot_z,
                            "back_distance": back_distance,
                            "width": chosen_width,
                            "depth": chosen_depth,
                        }
        finally:
            self.arm.move_group.set_planning_time(self.default_planning_time)
            self.arm.move_group.set_num_planning_attempts(self.default_planning_attempts)
            self.arm.move_group.clear_pose_targets()

        rospy.logwarn("⚠️ 工程兜底直接抓取也未找到可执行路径。")
        return None

    def execute_direct_grasp_from_current_pose(self, target_pose, robot_z, chosen_width, chosen_depth, direct_insert_distance=0.0):
        """直接抓取兜底的闭爪、附着和撤出流程。"""
        rospy.logwarn("[4.5/6] 工程兜底：已到兜底预抓取位，跳过腕部对齐。")
        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm > 1e-6:
            robot_z = robot_z / norm

        direct_insert_distance = max(0.0, float(direct_insert_distance))
        if direct_insert_distance > 1e-6 and norm > 1e-6:
            rospy.loginfo(">>> 工程兜底短距离插入 %.3f m 后再闭爪。", direct_insert_distance)
            if not self.segmented_cartesian_move(
                dx=robot_z[0] * direct_insert_distance,
                dy=robot_z[1] * direct_insert_distance,
                dz=robot_z[2] * direct_insert_distance,
                description="工程兜底短距离插入",
                step_size=min(self.insert_step_size, max(0.005, direct_insert_distance)),
                min_step=0.005,
                avoid_collisions=False,
            ):
                rospy.logwarn("🛑 工程兜底短距离插入失败。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("pregrasp_unreachable", "direct_insert_failed")
                return False

        self.log_gripper_width_snapshot("直接抓取闭爪前")
        functional_tracking_ready = self.begin_gazebo_functional_grasp_tracking(target_pose)

        command_width, lightly_shrunk_width = self.compute_command_grasp_width(chosen_width)
        rospy.loginfo(
            f"[6/6] 直接闭合夹爪 (预测宽度: {chosen_width:.3f}m, "
            f"收紧边距: {self.last_grasp_width_margin_used:.3f}m, "
            f"收紧后: {lightly_shrunk_width:.3f}m, 下发宽度: {command_width:.3f}m)"
        )

        close_ok = self.gripper.close(width=command_width, force=self.grasp_force)
        if not close_ok:
            observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
            if observed_width is not None and float(observed_width) <= self.grasp_empty_close_threshold:
                rospy.logwarn(
                    "🛑 直接抓取闭爪后夹爪几乎完全闭合(observed=%.3f m <= %.3f m)，判定为空夹。",
                    float(observed_width),
                    self.grasp_empty_close_threshold,
                )
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("grasp_failed", "direct_grasp_empty_close")
                return False

            retry_width = float(
                np.clip(command_width - 0.006, self.grasp_command_width_min, self.grasp_command_width_max)
            )
            if retry_width + 1e-4 < command_width:
                rospy.logwarn(
                    "⚠️ 直接抓取首次闭爪失败，尝试窄宽度重试: %.3f -> %.3f m",
                    command_width,
                    retry_width,
                )
                close_ok = self.gripper.close(width=retry_width, force=self.grasp_force)
                if close_ok:
                    command_width = retry_width

        hold_check_width = max(float(chosen_width), float(command_width))
        if not close_ok:
            if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                rospy.logwarn("⚠️ 直接抓取闭爪动作返回失败，继续撤出并用 Gazebo 功能性校验判断。")
            else:
                rospy.logwarn("🛑 直接抓取夹爪闭合动作失败。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("grasp_failed", "direct_grasp_close_failed")
                return False

        if self.force_grasp_verify_after_close:
            close_width_ok = self.verify_force_grasp_hold("直接抓取闭爪后", hold_check_width)
            if not close_width_ok:
                if self.direct_grasp_require_width_hold:
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("grasp_failed", "direct_grasp_unstable_after_close")
                    return False
                rospy.logwarn("⚠️ 直接抓取闭爪后宽度校验未通过；工程兜底模式继续执行撤出。")

        self.last_grasp_width_hint = float(chosen_width)
        self.last_grasp_depth_hint = float(chosen_depth)
        self.current_item_collision_size = self.estimate_grasped_object_size(
            width_hint=chosen_width,
            depth_hint=chosen_depth,
        )
        attach_ok = self.attach_gazebo_object_if_possible(target_pose)
        if self.enable_gazebo_attach and getattr(self.gripper, "last_grasp_was_soft_success", False) and not attach_ok:
            rospy.logwarn("🛑 直接抓取仅达到软成功，且 Gazebo 未找到可附着目标物体，按未抓住处理。")
            self.set_failure_reason("grasp_failed", "direct_soft_grasp_without_attach")
            return False
        self.attach_grasped_object(width_hint=chosen_width, depth_hint=chosen_depth)
        rospy.sleep(0.8)
        self.save_task_keyframe("grasped_direct")

        post_grasp_lift = max(0.0, float(self.post_grasp_lift_distance))
        if post_grasp_lift > 1e-6:
            rospy.loginfo(">>> 直接抓取后先抬高 %.3f m。", post_grasp_lift)
            if not self.segmented_cartesian_move(
                dx=0.0,
                dy=0.0,
                dz=post_grasp_lift,
                description="直接抓取后抬升稳定物体",
                step_size=min(self.lift_step_size, max(0.005, post_grasp_lift)),
                min_step=0.005,
                avoid_collisions=False,
            ):
                rospy.logwarn("🛑 直接抓取后抬升失败。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("retreat_failed", "direct_post_grasp_lift_failed")
                return False
            rospy.sleep(0.2)

        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm > 1e-6:
            robot_z = robot_z / norm
        retreat_distance = max(0.0, float(self.direct_grasp_retreat_distance))
        if retreat_distance > 1e-6 and norm > 1e-6:
            rospy.loginfo(">>> 直接抓取后沿插入反方向撤出 %.3f m。", retreat_distance)
            if not self.segmented_cartesian_move(
                dx=-robot_z[0] * retreat_distance,
                dy=-robot_z[1] * retreat_distance,
                dz=-robot_z[2] * retreat_distance,
                description="直接抓取后撤出",
                step_size=self.retreat_step_size,
                min_step=0.01,
                avoid_collisions=False,
            ):
                rospy.logwarn("🛑 直接抓取后撤出失败。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("retreat_failed", "direct_retreat_failed")
                return False

        if self.force_grasp_verify_after_retreat and self.direct_grasp_require_width_hold:
            if not self.verify_force_grasp_hold("直接抓取撤出后", hold_check_width):
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("retreat_failed", "direct_grasp_unstable_after_retreat")
                return False

        gazebo_ok = self.verify_gazebo_functional_grasp("直接抓取撤出后")
        if gazebo_ok is False and self.direct_grasp_require_width_hold:
            self.clear_functional_grasp_candidate()
            self.set_failure_reason("retreat_failed", "direct_functional_grasp_verify_failed")
            return False

        self.grasp_pose_array_received = None
        self.grasp_infos = []
        return True

    def execute_simple_front_grasp(
        self,
        target_pose,
        robot_z,
        selected_back_distance,
        chosen_width,
        chosen_depth,
    ):
        """固定前插流程：VLM 对准后沿插入轴前插、闭爪、水平撤出。"""
        robot_z = np.array(robot_z, dtype=np.float64)
        norm = np.linalg.norm(robot_z)
        if norm < 1e-6:
            self.set_failure_reason("insert_failed", "invalid_insert_axis")
            return False
        robot_z = robot_z / norm

        # 根据物体深度动态计算额外插入量，而非硬编码 1cm
        dynamic_extra_depth = self.compute_insert_extra_depth(
            width_hint=chosen_width,
            depth_hint=chosen_depth,
        )
        rospy.loginfo(
            "[5/6] 动态 extra_depth: %.3f m (width=%.3f, depth=%.3f)",
            dynamic_extra_depth,
            float(chosen_width),
            float(chosen_depth),
        )
        raw_insert_distance = self.compute_remaining_insert_distance(
            target_pose,
            robot_z,
            dynamic_extra_depth,
        )
        # 诊断：打印插入距离的组成
        current_pose = self.arm.move_group.get_current_pose().pose
        delta_to_target = np.array(
            [
                target_pose.position.x - current_pose.position.x,
                target_pose.position.y - current_pose.position.y,
                target_pose.position.z - current_pose.position.z,
            ],
            dtype=np.float64,
        )
        remaining_to_target = max(0.0, float(np.dot(delta_to_target, robot_z)))
        rospy.loginfo(
            "[5/6] 插入距离诊断: remaining_to_target=%.3f m, dynamic_extra_depth=%.3f m, "
            "raw=%.3f m, selected_back=%.3f m, over_back_cap=%.3f m, absolute_max=%.3f m",
            remaining_to_target,
            float(dynamic_extra_depth),
            raw_insert_distance,
            float(selected_back_distance) if selected_back_distance is not None else -1.0,
            self.insert_distance_over_back_cap,
            self.insert_distance_absolute_max,
        )
        insert_distance = self.cap_final_insert_distance(raw_insert_distance, selected_back_distance)
        insert_distance = max(self.min_insert_distance, float(insert_distance))
        rospy.loginfo(
            "[5/6] 固定模式：VLM 对齐完成，沿固定轴直接插入 %.3f m。",
            insert_distance,
        )

        self.publish_active_grasp_target(
            target_pose,
            robot_z,
            selected_back_distance,
            width_hint=chosen_width,
            depth_hint=chosen_depth,
            pre_grasp_pose=self.arm.move_group.get_current_pose().pose,
            observation_pose=self.arm.move_group.get_current_pose().pose,
            insert_distance=insert_distance,
        )
        self.refresh_shelf_collision_space()

        if not self.segmented_cartesian_move(
            dx=robot_z[0] * insert_distance,
            dy=robot_z[1] * insert_distance,
            dz=robot_z[2] * insert_distance,
            description="[5/6] 固定模式前插",
            step_size=self.insert_step_size,
            min_step=0.01,
            avoid_collisions=False,
        ):
            rospy.logwarn("🛑 固定模式前插失败。")
            self.set_failure_reason("insert_failed", "simple_front_insert_failed")
            return False

        rospy.sleep(0.2)
        self.log_gripper_width_snapshot("固定模式闭爪前")
        functional_tracking_ready = self.begin_gazebo_functional_grasp_tracking(target_pose)

        command_width, lightly_shrunk_width = self.compute_command_grasp_width(chosen_width)
        rospy.loginfo(
            "[6/6] 固定模式闭爪 (预测宽度: %.3fm, 收紧边距: %.3fm, 收紧后: %.3fm, 下发宽度: %.3fm)",
            float(chosen_width),
            float(self.last_grasp_width_margin_used),
            float(lightly_shrunk_width),
            float(command_width),
        )
        close_ok = self.gripper.close(width=command_width, force=self.grasp_force)
        hold_check_width = max(float(chosen_width), float(command_width))
        if not close_ok:
            observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
            observed_text = "unknown" if observed_width is None else f"{float(observed_width):.3f}"
            if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                rospy.logwarn(
                    "⚠️ 固定模式闭爪动作返回失败，当前观测宽度=%s m，继续撤出后做功能性校验。",
                    observed_text,
                )
            else:
                rospy.logwarn("🛑 固定模式闭爪失败。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("grasp_failed", "simple_front_close_failed")
                return False

        if self.force_grasp_verify_after_close:
            if not self.verify_force_grasp_hold("固定模式闭爪后", hold_check_width):
                if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                    rospy.logwarn("⚠️ 固定模式闭爪后宽度校验未通过，继续撤出后做功能性校验。")
                else:
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("grasp_failed", "simple_front_unstable_after_close")
                    return False

        self.last_grasp_width_hint = float(chosen_width)
        self.last_grasp_depth_hint = float(chosen_depth)
        self.current_item_collision_size = self.estimate_grasped_object_size(
            width_hint=chosen_width,
            depth_hint=chosen_depth,
        )
        attach_ok = self.attach_gazebo_object_if_possible(target_pose)
        if self.enable_gazebo_attach and getattr(self.gripper, "last_grasp_was_soft_success", False) and not attach_ok:
            rospy.logwarn("🛑 固定模式仅达到软成功，且 Gazebo 未找到可附着目标，按未抓住处理。")
            self.set_failure_reason("grasp_failed", "simple_front_soft_grasp_without_attach")
            return False
        self.attach_grasped_object(width_hint=chosen_width, depth_hint=chosen_depth)
        rospy.sleep(0.3)
        self.save_task_keyframe("grasped_simple_front")

        retreat_distance = insert_distance + max(0.0, float(self.simple_front_extra_retreat))
        rospy.loginfo(">>> 固定模式：闭爪后沿原路水平撤出 %.3f m。", retreat_distance)
        if not self.segmented_cartesian_move(
            dx=-robot_z[0] * retreat_distance,
            dy=-robot_z[1] * retreat_distance,
            dz=-robot_z[2] * retreat_distance,
            description="固定模式水平撤出",
            step_size=self.retreat_step_size,
            min_step=0.01,
            avoid_collisions=False,
        ):
            rospy.logwarn("🛑 固定模式撤出失败。")
            self.clear_functional_grasp_candidate()
            self.set_failure_reason("retreat_failed", "simple_front_retreat_failed")
            return False

        post_grasp_lift = max(0.0, float(self.post_grasp_lift_distance))
        if post_grasp_lift > 1e-6:
            self.segmented_cartesian_move(
                dx=0.0,
                dy=0.0,
                dz=post_grasp_lift,
                description="固定模式撤出后抬升",
                step_size=min(self.lift_step_size, max(0.005, post_grasp_lift)),
                min_step=0.005,
                avoid_collisions=False,
            )

        gazebo_ok = self.verify_gazebo_functional_grasp("固定模式撤出后")
        if gazebo_ok is False:
            self.clear_functional_grasp_candidate()
            self.set_failure_reason("retreat_failed", "simple_front_functional_verify_failed")
            return False
        if self.force_grasp_verify_after_retreat:
            if not self.verify_force_grasp_hold("固定模式撤出后", hold_check_width, wait_sec=0.10):
                if gazebo_ok is not True:
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("retreat_failed", "simple_front_width_lost_after_retreat")
                    return False

        self.grasp_pose_array_received = None
        self.grasp_infos = []
        return True

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
        direct_grasp_fallback_used = False
        pose_count = len(self.grasp_pose_array_received.poses)
        pose_infos, seed_q_global = self.parse_grasp_info_payload(pose_count)

        self.clear_active_grasp_target()
        if self.refresh_collision_before_pick:
            self.refresh_shelf_collision_space()
        self.normalize_current_joint_state_if_needed()
        self.arm.move_group.set_planning_time(self.observation_planning_time)
        self.arm.move_group.set_num_planning_attempts(self.observation_planning_attempts)
        search_start_wall = time.time()
        search_timeout_hit = False

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
            if (time.time() - search_start_wall) > self.max_observation_search_time:
                search_timeout_hit = True
                break
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
            raw_approach_axis, raw_open_axis = self.get_configured_grasp_pose_axes(mat_orig)
            if raw_approach_axis is None:
                continue

            flat_approach = np.array([raw_approach_axis[0], raw_approach_axis[1], 0.0])
            norm = np.linalg.norm(flat_approach)

            if norm < 0.001:
                continue

            robot_z = flat_approach / norm
            if shelf_inward_axis is not None:
                robot_z = shelf_inward_axis.copy()
                if float(np.dot(robot_z, flat_approach)) < 0.0:
                    robot_z = -robot_z

            if target_pose_stamped.pose.position.z < 0.08:
                target_pose_stamped.pose.position.z = 0.08
            target_pose_stamped.pose = self.correct_target_height_from_gazebo(
                target_pose_stamped.pose,
                label=f"姿态{i+1}",
            )

            seed_q = list(seed_q_global) if seed_q_global is not None else None

            success = False
            orientation_options = self.build_grasp_orientation_options(
                robot_z,
                raw_open_axis,
                q_orig,
                include_raw=False,
            )
            for back_distance in self.observation_back_distance_candidates:
                if (time.time() - search_start_wall) > self.max_observation_search_time:
                    search_timeout_hit = True
                    break

                for orientation_label, q_new in orientation_options:
                    if (time.time() - search_start_wall) > self.max_observation_search_time:
                        search_timeout_hit = True
                        break

                    candidate_target_pose = copy.deepcopy(target_pose_stamped.pose)
                    candidate_target_pose.orientation.x = q_new[0]
                    candidate_target_pose.orientation.y = q_new[1]
                    candidate_target_pose.orientation.z = q_new[2]
                    candidate_target_pose.orientation.w = q_new[3]

                    pre_grasp_pose = copy.deepcopy(candidate_target_pose)
                    pre_grasp_pose.position.x -= robot_z[0] * back_distance
                    pre_grasp_pose.position.y -= robot_z[1] * back_distance
                    pre_grasp_pose.position.z -= robot_z[2] * back_distance

                    observation_pose = copy.deepcopy(pre_grasp_pose)
                    observation_pose.position.x -= robot_z[0] * self.wrist_observation_backoff
                    observation_pose.position.y -= robot_z[1] * self.wrist_observation_backoff
                    observation_pose.position.z -= robot_z[2] * self.wrist_observation_backoff

                    config_success = False
                    for obs_lift in self.observation_lift_candidates:
                        observation_pose_trial = copy.deepcopy(observation_pose)
                        observation_pose_trial.position.z += float(obs_lift)

                        planned_observation_pose, plan, observation_plan_label, observation_status = (
                            self.plan_observation_pose_with_relaxed_orientation(
                                observation_pose_trial,
                                orientation_label,
                                robot_z,
                                raw_open_axis,
                                q_orig,
                                seed_q,
                                planning_frame,
                            )
                        )
                        if plan is None:
                            log_fn = rospy.logwarn if observation_status == "planning_failed" else rospy.logdebug
                            log_fn(
                                f"⚠️ 观察位 IK 失败: 姿态 {i+1}, mode={orientation_label}, "
                                f"back_distance={back_distance:.3f} m, obs_lift={obs_lift:.3f} m, "
                                f"status={observation_status}"
                            )
                            continue

                        observation_note = (
                            ""
                            if observation_plan_label == str(orientation_label)
                            else f"obs_mode={observation_plan_label}, "
                        )
                        rospy.loginfo(
                            f"🎉 成功锁定抓取姿态！(mode={orientation_label}, "
                            f"{observation_note}"
                            f"axis={self.grasp_pose_approach_axis_label}, open={self.grasp_pose_open_axis_label}, "
                            f"insert_axis=({robot_z[0]:.3f},{robot_z[1]:.3f},{robot_z[2]:.3f}), "
                            f"back_distance={back_distance:.3f} m, obs_lift={obs_lift:.3f} m)"
                        )
                        stored_target_pose, stored_pre_grasp_pose, retained_lift = (
                            self.build_lift_retained_grasp_poses(
                                candidate_target_pose,
                                pre_grasp_pose,
                                obs_lift,
                            )
                        )
                        if retained_lift > 1e-4:
                            rospy.loginfo(
                                "📐 保留观察抬高量到预抓取/插入高度: retained_lift=%.3f m",
                                retained_lift,
                            )
                        best_plan = plan
                        best_pre_grasp_pose = stored_pre_grasp_pose
                        best_target_pose = stored_target_pose
                        best_observation_pose = planned_observation_pose
                        best_robot_z = robot_z
                        selected_back_distance = back_distance
                        if i < len(pose_infos):
                            chosen_width = float(pose_infos[i][0])
                            chosen_depth = float(pose_infos[i][2])
                        else:
                            chosen_width = 0.04
                            chosen_depth = 0.03
                        rospy.loginfo(
                            "📍 选中候选位姿诊断: pose=%d, target=(%.3f, %.3f, %.3f), "
                            "pregrasp=(%.3f, %.3f, %.3f), observe=(%.3f, %.3f, %.3f), "
                            "width=%.3f, depth=%.3f",
                            i + 1,
                            best_target_pose.position.x,
                            best_target_pose.position.y,
                            best_target_pose.position.z,
                            best_pre_grasp_pose.position.x,
                            best_pre_grasp_pose.position.y,
                            best_pre_grasp_pose.position.z,
                            best_observation_pose.position.x,
                            best_observation_pose.position.y,
                            best_observation_pose.position.z,
                            chosen_width,
                            chosen_depth,
                        )
                        config_success = True
                        break

                    if config_success:
                        success = True
                        break

                    rospy.logwarn(
                        f"⚠️ 姿态 {i+1} 在 back_distance={back_distance:.3f} m, mode={orientation_label} 时观察位不可达，尝试其他构型。"
                    )

                if search_timeout_hit:
                    break

                if success:
                    break

            if search_timeout_hit:
                break

            if success:
                break

            rospy.logwarn(
                f"❌ 姿态 {i+1} 未找到可达观察位/规划路径，被淘汰"
                "（可能是 IK 不可达、关节限制或碰撞约束）。"
            )

        if best_plan is None:
            rospy.logwarn("⚠️ 强约束抓取搜索失败，启动宽松回退搜索（关闭货架法向强制 + 使用轴映射/level 姿态）。")
            relaxed_back_candidates = list(self.observation_back_distance_candidates)
            for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
                if (time.time() - search_start_wall) > self.max_observation_search_time:
                    search_timeout_hit = True
                    break
                ps = PoseStamped()
                ps.header = self.grasp_pose_array_received.header
                ps.pose = raw_pose

                target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
                if target_pose_stamped is None:
                    continue

                q_orig = [
                    target_pose_stamped.pose.orientation.x,
                    target_pose_stamped.pose.orientation.y,
                    target_pose_stamped.pose.orientation.z,
                    target_pose_stamped.pose.orientation.w,
                ]
                mat_orig = tft.quaternion_matrix(q_orig)
                raw_approach_axis, raw_open_axis = self.get_configured_grasp_pose_axes(mat_orig)
                if raw_approach_axis is None:
                    continue
                flat_approach = np.array([raw_approach_axis[0], raw_approach_axis[1], 0.0])
                norm = np.linalg.norm(flat_approach)
                if norm < 0.001:
                    continue
                robot_z_relaxed = flat_approach / norm

                if target_pose_stamped.pose.position.z < 0.08:
                    target_pose_stamped.pose.position.z = 0.08
                target_pose_stamped.pose = self.correct_target_height_from_gazebo(
                    target_pose_stamped.pose,
                    label=f"回退姿态{i+1}",
                )

                seed_q = list(seed_q_global) if seed_q_global is not None else None

                orientation_options = self.build_grasp_orientation_options(
                    robot_z_relaxed,
                    raw_open_axis,
                    q_orig,
                    include_raw=True,
                )

                fallback_success = False
                for back_distance in relaxed_back_candidates:
                    if (time.time() - search_start_wall) > self.max_observation_search_time:
                        search_timeout_hit = True
                        break

                    for orientation_label, q_candidate in orientation_options:
                        if (time.time() - search_start_wall) > self.max_observation_search_time:
                            search_timeout_hit = True
                            break

                        candidate_target_pose = copy.deepcopy(target_pose_stamped.pose)
                        candidate_target_pose.orientation.x = q_candidate[0]
                        candidate_target_pose.orientation.y = q_candidate[1]
                        candidate_target_pose.orientation.z = q_candidate[2]
                        candidate_target_pose.orientation.w = q_candidate[3]

                        pre_grasp_pose = copy.deepcopy(candidate_target_pose)
                        pre_grasp_pose.position.x -= robot_z_relaxed[0] * back_distance
                        pre_grasp_pose.position.y -= robot_z_relaxed[1] * back_distance
                        pre_grasp_pose.position.z -= robot_z_relaxed[2] * back_distance

                        observation_pose = copy.deepcopy(pre_grasp_pose)
                        observation_pose.position.x -= robot_z_relaxed[0] * self.wrist_observation_backoff
                        observation_pose.position.y -= robot_z_relaxed[1] * self.wrist_observation_backoff
                        observation_pose.position.z -= robot_z_relaxed[2] * self.wrist_observation_backoff

                        trial_success = False
                        for obs_lift in self.observation_lift_candidates:
                            observation_pose_trial = copy.deepcopy(observation_pose)
                            observation_pose_trial.position.z += float(obs_lift)

                            planned_observation_pose, plan, observation_plan_label, observation_status = (
                                self.plan_observation_pose_with_relaxed_orientation(
                                    observation_pose_trial,
                                    orientation_label,
                                    robot_z_relaxed,
                                    raw_open_axis,
                                    q_orig,
                                    seed_q,
                                    planning_frame,
                                )
                            )
                            if plan is None:
                                log_fn = rospy.logwarn if observation_status == "planning_failed" else rospy.logdebug
                                log_fn(
                                    f"⚠️ 回退搜索 IK 失败: 姿态 {i+1}, mode={orientation_label}, "
                                    f"back_distance={back_distance:.3f} m, obs_lift={obs_lift:.3f} m, "
                                    f"status={observation_status}"
                                )
                                continue

                            observation_note = (
                                ""
                                if observation_plan_label == str(orientation_label)
                                else f"obs_mode={observation_plan_label}, "
                            )
                            rospy.loginfo(
                                f"✅ 宽松回退搜索成功: 姿态 {i+1}, mode={orientation_label}, "
                                f"{observation_note}"
                                f"axis={self.grasp_pose_approach_axis_label}, open={self.grasp_pose_open_axis_label}, "
                                f"insert_axis=({robot_z_relaxed[0]:.3f},{robot_z_relaxed[1]:.3f},{robot_z_relaxed[2]:.3f}), "
                                f"back_distance={back_distance:.3f} m, obs_lift={obs_lift:.3f} m"
                            )
                            stored_target_pose, stored_pre_grasp_pose, retained_lift = (
                                self.build_lift_retained_grasp_poses(
                                    candidate_target_pose,
                                    pre_grasp_pose,
                                    obs_lift,
                                )
                            )
                            if retained_lift > 1e-4:
                                rospy.loginfo(
                                    "📐 保留观察抬高量到预抓取/插入高度: retained_lift=%.3f m",
                                    retained_lift,
                                )
                            best_plan = plan
                            best_pre_grasp_pose = stored_pre_grasp_pose
                            best_target_pose = stored_target_pose
                            best_observation_pose = planned_observation_pose
                            best_robot_z = robot_z_relaxed
                            selected_back_distance = back_distance
                            if i < len(pose_infos):
                                chosen_width = float(pose_infos[i][0])
                                chosen_depth = float(pose_infos[i][2])
                            else:
                                chosen_width = 0.04
                                chosen_depth = 0.03
                            rospy.loginfo(
                                "📍 回退选中候选位姿诊断: pose=%d, target=(%.3f, %.3f, %.3f), "
                                "pregrasp=(%.3f, %.3f, %.3f), observe=(%.3f, %.3f, %.3f), "
                                "width=%.3f, depth=%.3f",
                                i + 1,
                                best_target_pose.position.x,
                                best_target_pose.position.y,
                                best_target_pose.position.z,
                                best_pre_grasp_pose.position.x,
                                best_pre_grasp_pose.position.y,
                                best_pre_grasp_pose.position.z,
                                best_observation_pose.position.x,
                                best_observation_pose.position.y,
                                best_observation_pose.position.z,
                                chosen_width,
                                chosen_depth,
                            )
                            trial_success = True
                            break

                        if trial_success:
                            fallback_success = True
                            break

                        rospy.logwarn(
                            f"⚠️ 回退搜索失败: 姿态 {i+1}, back_distance={back_distance:.3f} m, "
                            f"mode={orientation_label}"
                        )

                    if search_timeout_hit:
                        break

                    if fallback_success:
                        break

                if search_timeout_hit:
                    break

                if fallback_success:
                    break

        if best_plan is None:
            direct_fallback = self.plan_direct_grasp_fallback(
                planning_frame,
                pose_infos,
                seed_q_global,
            )
            if direct_fallback is not None:
                best_plan = direct_fallback["plan"]
                best_pre_grasp_pose = direct_fallback["pre_grasp_pose"]
                best_target_pose = direct_fallback["target_pose"]
                best_observation_pose = direct_fallback["observation_pose"]
                best_robot_z = direct_fallback["robot_z"]
                selected_back_distance = direct_fallback["back_distance"]
                chosen_width = direct_fallback["width"]
                chosen_depth = direct_fallback["depth"]
                direct_grasp_fallback_used = True

        if best_plan is None:
            self.arm.move_group.set_planning_time(self.default_planning_time)
            self.arm.move_group.set_num_planning_attempts(self.default_planning_attempts)
            if search_timeout_hit:
                rospy.logwarn(
                    "⚠️ 观察位搜索达到时间上限 %.1f s，提前结束当前目标抓取搜索。",
                    self.max_observation_search_time,
                )
            rospy.logerr("所有候选姿态都不可达！")
            self.log_workspace_diagnostics(pose_infos, planning_frame)
            self.grasp_pose_array_received = None
            self.set_failure_reason("observation_unreachable", "all_candidates_unreachable")
            return False

        self.arm.move_group.set_planning_time(self.default_planning_time)
        self.arm.move_group.set_num_planning_attempts(self.default_planning_attempts)
        self.detach_grasped_object()
        open_ok = self.gripper.open(width=0.08)
        self.log_gripper_width_snapshot("开爪后")
        if not open_ok:
            rospy.logwarn("🛑 夹爪张开失败，停止本次抓取，避免在未知夹爪状态下继续前进。")
            self.set_failure_reason("gripper_open_failed", "open_before_approach_failed")
            return False
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

        if direct_grasp_fallback_used:
            return self.execute_direct_grasp_from_current_pose(
                best_target_pose,
                best_robot_z,
                chosen_width,
                chosen_depth,
                direct_insert_distance=selected_back_distance,
            )

        wrist_refine_bbox = None
        pregrasp_reached_from_observation = False
        final_wrist_alignment_done = False

        if self.wrist_alignment_before_pregrasp:
            observation_distance = float(selected_back_distance) + max(
                0.0,
                float(self.wrist_observation_backoff),
            )
            rospy.loginfo(
                "[4.2/6] 📸 在观察位执行腕部相机对齐（宽视野，距离目标约 %.3f m）...",
                observation_distance,
            )
            ok_to_continue, wrist_refine_bbox = self.run_wrist_alignment_stage(
                "[4.2/6] 观察位腕部对齐"
            )
            if not ok_to_continue:
                return False
            rospy.sleep(0.2)
            self.log_gripper_width_snapshot("观察位腕部对齐后")

        # 观察位完成横向对齐后，再推进到真正预抓取位。
        if self.wrist_observation_backoff > 1e-4:
            if self.wrist_alignment_before_pregrasp:
                rospy.loginfo("[4.3/6] 从观察位推进到预抓取位（VLM 已在宽视野完成对齐）...")
            else:
                rospy.loginfo("[4.3/6] 从观察位推进到预抓取位（靠近目标，准备腕部 VLM 对准）...")
            if not self.segmented_cartesian_move(
                dx=best_robot_z[0] * self.wrist_observation_backoff,
                dy=best_robot_z[1] * self.wrist_observation_backoff,
                dz=best_robot_z[2] * self.wrist_observation_backoff,
                description="[4.3/6] 靠近到预抓取位",
                step_size=0.02,
                min_step=0.01,
                avoid_collisions=False,
            ):
                rospy.logwarn("🛑 无法从观察位推进到预抓取位，放弃本次抓取。")
                self.set_failure_reason("observation_unreachable", "advance_to_pregrasp_failed")
                return False
            rospy.sleep(0.3)
            pregrasp_reached_from_observation = True

        if not self.wrist_alignment_before_pregrasp:
            rospy.loginfo(
                "[4.5/6] 📸 启动腕部相机，执行抓前对齐（预抓取位，距离目标 %.3f m）...",
                selected_back_distance,
            )
            ok_to_continue, wrist_refine_bbox = self.run_wrist_alignment_stage(
                "[4.5/6] 预抓取位腕部对齐"
            )
            if not ok_to_continue:
                return False
            rospy.sleep(0.5)
            self.log_gripper_width_snapshot("腕部对齐后")
        else:
            self.log_gripper_width_snapshot("预抓取位")

        if self.use_final_wrist_alignment:
            rospy.loginfo("[4.9/6] 📸 在预抓取位做最终腕部对齐（允许图像上下偏差映射到末端 Z 修正）...")
            ok_to_continue, final_bbox = self.run_wrist_alignment_stage(
                "[4.9/6] 预抓取位最终腕部对齐"
            )
            if not ok_to_continue:
                return False
            if final_bbox is not None:
                wrist_refine_bbox = final_bbox
            final_wrist_alignment_done = True
            rospy.sleep(0.2)
            self.log_gripper_width_snapshot("最终腕部对齐后")

        if not self.apply_final_preinsert_world_z_lift():
            return False

        if self.simple_front_grasp_mode:
            return self.execute_simple_front_grasp(
                best_target_pose,
                best_robot_z,
                selected_back_distance,
                chosen_width,
                chosen_depth,
            )

        wrist_refined = self.refine_grasp_with_wrist_graspnet(
            planning_frame,
            selected_back_distance,
            bbox_hint=wrist_refine_bbox,
            seed_q=seed_q_global,
        )
        if wrist_refined is not None:
            best_target_pose = wrist_refined["target_pose"]
            best_pre_grasp_pose = wrist_refined["pre_grasp_pose"]
            best_observation_pose = wrist_refined["observation_pose"]
            best_robot_z = wrist_refined["robot_z"]
            selected_back_distance = wrist_refined["back_distance"]
            chosen_width = wrist_refined["width"]
            chosen_depth = wrist_refined["depth"]
            pregrasp_reached_from_observation = False
            self.publish_active_grasp_target(
                best_target_pose,
                best_robot_z,
                selected_back_distance,
                width_hint=chosen_width,
                depth_hint=chosen_depth,
                pre_grasp_pose=best_pre_grasp_pose,
                observation_pose=best_observation_pose,
            )
        # =========================================================================

        if self.wrist_observation_backoff > 1e-4 and not pregrasp_reached_from_observation:
            rospy.loginfo("[4.8/6] 从观察位推进到真正的预抓取位...")
            pregrasp_delta = np.array(
                [
                    best_pre_grasp_pose.position.x - best_observation_pose.position.x,
                    best_pre_grasp_pose.position.y - best_observation_pose.position.y,
                    best_pre_grasp_pose.position.z - best_observation_pose.position.z,
                ],
                dtype=np.float64,
            )
            nominal_forward_delta = np.array(best_robot_z, dtype=np.float64) * float(self.wrist_observation_backoff)
            if np.linalg.norm(pregrasp_delta) < 1e-6:
                pregrasp_delta = nominal_forward_delta
            if abs(float(pregrasp_delta[2])) > 1e-4:
                rospy.loginfo(
                    "[4.8/6] 观察位包含额外抬高量，回到预抓取位时同步回收高度: "
                    f"dz={float(pregrasp_delta[2]):.4f} m"
                )

            current_pose_for_pregrasp = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
            pregrasp_target_from_current = copy.deepcopy(current_pose_for_pregrasp)
            pregrasp_target_from_current.position.x += float(pregrasp_delta[0])
            pregrasp_target_from_current.position.y += float(pregrasp_delta[1])
            pregrasp_target_from_current.position.z += float(pregrasp_delta[2])
            pregrasp_target_from_current.orientation = copy.deepcopy(best_pre_grasp_pose.orientation)

            pregrasp_reached = False
            if not self.orientations_match(
                current_pose_for_pregrasp.orientation,
                best_pre_grasp_pose.orientation,
                tol_deg=3.0,
            ):
                rospy.loginfo("[4.8/6] 观察位使用临时姿态，先在后方切回最终抓取姿态...")
                if not self.align_current_pose_orientation(
                    best_pre_grasp_pose.orientation,
                    "[4.8/6] 后方切回最终抓取姿态",
                    retries=2,
                ):
                    rospy.logwarn("⚠️ 后方原地切姿态失败，尝试直接规划到真实预抓取位。")
                    pregrasp_reached = self.execute_pose_goal(
                        pregrasp_target_from_current,
                        "[4.8/6] 直接规划到真实预抓取位",
                        retries=2,
                        orientation_path_lock=copy.deepcopy(best_pre_grasp_pose.orientation),
                        preferred_seed=self.get_current_joint_seed(),
                    )
                else:
                    current_pose_for_pregrasp = copy.deepcopy(self.arm.move_group.get_current_pose().pose)

            if not pregrasp_reached:
                if not self.segmented_cartesian_move(
                    dx=float(pregrasp_delta[0]),
                    dy=float(pregrasp_delta[1]),
                    dz=float(pregrasp_delta[2]),
                    description="[4.8/6] 回到真实预抓取位",
                    step_size=min(0.02, max(float(np.linalg.norm(pregrasp_delta)), self.wrist_observation_backoff)),
                    min_step=0.01,
                    avoid_collisions=False
                ):
                    rospy.logwarn("⚠️ 笛卡尔回到预抓取位失败，改用位姿规划补齐。")
                    pregrasp_reached = self.execute_pose_goal(
                        pregrasp_target_from_current,
                        "[4.8/6] 位姿规划补齐真实预抓取位",
                        retries=2,
                        orientation_path_lock=copy.deepcopy(best_pre_grasp_pose.orientation),
                        preferred_seed=self.get_current_joint_seed(),
                    )
                else:
                    pregrasp_reached = True

            if not pregrasp_reached:
                rospy.logwarn("🛑 无法切换到真实预抓取位，禁止继续前插。")
                self.set_failure_reason("pregrasp_unreachable", "pregrasp_transition_failed")
                return False
            rospy.sleep(0.3)
        self.log_gripper_width_snapshot("预抓取位")

        if self.use_final_wrist_alignment and not final_wrist_alignment_done:
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

        if not self.apply_final_preinsert_world_z_lift():
            return False
        
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
        fallback_insert_distance = total_insert_distance
        total_insert_distance = self.compute_insert_distance_from_wrist(
            chosen_depth=chosen_depth,
            fallback_distance=fallback_insert_distance,
        )
        total_insert_distance = self.cap_final_insert_distance(
            total_insert_distance,
            selected_back_distance,
        )
        rospy.loginfo(
            f"[5/6] 插入距离规划: extra_depth={extra_depth:.3f} m, "
            f"fallback_total={fallback_insert_distance:.3f} m, "
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

        insert_attempts = [
            ("primary", float(total_insert_distance), float(self.insert_step_size)),
        ]
        fallback_insert_distance = max(
            self.min_insert_distance,
            min(float(total_insert_distance) * 0.72, float(total_insert_distance) - 0.02),
        )
        if fallback_insert_distance + 1e-4 < float(total_insert_distance):
            insert_attempts.append(("short_fallback", float(fallback_insert_distance), min(0.015, float(self.insert_step_size))))

        executed_insert_distance = None
        for attempt_label, attempt_distance, attempt_step in insert_attempts:
            in_dx_try = best_robot_z[0] * attempt_distance
            in_dy_try = best_robot_z[1] * attempt_distance
            in_dz_try = best_robot_z[2] * attempt_distance
            desc = "[5/6] 水平滑入柜子" if attempt_label == "primary" else f"[5/6] 水平滑入柜子（{attempt_label}）"
            if self.segmented_cartesian_move(
                dx=in_dx_try,
                dy=in_dy_try,
                dz=in_dz_try,
                description=desc,
                step_size=attempt_step,
                min_step=0.01,
                avoid_collisions=False
            ):
                executed_insert_distance = float(attempt_distance)
                if attempt_label != "primary":
                    rospy.logwarn(
                        "⚠️ 主插入失败后，短距回退插入成功：distance=%.3f m, step=%.3f m",
                        executed_insert_distance,
                        float(attempt_step),
                    )
                break
            rospy.logwarn(
                "⚠️ 插入尝试失败: mode=%s, distance=%.3f m, step=%.3f m",
                attempt_label,
                float(attempt_distance),
                float(attempt_step),
            )

        if executed_insert_distance is None:
            self.log_gripper_width_snapshot("插入失败后")
            rospy.logwarn("🛑 拦截：直线插入控制失败！")
            self.set_failure_reason("insert_failed", "cartesian_insert_failed")
            return False

        in_dx = best_robot_z[0] * executed_insert_distance
        in_dy = best_robot_z[1] * executed_insert_distance
        in_dz = best_robot_z[2] * executed_insert_distance

        rospy.sleep(0.5)
        self.log_gripper_width_snapshot("闭爪前")
        functional_tracking_ready = self.begin_gazebo_functional_grasp_tracking(best_target_pose)

        command_width, lightly_shrunk_width = self.compute_command_grasp_width(chosen_width)
        rospy.loginfo(
            f"[6/6] 闭合夹爪 (预测宽度: {chosen_width:.3f}m, "
            f"收紧边距: {self.last_grasp_width_margin_used:.3f}m, "
            f"收紧后: {lightly_shrunk_width:.3f}m, 下发宽度: {command_width:.3f}m)"
        )
        close_ok = self.gripper.close(width=command_width, force=self.grasp_force)
        empty_close_after_first_attempt = False
        if not close_ok:
            observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
            if observed_width is not None and float(observed_width) <= self.grasp_empty_close_threshold:
                empty_close_after_first_attempt = True
                rospy.logwarn(
                    "⚠️ 首次闭爪后夹爪几乎完全闭合(observed=%.3f m <= %.3f m)，"
                    "判定为空夹；跳过同位窄宽度重试，避免闭合状态下再次 grasp 导致回张。",
                    float(observed_width),
                    self.grasp_empty_close_threshold,
                )
            else:
                retry_width = float(
                    np.clip(command_width - 0.006, self.grasp_command_width_min, self.grasp_command_width_max)
                )
                if retry_width + 1e-4 < command_width:
                    rospy.logwarn(
                        "⚠️ 首次闭爪失败，尝试窄宽度重试: %.3f -> %.3f m",
                        command_width,
                        retry_width,
                    )
                    close_ok = self.gripper.close(width=retry_width, force=self.grasp_force)
                    if close_ok:
                        command_width = retry_width
        hold_check_width = max(float(chosen_width), float(command_width))
        provisional_functional_grasp = False
        if not close_ok:
            if empty_close_after_first_attempt:
                rospy.logwarn("🛑 首次闭爪已判定为空夹，直接结束当前抓取。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("grasp_failed", "gripper_empty_close")
                return False
            if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
                observed_text = "unknown" if observed_width is None else f"{float(observed_width):.3f}"
                rospy.logwarn(
                    "⚠️ 夹爪动作返回失败，但当前观测宽度=%s m。"
                    "继续执行撤出，并使用 Gazebo 功能性抓取校验决定是否视为成功。",
                    observed_text,
                )
                provisional_functional_grasp = True
            else:
                rospy.logwarn("🛑 夹爪闭合动作失败，放弃附着碰撞体。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("grasp_failed", "gripper_close_failed")
                return False
        close_width_ok = True
        if self.force_grasp_verify_after_close:
            close_width_ok = self.verify_force_grasp_hold("闭爪后", hold_check_width)
            if not close_width_ok:
                if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                    rospy.logwarn(
                        "⚠️ 闭爪后宽度校验未通过，继续执行撤出，并使用 Gazebo 功能性抓取校验决定是否视为成功。"
                    )
                    provisional_functional_grasp = True
                else:
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("grasp_failed", "force_grasp_unstable_after_close")
                    return False
        remaining_insert_distance = float(executed_insert_distance)
        probe_distance = self.compute_grasp_probe_distance(executed_insert_distance)
        if functional_tracking_ready and probe_distance > 1e-6:
            probe_dx = best_robot_z[0] * probe_distance
            probe_dy = best_robot_z[1] * probe_distance
            probe_dz = best_robot_z[2] * probe_distance
            probe_verified = False
            for probe_attempt_idx in range(2):
                probe_label = "短退后" if probe_attempt_idx == 0 else "二次短退后"
                probe_desc = "抓后短退探测" if probe_attempt_idx == 0 else "二次收紧后短退探测"
                if not self.segmented_cartesian_move(
                    dx=-probe_dx,
                    dy=-probe_dy,
                    dz=-probe_dz,
                    description=probe_desc,
                    step_size=min(self.retreat_step_size, max(0.010, probe_distance)),
                    min_step=0.005,
                    avoid_collisions=False
                ):
                    rospy.logwarn("🛑 %s 失败。", probe_desc)
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("retreat_failed", "probe_retreat_failed")
                    return False
                if self.grasp_probe_wait_sec > 1e-4:
                    rospy.sleep(self.grasp_probe_wait_sec)
                gazebo_probe_ok = self.verify_gazebo_functional_grasp(
                    probe_label,
                    min_object_motion=self.gazebo_functional_grasp_probe_min_object_motion,
                    min_follow_ratio=self.gazebo_functional_grasp_probe_min_follow_ratio,
                )
                if gazebo_probe_ok is not False:
                    remaining_insert_distance = max(0.0, float(executed_insert_distance) - probe_distance)
                    probe_verified = True
                    break
                if not self.enable_grasp_resqueeze_retry or probe_attempt_idx > 0:
                    break
                if not self.segmented_cartesian_move(
                    dx=probe_dx,
                    dy=probe_dy,
                    dz=probe_dz,
                    description="返回插入位准备二次收紧",
                    step_size=min(self.insert_step_size, max(0.010, probe_distance)),
                    min_step=0.005,
                    avoid_collisions=False
                ):
                    rospy.logwarn("🛑 返回插入位准备二次收紧失败。")
                    self.clear_functional_grasp_candidate()
                    self.set_failure_reason("grasp_failed", "resqueeze_reapproach_failed")
                    return False
                functional_tracking_ready = self.begin_gazebo_functional_grasp_tracking(best_target_pose)
                resqueeze_width = self.compute_resqueeze_width(command_width)
                if resqueeze_width + 1e-4 >= command_width:
                    rospy.logwarn(
                        "⚠️ 当前下发宽度已接近下限，无法继续二次收紧: current=%.3f m, candidate=%.3f m",
                        float(command_width),
                        float(resqueeze_width),
                    )
                    break
                resqueeze_force = min(80.0, float(self.grasp_force) + max(0.0, self.grasp_resqueeze_force_boost))
                rospy.logwarn(
                    "⚠️ 抓后短退未检测到物体跟随，执行二次收紧: %.3f -> %.3f m, force=%.1f N",
                    float(command_width),
                    float(resqueeze_width),
                    float(resqueeze_force),
                )
                command_width = resqueeze_width
                hold_check_width = max(float(chosen_width), float(command_width))
                close_ok = self.gripper.close(width=command_width, force=resqueeze_force)
                if not close_ok:
                    if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                        observed_width = getattr(self.gripper, "last_observed_gripper_width", None)
                        observed_text = "unknown" if observed_width is None else f"{float(observed_width):.3f}"
                        rospy.logwarn(
                            "⚠️ 二次收紧动作返回失败，但当前观测宽度=%s m。继续使用 Gazebo 功能性抓取校验。",
                            observed_text,
                        )
                        provisional_functional_grasp = True
                    else:
                        self.clear_functional_grasp_candidate()
                        self.set_failure_reason("grasp_failed", "resqueeze_close_failed")
                        return False
                if self.force_grasp_verify_after_close:
                    close_width_ok = self.verify_force_grasp_hold("二次收紧后", hold_check_width)
                    if not close_width_ok:
                        if functional_tracking_ready and self.can_attempt_gazebo_functional_recovery(hold_check_width):
                            rospy.logwarn("⚠️ 二次收紧后宽度校验未通过，继续依赖 Gazebo 功能性抓取校验。")
                            provisional_functional_grasp = True
                        else:
                            self.clear_functional_grasp_candidate()
                            self.set_failure_reason("grasp_failed", "force_grasp_unstable_after_resqueeze")
                            return False
            if not probe_verified:
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("retreat_failed", "probe_functional_grasp_verify_failed")
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

        post_grasp_lift = max(0.0, float(self.post_grasp_lift_distance))
        if post_grasp_lift > 1e-6:
            rospy.loginfo(">>> 抓稳后先抬高 %.3f m，再水平撤出...", post_grasp_lift)
            if not self.segmented_cartesian_move(
                dx=0.0,
                dy=0.0,
                dz=post_grasp_lift,
                description="抓后抬升稳定物体",
                step_size=min(self.lift_step_size, max(0.005, post_grasp_lift)),
                min_step=0.005,
                avoid_collisions=False,
            ):
                rospy.logwarn("🛑 抓后抬升失败，停止撤出以避免拖拽物体或撞击货架。")
                self.clear_functional_grasp_candidate()
                self.set_failure_reason("retreat_failed", "post_grasp_lift_failed")
                return False
            rospy.sleep(0.2)

        rospy.loginfo(">>> 保持水平原路撤出...")
        remaining_dx = best_robot_z[0] * remaining_insert_distance
        remaining_dy = best_robot_z[1] * remaining_insert_distance
        remaining_dz = best_robot_z[2] * remaining_insert_distance
        if not self.segmented_cartesian_move(
            dx=-remaining_dx,
            dy=-remaining_dy,
            dz=-remaining_dz,
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

        gazebo_functional_ok = self.verify_gazebo_functional_grasp("撤出后")
        retreat_width_ok = True
        if self.force_grasp_verify_after_retreat:
            retreat_width_ok = self.verify_force_grasp_hold("撤出后", hold_check_width, wait_sec=0.10)
        if gazebo_functional_ok is False:
            self.clear_functional_grasp_candidate()
            self.set_failure_reason("retreat_failed", "gazebo_functional_grasp_verify_failed")
            return False
        if gazebo_functional_ok is None and self.force_grasp_verify_after_retreat and not retreat_width_ok:
            self.clear_functional_grasp_candidate()
            self.set_failure_reason("retreat_failed", "force_grasp_lost_during_retreat")
            return False
        if gazebo_functional_ok is True and provisional_functional_grasp:
            rospy.loginfo("✅ Gazebo 功能性抓取校验通过，接受本次动作级失败后的功能性抓取成功。")

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
        release_z = basket_top_z + self.place_release_height_above_basket
        transport_clearance = max(
            self.place_hover_height,
            self.place_release_height_above_basket + 0.05,
            item_half_height + 0.015,
        )
        transit_z = basket_top_z + transport_clearance
        slot_x, slot_y = self.get_next_basket_slot()
        place_orientation = self.get_place_transport_orientation(current_pose)

        if self.place_orientation_mode == "downward":
            rospy.loginfo("🧭 放置阶段姿态模式: downward（末端切到固定朝下姿态后再搬运）。")
        else:
            rospy.loginfo("🧭 放置阶段姿态模式: preserve（锁定放置入口时的世界姿态并贯穿后续动作）。")
        rospy.loginfo(
            "🧺 放置高度策略: basket_top_z=%.3f m, release_z=%.3f m（篮子上方 %.3f m）, transit_z=%.3f m",
            float(basket_top_z),
            float(release_z),
            float(self.place_release_height_above_basket),
            float(transit_z),
        )

        rospy.loginfo("[放置 1/4] 必要时调整到放置高度...")
        if self.place_orientation_mode == "preserve":
            pre_align_orientation = copy.deepcopy(place_orientation)
        else:
            pre_align_orientation = copy.deepcopy(current_pose.orientation)
        transport_z = float(max(current_pose.position.z, transit_z))
        if current_pose.position.z + 0.003 < transit_z:
            if not self.move_vertical_to_z(
                transit_z,
                "放置前高度调整",
                orientation=pre_align_orientation,
                allow_position_only_fallback=self.place_allow_position_only_fallback,
            ):
                rospy.logwarn("⚠️ 无法调整到放置高度，放置中止。")
                self.set_failure_reason("place_transit_failed", "pre_place_height_adjust_failed")
                return False
        else:
            rospy.loginfo(
                "放置前高度调整：当前高度 %.3f m 已高于安全搬运高度 %.3f m，"
                "保持当前高度先平移，避免在货架旁做多余下降。",
                float(current_pose.position.z),
                float(transit_z),
            )

        current_pose = copy.deepcopy(self.arm.move_group.get_current_pose().pose)

        if self.place_orientation_mode == "downward":
            if not self.orientations_match(current_pose.orientation, place_orientation, tol_deg=3.0):
                rospy.loginfo("[放置 1.5/4] 切换到固定放置姿态...")
                if not self.align_current_pose_orientation(
                    place_orientation,
                    "切换到固定放置姿态",
                    retries=2,
                ):
                    rospy.logwarn("⚠️ 无法切换到固定放置姿态，放置中止。")
                    self.set_failure_reason("place_transit_failed", "align_place_orientation_failed")
                    return False

        place_orientation_for_transport = copy.deepcopy(place_orientation)
        rospy.loginfo(
            f"[放置 2/4] 平移到篮子上方: x={slot_x:.2f}, "
            f"y={slot_y:.2f}, z={transport_z:.2f}"
        )
        if not self.move_horizontal_to_xy(
            slot_x,
            slot_y,
            "移动到篮子上方安全位",
            orientation=place_orientation_for_transport,
            allow_position_only_fallback=self.place_allow_position_only_fallback,
        ):
            rospy.logwarn("⚠️ 到达篮子上方失败，放置中止。")
            self.set_failure_reason("place_transit_failed", "move_over_basket_failed")
            return False

        rospy.loginfo("[放置 3/4] 保持末端姿态，竖直下降到篮子上方 %.3f m 释放高度...", self.place_release_height_above_basket)
        if not self.move_vertical_to_z(
            release_z,
            "下放到篮子上方固定释放高度",
            orientation=place_orientation_for_transport,
            allow_position_only_fallback=self.place_allow_position_only_fallback,
        ):
            rospy.logwarn("⚠️ 下放到篮子上方固定释放高度失败，放置中止。")
            self.set_failure_reason("place_drop_failed", "descend_to_release_failed")
            return False

        rospy.loginfo("[放置 4/4] 张开夹爪，释放物体。")
        if not self.release_grasped_item():
            return False

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
            estimated_seed_joint_cost = self.estimate_task_joint_cost()
            self.start_task_joint_cost_tracking()

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
            task_joint_cost, task_joint_l1_cost = self.stop_task_joint_cost_tracking()
            if task_joint_cost is not None:
                rospy.loginfo(
                    f"📊 本次任务统计: task_joint_cost={task_joint_cost:.4f} rad, "
                    f"task_joint_l1={task_joint_l1_cost:.4f} rad, "
                    f"seed_estimate={estimated_seed_joint_cost if estimated_seed_joint_cost is not None else -1.0:.4f} rad, "
                    f"task_time={task_elapsed:.3f} s"
                )
            else:
                rospy.loginfo(f"📊 本次任务统计: task_time={task_elapsed:.3f} s")
            if not task_success or self.stop_requested:
                rospy.loginfo(">>> 任务失败，回到 Home 保证安全。")
                self.detach_gazebo_object_if_needed()
                self.detach_grasped_object()
                self.clear_functional_grasp_candidate()
                self.clear_active_grasp_target()
                self.current_item_collision_size = None
                self.arm.go_to_home()
            rospy.loginfo(">>> 已复位，准备接收下一任务或等待调度器通知完成...")
            task_status = "DONE" if task_success else "FAILED"
            self.publish_task_metrics(
                task_status,
                task_joint_cost,
                task_elapsed,
                estimated_seed_joint_cost=estimated_seed_joint_cost,
                task_joint_l1_cost=task_joint_l1_cost,
            )
            self.status_pub.publish(task_status)

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
