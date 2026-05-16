#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np
import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped


class ExecutionIKProjector:
    """Build the same observation-pose IK targets used by demo.py before VLM scoring."""

    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.force_shelf_normal_approach = bool(rospy.get_param("~force_shelf_normal_approach", True))
        self.shelf_pose_fallback_yaw = float(rospy.get_param("~shelf_pose_fallback_yaw", 0.0))
        self.wrist_observation_backoff = float(rospy.get_param("~wrist_observation_backoff", 0.04))
        self.pre_grasp_back_distance_candidates = [
            float(v) for v in rospy.get_param("~pre_grasp_back_distance_candidates", [0.12, 0.15, 0.18])
        ]
        self.observation_back_distance_candidates = sorted(
            set([0.08, 0.10] + [float(v) for v in self.pre_grasp_back_distance_candidates])
        )
        self.observation_lift_candidates = [
            float(v) for v in rospy.get_param("~observation_lift_candidates", [0.0, 0.04])
        ]
        self.grasp_roll_variants = [
            str(v) for v in rospy.get_param("~grasp_roll_variants", ["x_up", "x_down"])
        ]
        self.enforce_horizontal_gripper_open_axis = bool(
            rospy.get_param("~enforce_horizontal_gripper_open_axis", True)
        )
        self.max_gripper_open_axis_vertical_component = float(
            rospy.get_param("~max_gripper_open_axis_vertical_component", 0.25)
        )
        self.grasp_pose_approach_axis, self.grasp_pose_approach_sign = self.parse_axis_param(
            rospy.get_param("~grasp_pose_approach_axis", "+z"),
            default_axis="z",
            default_sign=1.0,
        )
        self.grasp_pose_open_axis, self.grasp_pose_open_sign = self.parse_axis_param(
            rospy.get_param("~grasp_pose_open_axis", "+x"),
            default_axis="x",
            default_sign=1.0,
        )
        self.transform_timeout_sec = float(rospy.get_param("~execution_ik_transform_timeout_sec", 0.25))
        self.max_observation_ik_attempts_per_candidate = int(
            rospy.get_param("~max_observation_ik_attempts_per_candidate", 0)
        )

    def parse_axis_param(self, value, default_axis="z", default_sign=1.0):
        text = str(value).strip().lower()
        sign = float(default_sign)
        if text.startswith("-"):
            sign = -1.0
            text = text[1:]
        elif text.startswith("+"):
            sign = 1.0
            text = text[1:]
        axis = {"x": "x", "y": "y", "z": "z", "0": "x", "1": "y", "2": "z"}.get(text)
        if axis is None:
            axis = str(default_axis).strip().lower()
            if axis not in ("x", "y", "z"):
                axis = "z"
            sign = float(default_sign)
        return axis, sign

    def transform_pose(self, pose, source_frame, target_frame):
        ps = PoseStamped()
        ps.header.frame_id = source_frame
        ps.header.stamp = rospy.Time(0)
        ps.pose = pose
        try:
            self.tf_listener.waitForTransform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout_sec),
            )
            return self.tf_listener.transformPose(target_frame, ps).pose
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "execution IK pose transform failed: %s -> %s: %s",
                source_frame,
                target_frame,
                exc,
            )
            return None

    def get_axis_vector(self, rot_mat, axis_name, axis_sign=1.0):
        axis_index = {"x": 0, "y": 1, "z": 2}.get(str(axis_name).lower())
        if axis_index is None:
            return None
        vec = float(axis_sign) * np.array(rot_mat[:3, axis_index], dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    def get_configured_grasp_pose_axes(self, rot_mat):
        approach_axis = self.get_axis_vector(
            rot_mat,
            self.grasp_pose_approach_axis,
            self.grasp_pose_approach_sign,
        )
        open_axis = self.get_axis_vector(
            rot_mat,
            self.grasp_pose_open_axis,
            self.grasp_pose_open_sign,
        )
        return approach_axis, open_axis

    def get_shelf_inward_axis_world(self):
        shelf_tf = tft.euler_matrix(0.0, 0.0, self.shelf_pose_fallback_yaw)
        inward = shelf_tf[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = np.linalg.norm(inward)
        if norm < 1e-6:
            return None
        return inward / norm

    def gripper_open_axis_is_allowed(self, q_candidate):
        if not self.enforce_horizontal_gripper_open_axis:
            return True
        rot_mat = tft.quaternion_matrix(q_candidate)
        open_axis = np.array(rot_mat[:3, 1], dtype=np.float64)
        norm = np.linalg.norm(open_axis)
        if norm < 1e-6:
            return False
        open_axis = open_axis / norm
        return abs(float(open_axis[2])) <= float(self.max_gripper_open_axis_vertical_component)

    def build_level_grasp_quaternion(self, robot_z, roll_variant="x_up"):
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

    def build_orientation_options(self, robot_z):
        options = []
        for roll_variant in self.grasp_roll_variants:
            q_level = self.build_level_grasp_quaternion(robot_z, roll_variant=roll_variant)
            if q_level is not None and self.gripper_open_axis_is_allowed(q_level):
                options.append((str(roll_variant), q_level))
        return options

    def build_observation_poses(self, raw_pose, source_frame, planning_frame):
        target_pose = self.transform_pose(raw_pose, source_frame, planning_frame)
        if target_pose is None:
            return []

        q_orig = [
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        ]
        mat_orig = tft.quaternion_matrix(q_orig)
        raw_approach_axis, _raw_open_axis = self.get_configured_grasp_pose_axes(mat_orig)
        if raw_approach_axis is None:
            return []

        flat_approach = np.array([raw_approach_axis[0], raw_approach_axis[1], 0.0], dtype=np.float64)
        norm = np.linalg.norm(flat_approach)
        if norm < 0.001:
            return []

        robot_z = flat_approach / norm
        shelf_inward_axis = self.get_shelf_inward_axis_world() if self.force_shelf_normal_approach else None
        if shelf_inward_axis is not None:
            robot_z = shelf_inward_axis.copy()
            if float(np.dot(robot_z, flat_approach)) < 0.0:
                robot_z = -robot_z

        target_pose.position.z = max(float(target_pose.position.z), 0.08)
        orientation_options = self.build_orientation_options(robot_z)
        if not orientation_options:
            return []

        projected = []
        for back_distance in self.observation_back_distance_candidates:
            for orientation_label, q_new in orientation_options:
                candidate_target = copy.deepcopy(target_pose)
                candidate_target.orientation.x = q_new[0]
                candidate_target.orientation.y = q_new[1]
                candidate_target.orientation.z = q_new[2]
                candidate_target.orientation.w = q_new[3]

                pre_grasp = copy.deepcopy(candidate_target)
                pre_grasp.position.x -= robot_z[0] * float(back_distance)
                pre_grasp.position.y -= robot_z[1] * float(back_distance)
                pre_grasp.position.z -= robot_z[2] * float(back_distance)

                observation = copy.deepcopy(pre_grasp)
                observation.position.x -= robot_z[0] * self.wrist_observation_backoff
                observation.position.y -= robot_z[1] * self.wrist_observation_backoff
                observation.position.z -= robot_z[2] * self.wrist_observation_backoff

                for obs_lift in self.observation_lift_candidates:
                    trial = copy.deepcopy(observation)
                    trial.position.z += float(obs_lift)
                    projected.append(
                        (
                            trial,
                            {
                                "mode": orientation_label,
                                "back_distance": float(back_distance),
                                "obs_lift": float(obs_lift),
                                "target_z": float(candidate_target.position.z),
                                "observe_z": float(trial.position.z),
                            },
                        )
                    )
        if self.max_observation_ik_attempts_per_candidate > 0:
            return projected[: self.max_observation_ik_attempts_per_candidate]
        return projected
