#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ik_filter.py — IK 可达性过滤节点

位于 GraspNet 和 Semantic Reranker 之间：
  GraspNet raw poses → IK filter → 仅保留机械臂可达的候选 → Semantic Reranker

动机：VLM 调用昂贵，IK 计算便宜。先过滤掉不可达候选，VLM 只在可达候选中评分，
      节省 API 调用量，且 VLM 评分更有意义。
"""

import sys
import json
import time
import threading

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Float32MultiArray
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
import moveit_commander
from sam_perception.execution_ik_utils import ExecutionIKProjector


class IKFilterNode:
    def __init__(self):
        rospy.init_node("ik_filter")

        # --- MoveIt 初始化 ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()

        connected = False
        rospy.loginfo("⏳ IK filter 等待 MoveGroup...")
        while not rospy.is_shutdown() and not connected:
            try:
                self.arm = moveit_commander.MoveGroupCommander("panda_manipulator")
                connected = True
            except RuntimeError as e:
                rospy.logwarn("IK filter MoveGroup 未就绪: %s", e)
                rospy.sleep(2.0)

        rospy.loginfo("⏳ IK filter 等待 /compute_ik 服务...")
        rospy.wait_for_service("/compute_ik")
        self.compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("✅ IK filter 就绪！")
        self.execution_ik_projector = ExecutionIKProjector()

        # --- 参数 ---
        self.pose_topic_in = rospy.get_param("~pose_topic_in", "/graspnet/grasp_pose_array_raw")
        self.info_topic_in = rospy.get_param("~info_topic_in", "/graspnet/grasp_info_raw")
        self.pose_topic_out = rospy.get_param(
            "~pose_topic_out", "/graspnet/grasp_pose_array_ik"
        )
        self.info_topic_out = rospy.get_param(
            "~info_topic_out", "/graspnet/grasp_info_ik"
        )
        self.ik_timeout = float(rospy.get_param("~ik_timeout", 0.15))
        self.min_feasible_per_object = int(rospy.get_param("~min_feasible_per_object", 1))
        self.max_feasible_per_object = max(
            self.min_feasible_per_object,
            int(rospy.get_param("~max_feasible_per_object", 6)),
        )
        self.max_ik_checks_per_object = int(rospy.get_param("~max_ik_checks_per_object", 10))
        self.assume_all_reachable = bool(rospy.get_param("~assume_all_reachable", False))
        self.publish_stats = bool(rospy.get_param("~publish_stats", True))

        # --- 同步接收 ---
        self.pending_pose_msg = None
        self.pending_info_msg = None
        self._lock = threading.Lock()

        rospy.Subscriber(self.pose_topic_in, PoseArray, self.pose_cb, queue_size=1)
        rospy.Subscriber(self.info_topic_in, Float32MultiArray, self.info_cb, queue_size=1)

        self.pub_pose = rospy.Publisher(self.pose_topic_out, PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher(self.info_topic_out, Float32MultiArray, queue_size=1)

        rospy.loginfo(
            "🔧 IK filter: %s/%s → %s/%s (execution-style observation IK, ik_timeout=%.2fs, max_feasible/object=%d, max_checks/object=%d, assume_all_reachable=%s)",
            self.pose_topic_in,
            self.info_topic_in,
            self.pose_topic_out,
            self.info_topic_out,
            self.ik_timeout,
            self.max_feasible_per_object,
            self.max_ik_checks_per_object,
            str(self.assume_all_reachable).lower(),
        )

    def pose_cb(self, msg):
        with self._lock:
            self.pending_pose_msg = msg
            self.try_process()

    def info_cb(self, msg):
        with self._lock:
            self.pending_info_msg = msg
            self.try_process()

    def try_process(self):
        if self.pending_pose_msg is None or self.pending_info_msg is None:
            return

        pose_msg = self.pending_pose_msg
        info_data = list(self.pending_info_msg.data)
        self.pending_pose_msg = None
        self.pending_info_msg = None

        try:
            out_pose, out_info = self.filter_by_ik(pose_msg, info_data)
        except Exception as exc:
            rospy.logerr("IK filter 异常，透传原始候选: %s", exc)
            out_pose = pose_msg
            out_info = info_data

        self.pub_pose.publish(out_pose)
        self.pub_info.publish(Float32MultiArray(data=out_info))

    def get_ik(self, pose, frame_id):
        """返回 (joints_list, success_bool)"""
        req = PositionIKRequest()
        req.group_name = "panda_manipulator"
        try:
            req.robot_state = self.robot.get_current_state()
        except Exception:
            pass
        req.pose_stamped = PoseStamped()
        req.pose_stamped.header.frame_id = frame_id
        req.pose_stamped.pose = pose
        req.timeout = rospy.Duration(self.ik_timeout)

        try:
            res = self.compute_ik(req)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, "IK filter /compute_ik 调用失败: %s", e)
            return None, False

        if res.error_code.val == 1:
            return list(res.solution.joint_state.position[:7]), True
        return None, False

    def get_execution_style_ik(self, raw_pose, source_frame):
        planning_frame = self.robot.get_planning_frame()
        projected = self.execution_ik_projector.build_observation_poses(
            raw_pose,
            source_frame,
            planning_frame,
        )
        for observation_pose, meta in projected:
            joints, ok = self.get_ik(observation_pose, planning_frame)
            if ok:
                return joints, True, meta
        return None, False, None

    def filter_by_ik(self, pose_msg, info_data):
        num_poses = len(pose_msg.poses)
        info_stride = (
            int(len(info_data) / num_poses)
            if num_poses > 0 and len(info_data) % num_poses == 0
            else 0
        )
        if num_poses == 0 or info_stride < 4:
            return pose_msg, info_data

        if self.assume_all_reachable:
            kept_indices = []
            per_object_counts = {}
            for idx in range(num_poses):
                base = idx * info_stride
                obj_id = int(info_data[base])
                count = per_object_counts.get(obj_id, 0)
                if count >= self.max_feasible_per_object:
                    continue
                per_object_counts[obj_id] = count + 1
                kept_indices.append(idx)

            out_pose = PoseArray()
            out_pose.header = pose_msg.header
            out_info = []
            for idx in kept_indices:
                out_pose.poses.append(pose_msg.poses[idx])
                base = idx * info_stride
                obj_id = int(info_data[base])
                width = float(info_data[base + 1])
                score = float(info_data[base + 2])
                depth = float(info_data[base + 3])
                out_info.extend([obj_id, width, score, depth, 1.0])
            rospy.loginfo(
                "🔧 IK filter 宽松透传: %d → %d 候选全部标记为可达，跳过 /compute_ik (max/object=%d)。",
                num_poses,
                len(kept_indices),
                self.max_feasible_per_object,
            )
            return out_pose, out_info

        # 解析每个候选
        object_ids = [int(info_data[i * info_stride]) for i in range(num_poses)]
        unique_objects = sorted(set(object_ids))

        # 按 object 分组跑 IK
        object_feasible = {oid: [] for oid in unique_objects}
        object_infeasible = {oid: [] for oid in unique_objects}
        ik_joint_cache = {}  # idx → joint list

        t_start = time.time()
        feasible_meta = {}
        checked_indices = set()
        checked_count = 0

        def check_candidate(idx):
            nonlocal checked_count
            if idx in checked_indices:
                return
            checked_indices.add(idx)
            checked_count += 1
            pose = pose_msg.poses[idx]
            oid = object_ids[idx]
            joints, ok, meta = self.get_execution_style_ik(pose, pose_msg.header.frame_id)
            if ok:
                object_feasible[oid].append(idx)
                ik_joint_cache[idx] = joints
                feasible_meta[idx] = meta
            else:
                object_infeasible[oid].append(idx)

        for oid in unique_objects:
            object_indices = [idx for idx, obj_id in enumerate(object_ids) if obj_id == oid]
            checks_for_object = 0
            for idx in object_indices:
                if len(object_feasible[oid]) >= self.max_feasible_per_object:
                    break
                if (
                    self.max_ik_checks_per_object > 0
                    and checks_for_object >= self.max_ik_checks_per_object
                    and len(object_feasible[oid]) >= self.min_feasible_per_object
                ):
                    break
                check_candidate(idx)
                checks_for_object += 1

            # 若快速检查没找到任何可达位，继续扫剩余候选，避免把可抓目标误判成 IK=0。
            if len(object_feasible[oid]) < self.min_feasible_per_object:
                for idx in object_indices:
                    if idx in checked_indices:
                        continue
                    check_candidate(idx)
                    if len(object_feasible[oid]) >= self.min_feasible_per_object:
                        break

            for idx in object_indices:
                if idx not in checked_indices and idx not in object_infeasible[oid]:
                    object_infeasible[oid].append(idx)

        ik_time_ms = (time.time() - t_start) * 1000.0

        # 构建输出：每个 object 保留可行候选，不可行的排到末尾
        out_indices = []
        total_raw = num_poses
        total_feasible = 0

        for oid in unique_objects:
            feasible = object_feasible[oid]
            infeasible = object_infeasible[oid]
            if len(feasible) >= self.min_feasible_per_object:
                out_indices.extend(feasible)
                total_feasible += len(feasible)
                # 不可行候选放到末尾（低优先级）
                out_indices.extend(infeasible)
            else:
                # 该目标可行候选太少，保留全部（标记为低质量）
                rospy.logwarn(
                    "IK filter: 目标 %d 仅 %d 个 IK 可达候选（阈值=%d），保留全部 %d 个",
                    oid,
                    len(feasible),
                    self.min_feasible_per_object,
                    len(feasible) + len(infeasible),
                )
                out_indices.extend(feasible)
                out_indices.extend(infeasible)
                total_feasible += len(feasible)

        # 重排 pose 和 info
        out_pose = PoseArray()
        out_pose.header = pose_msg.header
        out_info = []

        for idx in out_indices:
            out_pose.poses.append(pose_msg.poses[idx])
            base = idx * info_stride
            # 保留原有 4 个值 (object_id, width, score, depth)
            obj_id = int(info_data[base])
            width = float(info_data[base + 1])
            score = float(info_data[base + 2])
            depth = float(info_data[base + 3])
            ik_flag = 1.0 if idx in ik_joint_cache else 0.0
            out_info.extend([obj_id, width, score, depth, ik_flag])

        rospy.loginfo(
            "🔧 IK filter: %d → %d 候选 (%.1f%% 执行观察位可达), checked=%d/%d, %d objects, IK耗时 %.0f ms",
            total_raw,
            total_feasible,
            100.0 * total_feasible / max(1, total_raw),
            checked_count,
            total_raw,
            len(unique_objects),
            ik_time_ms,
        )
        if feasible_meta:
            first_idx = sorted(feasible_meta.keys())[0]
            meta = feasible_meta[first_idx] or {}
            rospy.loginfo(
                "🔧 IK filter 首个可达观察位: idx=%d mode=%s back=%.3f lift=%.3f target_z=%.3f observe_z=%.3f",
                first_idx,
                str(meta.get("mode", "?")),
                float(meta.get("back_distance", -1.0)),
                float(meta.get("obs_lift", -1.0)),
                float(meta.get("target_z", -1.0)),
                float(meta.get("observe_z", -1.0)),
            )

        return out_pose, out_info


if __name__ == "__main__":
    try:
        node = IKFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
