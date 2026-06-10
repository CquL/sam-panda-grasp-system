#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_order_scheduler.py — 随机顺序调度器（GTSP 消融实验基线）

与 gtsp_scheduler.py 的区别：
  - 不做 GA/GTSP 优化
  - 物体抓取顺序随机排列
  - 每个物体随机选取一个 IK 可达候选姿态
  - 用于对比验证 GTSP 建模的关节代价和 TCT 收益
"""

import sys
import json
import time
import random
import threading

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Float32MultiArray, String
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
import moveit_commander


class RandomOrderScheduler:
    def __init__(self):
        rospy.init_node("random_order_scheduler")

        # --- MoveIt ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        connected = False
        rospy.loginfo("⏳ 随机调度器等待 MoveGroup...")
        while not rospy.is_shutdown() and not connected:
            try:
                self.arm = moveit_commander.MoveGroupCommander("panda_manipulator")
                connected = True
            except RuntimeError as e:
                rospy.logwarn("随机调度器 MoveGroup 未就绪: %s", e)
                rospy.sleep(2.0)
        rospy.loginfo("✅ 随机调度器成功连接到 MoveGroup！")

        rospy.loginfo("⏳ 等待 /compute_ik 服务...")
        rospy.wait_for_service("/compute_ik")
        self.compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("✅ IK 服务就绪！")

        # --- 参数 ---
        self.pose_topic = rospy.get_param("~pose_topic", "/graspnet/grasp_pose_array_semantic")
        self.info_topic = rospy.get_param("~info_topic", "/graspnet/grasp_info_semantic")
        self.publish_metrics = bool(rospy.get_param("~publish_metrics", True))
        self.ik_timeout_sec = float(rospy.get_param("~ik_timeout_sec", 0.15))
        self.assume_all_reachable = bool(rospy.get_param("~assume_all_reachable", False))

        # --- 状态 ---
        self.task_queue = []
        self.is_executing = False
        self.saved_header = None
        self.current_run_metrics = None
        self.primary_pose_msg = None
        self.primary_info_msg = None
        self.pending_pose_msg = None
        self.pending_info_msg = None
        self.processing_schedule = False
        self.schedule_lock = threading.Lock()

        # --- 订阅 ---
        rospy.Subscriber(self.pose_topic, PoseArray, self.pose_cb, queue_size=1)
        rospy.Subscriber(self.info_topic, Float32MultiArray, self.info_cb, queue_size=1)
        rospy.Subscriber("/demo/task_status", String, self.status_cb, queue_size=50)
        rospy.Subscriber("/demo/task_metrics", String, self.task_metrics_cb, queue_size=50)
        rospy.Subscriber("/demo/failure_reason", String, self.failure_reason_cb, queue_size=50)

        # --- 发布 ---
        self.pub_pose = rospy.Publisher("/graspnet/grasp_pose_array", PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher("/graspnet/grasp_info", Float32MultiArray, queue_size=1)
        self.pub_demo_cmd = rospy.Publisher("/demo/command", String, queue_size=1)
        self.pub_schedule_metrics = rospy.Publisher("/scheduler/run_metrics", String, queue_size=10)

        rospy.loginfo(
            "🎲 随机顺序调度器就绪！pose_topic=%s info_topic=%s (NO GTSP optimization)",
            self.pose_topic,
            self.info_topic,
        )

    def pose_cb(self, msg):
        self.primary_pose_msg = msg
        if self.is_executing:
            self.pending_pose_msg = msg
            return
        self.try_process()

    def info_cb(self, msg):
        self.primary_info_msg = msg.data
        if self.is_executing:
            self.pending_info_msg = msg.data
            return
        self.try_process()

    def try_process(self):
        if self.is_executing or self.processing_schedule:
            return
        if self.primary_pose_msg is None or self.primary_info_msg is None:
            return
        if not self.schedule_lock.acquire(blocking=False):
            return

        self.processing_schedule = True
        pose_msg = self.primary_pose_msg
        info_values = list(self.primary_info_msg)
        self.primary_pose_msg = None
        self.primary_info_msg = None
        try:
            self.process_random_schedule(pose_msg, info_values)
        finally:
            self.processing_schedule = False
            self.schedule_lock.release()

    def get_ik(self, pose, frame_id):
        req = PositionIKRequest()
        req.group_name = "panda_manipulator"
        try:
            req.robot_state = self.robot.get_current_state()
        except Exception:
            pass
        req.pose_stamped = PoseStamped()
        req.pose_stamped.header.frame_id = frame_id
        req.pose_stamped.pose = pose
        req.timeout = rospy.Duration(self.ik_timeout_sec)
        try:
            res = self.compute_ik(req)
        except rospy.ServiceException as e:
            return None
        if res.error_code.val == 1:
            return list(res.solution.joint_state.position[:7])
        return None

    def process_random_schedule(self, pose_msg, info_values):
        schedule_start = time.time()

        num_poses = len(pose_msg.poses)
        info_stride = (
            int(len(info_values) / num_poses)
            if num_poses > 0 and len(info_values) % num_poses == 0
            else 0
        )
        if num_poses == 0 or info_stride < 4:
            rospy.logerr("随机调度器: 无效的输入 (num_poses=%d, info_stride=%d)", num_poses, info_stride)
            return

        rospy.loginfo("=" * 50)
        rospy.loginfo("🎲 随机顺序调度（无 GTSP 优化）")

        # 按 object_id 分组 + IK 过滤
        object_ids = [int(info_values[i * info_stride]) for i in range(num_poses)]
        unique_objects = sorted(set(object_ids))

        clusters = {oid: [] for oid in unique_objects}
        raw_counts = {oid: 0 for oid in unique_objects}
        for i, oid in enumerate(object_ids):
            raw_counts[oid] += 1
            pose = pose_msg.poses[i]
            base = i * info_stride
            info = info_values[base + 1 : base + 4]
            semantic_score = 0.0
            if info_stride >= 5 and (base + 4) < len(info_values):
                semantic_score = float(info_values[base + 4])
            if self.assume_all_reachable:
                q = [0.0] * 7  # dummy joint values
            else:
                q = self.get_ik(pose, pose_msg.header.frame_id)
            if q is not None:
                clusters[oid].append((pose, list(info), q, semantic_score))

        # 随机排列 + 每个物体随机选一个姿态
        order = list(unique_objects)
        random.shuffle(order)
        rospy.loginfo("🎲 随机抓取顺序: %s", str(order))

        self.task_queue = []
        for oid in order:
            feasible = clusters.get(oid, [])
            if not feasible:
                rospy.logwarn("  ↳ 物体 %d: 无 IK 可达候选，跳过", oid)
                continue
            chosen = random.choice(feasible)
            pose, info, q, sem_score = chosen
            rospy.loginfo(
                "  ↳ 物体 %d: %d 个候选中随机选 1 个 (semantic_score=%.2f)",
                oid,
                len(feasible),
                sem_score,
            )
            self.task_queue.append((pose, info, q, oid))

        if not self.task_queue:
            rospy.logwarn("🎲 随机调度: 无有效任务可派发")
            return

        scheduler_time = time.time() - schedule_start
        rospy.loginfo(
            "🎲 随机调度完成: %d 个任务 (order=%s), 耗时 %.3f s",
            len(self.task_queue),
            str([t[3] for t in self.task_queue]),
            scheduler_time,
        )

        self.current_run_metrics = {
            "task_count": len(self.task_queue),
            "cluster_count": len(unique_objects),
            "candidate_pose_count": num_poses,
            "joint_cost_rad": 0.0,
            "planned_joint_cost_rad": 0.0,
            "executed_joint_cost_rad": 0.0,
            "executed_task_count": 0,
            "scheduler_time_sec": float(scheduler_time),
            "run_start_sim_time": rospy.Time.now().to_sec(),
            "done_count": 0,
            "failed_count": 0,
            "failed_by_user_count": 0,
            "scheduler_type": "random_order",
        }
        if self.publish_metrics:
            self.pub_schedule_metrics.publish(String(data=json.dumps(self.current_run_metrics)))

        self.is_executing = True
        self.saved_header = pose_msg.header
        self.dispatch_next_task()

    def dispatch_next_task(self):
        if len(self.task_queue) > 0:
            rospy.loginfo("📦 随机调度器下发任务 (剩余 %d)...", len(self.task_queue) - 1)
            pose, info, q, oid = self.task_queue.pop(0)

            pa = PoseArray()
            pa.header = self.saved_header
            pa.poses.append(pose)
            self.pub_pose.publish(pa)

            info_msg = Float32MultiArray(data=list(info) + list(q))
            self.pub_info.publish(info_msg)
        else:
            if self.current_run_metrics:
                total_time = rospy.Time.now().to_sec() - self.current_run_metrics["run_start_sim_time"]
                executed_cost = self.current_run_metrics.get("executed_joint_cost_rad", 0.0)
                done_count = self.current_run_metrics.get("done_count", 0)
                failed_count = self.current_run_metrics.get("failed_count", 0)
                self.current_run_metrics["total_time_sec"] = float(total_time)
                self.current_run_metrics["joint_cost_rad"] = float(executed_cost)
                self.pub_schedule_metrics.publish(String(data=json.dumps(self.current_run_metrics)))
                rospy.loginfo(
                    "🎲 随机调度完成: %d 个任务 (done=%d, failed=%d), "
                    "总关节代价=%.4f rad, 总时间=%.1f s",
                    self.current_run_metrics["task_count"],
                    done_count,
                    failed_count,
                    executed_cost,
                    total_time,
                )
            self.is_executing = False
            self.current_run_metrics = None
            self.pub_demo_cmd.publish("all_done")

            # 执行期间缓存的新消息
            if self.pending_pose_msg is not None and self.pending_info_msg is not None:
                rospy.loginfo("📥 执行期间收到新候选，自动触发下一轮随机调度")
                self.primary_pose_msg = self.pending_pose_msg
                self.primary_info_msg = self.pending_info_msg
                self.pending_pose_msg = None
                self.pending_info_msg = None
                self.try_process()

    def status_cb(self, msg):
        status = msg.data.strip()
        if self.current_run_metrics:
            if status == "DONE":
                self.current_run_metrics["done_count"] += 1
            elif status == "FAILED":
                self.current_run_metrics["failed_count"] += 1
            elif status == "FAILED_BY_USER":
                self.current_run_metrics["failed_by_user_count"] += 1
        if status in ["DONE", "FAILED", "FAILED_BY_USER"]:
            self.dispatch_next_task()

    def task_metrics_cb(self, msg):
        if self.current_run_metrics is None:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        cost = payload.get("task_joint_cost_rad", None)
        if cost is None:
            return
        try:
            cost = float(cost)
        except Exception:
            return
        if not np.isfinite(cost):
            return
        self.current_run_metrics["executed_joint_cost_rad"] = (
            self.current_run_metrics.get("executed_joint_cost_rad", 0.0) + cost
        )
        self.current_run_metrics["executed_task_count"] = (
            self.current_run_metrics.get("executed_task_count", 0) + 1
        )

    def failure_reason_cb(self, msg):
        pass  # 不处理，仅消费以防 buffer 堆积


if __name__ == "__main__":
    try:
        scheduler = RandomOrderScheduler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
