#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import time
import threading
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import Float32MultiArray, String
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
import moveit_commander
import random
import copy

try:
    import rospkg
    _sam_scripts = os.path.join(rospkg.RosPack().get_path("sam_perception"), "scripts")
    if _sam_scripts not in sys.path:
        sys.path.insert(0, _sam_scripts)
except Exception:
    for _root in os.environ.get("ROS_PACKAGE_PATH", "").split(":"):
        _sam_scripts = os.path.join(_root, "sam_perception", "scripts")
        if os.path.exists(os.path.join(_sam_scripts, "execution_ik_utils.py")):
            if _sam_scripts not in sys.path:
                sys.path.insert(0, _sam_scripts)
            break

from execution_ik_utils import ExecutionIKProjector

class GTSPScheduler:
    def __init__(self):
        rospy.init_node('gtsp_scheduler_node')
        
        # 1. 启动 MoveIt 核心
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        
        # =======================================================
        # 💥 核心修复：无限循环等待 MoveIt 就绪，防止启动时崩溃
        # =======================================================
        connected = False
        rospy.loginfo("⏳ 正在等待 MoveIt 核心服务启动 (GTSP 调度器)...")
        while not rospy.is_shutdown() and not connected:
            try:
                self.arm = moveit_commander.MoveGroupCommander("panda_manipulator")
                connected = True
                rospy.loginfo("✅ 调度器成功连接到 MoveGroup！")
            except RuntimeError as e:
                rospy.logwarn(f"⚠️ MoveIt 尚未就绪，调度器等待中... {e}")
                rospy.sleep(2.0)
        
        # 2. 等待 IK 运动学服务
        rospy.loginfo("⏳ 等待 MoveIt IK 运动学计算服务...")
        rospy.wait_for_service('/compute_ik')
        self.compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        rospy.loginfo("✅ IK 服务就绪！")
        self.execution_ik_projector = ExecutionIKProjector()

        # 内部任务队列与状态
        self.task_queue = []
        self.is_executing = False
        self.primary_pose_msg = None
        self.primary_info_msg = None
        self.pending_pose_msg = None   # 执行中缓存的新消息
        self.pending_info_msg = None   # 执行中缓存的新消息
        self.fallback_pose_msg = None
        self.fallback_info_msg = None
        self.fallback_timer = None
        self.processing_schedule = False
        self.schedule_lock = threading.Lock()
        self.current_run_metrics = None
        self.last_successful_ik = None
        self.home_joint_seed = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.ik_seed_trials = max(1, int(rospy.get_param('~ik_seed_trials', 3)))
        self.ik_timeout_sec = float(rospy.get_param('~ik_timeout_sec', 0.35))
        self.semantic_fallback_timeout_min = float(
            rospy.get_param('~semantic_fallback_timeout_min', 12.0)
        )
        self.pose_topic = rospy.get_param('~pose_topic', '/graspnet/grasp_pose_array_raw')
        self.info_topic = rospy.get_param('~info_topic', '/graspnet/grasp_info_raw')
        self.fallback_pose_topic = rospy.get_param('~fallback_pose_topic', '/graspnet/grasp_pose_array_raw')
        self.fallback_info_topic = rospy.get_param('~fallback_info_topic', '/graspnet/grasp_info_raw')
        self.fallback_timeout_sec = float(rospy.get_param('~fallback_timeout_sec', 1.5))
        self.enable_fallback = bool(
            rospy.get_param('~enable_fallback', False if 'semantic' in self.pose_topic else True)
        )
        self.semantic_weight = float(rospy.get_param('~semantic_weight', 0.0))
        self.reachability_weight = float(rospy.get_param('~reachability_weight', 0.15))
        self.per_task_candidate_count = max(1, int(rospy.get_param('~per_task_candidate_count', 4)))
        self.allow_unreachable_candidate_dispatch = bool(
            rospy.get_param('~allow_unreachable_candidate_dispatch', True)
        )
        self._ik_seed_base = None

        semantic_pipeline = ('semantic' in self.pose_topic) or ('semantic' in self.info_topic)
        if semantic_pipeline and self.enable_fallback and self.fallback_timeout_sec < self.semantic_fallback_timeout_min:
            rospy.logwarn(
                "⚠️ 检测到语义候选管线，但 fallback_timeout_sec=%.2f s 偏小，自动提升到 %.2f s，"
                "避免语义重排尚未完成就回退 raw。",
                self.fallback_timeout_sec,
                self.semantic_fallback_timeout_min,
            )
            self.fallback_timeout_sec = float(self.semantic_fallback_timeout_min)

        # 3. 订阅主抓取候选话题，并在需要时订阅 raw 兜底话题
        rospy.Subscriber(self.pose_topic, PoseArray, self.pose_cb)
        rospy.Subscriber(self.info_topic, Float32MultiArray, self.info_cb)
        if self.enable_fallback and self.fallback_pose_topic != self.pose_topic:
            rospy.Subscriber(self.fallback_pose_topic, PoseArray, self.fallback_pose_cb)
        if self.enable_fallback and self.fallback_info_topic != self.info_topic:
            rospy.Subscriber(self.fallback_info_topic, Float32MultiArray, self.fallback_info_cb)
        
        # 4. 订阅底层执行器 (demo.py) 的状态反馈
        rospy.Subscriber('/demo/task_status', String, self.status_cb)
        rospy.Subscriber('/demo/task_metrics', String, self.task_metrics_cb)

        # 5. 把优选后的单体任务发布给执行器
        self.pub_pose = rospy.Publisher('/graspnet/grasp_pose_array', PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)
        self.pub_demo_cmd = rospy.Publisher('/demo/command', String, queue_size=1)
        self.pub_schedule_metrics = rospy.Publisher('/scheduler/run_metrics', String, queue_size=10)

        rospy.loginfo(
            "🚀 多目标 GTSP 任务调度中枢已全面启动并待命！pose_topic=%s info_topic=%s fallback=(%s,%s) semantic_weight=%.3f per_task_candidate_count=%d unreachable_dispatch=%s",
            self.pose_topic,
            self.info_topic,
            self.fallback_pose_topic,
            self.fallback_info_topic,
            self.semantic_weight,
            self.per_task_candidate_count,
            str(self.allow_unreachable_candidate_dispatch).lower(),
        )

# ==========================================
    # 💥 修改后：双重触发机制，彻底解决异步卡死！
    # ==========================================
    def pose_cb(self, msg):
        self.primary_pose_msg = msg
        if self.is_executing:
            self.pending_pose_msg = msg  # 缓存，执行完再处理
            return
        self.try_process_primary()

    def info_cb(self, msg):
        self.primary_info_msg = msg.data
        if self.is_executing:
            self.pending_info_msg = msg.data  # 缓存，执行完再处理
            return
        self.try_process_primary()

    def fallback_pose_cb(self, msg):
        if self.is_executing:
            return
        self.fallback_pose_msg = msg
        self.maybe_arm_fallback_timer()

    def fallback_info_cb(self, msg):
        if self.is_executing:
            return
        self.fallback_info_msg = msg.data
        self.maybe_arm_fallback_timer()

    def cancel_fallback_timer(self):
        if self.fallback_timer is not None:
            self.fallback_timer.shutdown()
            self.fallback_timer = None

    def try_process_primary(self):
        self.cancel_fallback_timer()
        if self.is_executing or self.processing_schedule:
            return
        if self.primary_pose_msg is None or self.primary_info_msg is None:
            self.maybe_arm_fallback_timer()
            return
        if not self.schedule_lock.acquire(blocking=False):
            return
        self.processing_schedule = True
        pose_msg = self.primary_pose_msg
        info_values = list(self.primary_info_msg)
        self.primary_pose_msg = None
        self.primary_info_msg = None
        try:
            self.process_and_schedule(pose_msg, info_values, candidate_source="primary")
        finally:
            self.processing_schedule = False
            self.schedule_lock.release()

    def maybe_arm_fallback_timer(self):
        if self.is_executing or self.processing_schedule:
            return
        if not self.enable_fallback:
            return
        if self.pose_topic == self.fallback_pose_topic and self.info_topic == self.fallback_info_topic:
            return
        if self.primary_pose_msg is not None and self.primary_info_msg is not None:
            return
        if self.fallback_pose_msg is None or self.fallback_info_msg is None:
            return
        if self.fallback_timer is None:
            self.fallback_timer = rospy.Timer(
                rospy.Duration(self.fallback_timeout_sec),
                self.fallback_timer_cb,
                oneshot=True,
            )

    def fallback_timer_cb(self, _event):
        self.fallback_timer = None
        if self.is_executing or self.processing_schedule:
            return
        if self.primary_pose_msg is not None and self.primary_info_msg is not None:
            return
        if self.fallback_pose_msg is None or self.fallback_info_msg is None:
            return
        if not self.schedule_lock.acquire(blocking=False):
            return
        self.processing_schedule = True
        rospy.logwarn(
            "⚠️ 主候选话题在 %.2f s 内未成对到达，回退使用 raw 抓取候选继续调度。",
            self.fallback_timeout_sec,
        )
        pose_msg = self.fallback_pose_msg
        info_values = list(self.fallback_info_msg)
        self.fallback_pose_msg = None
        self.fallback_info_msg = None
        self.primary_pose_msg = None
        self.primary_info_msg = None
        try:
            self.process_and_schedule(pose_msg, info_values, candidate_source="fallback")
        finally:
            self.processing_schedule = False
            self.schedule_lock.release()
        
    def status_cb(self, msg):
        status = msg.data
        if self.current_run_metrics is not None:
            if status == "DONE":
                self.current_run_metrics["done_count"] += 1
            elif status == "FAILED":
                self.current_run_metrics["failed_count"] += 1
            elif status == "FAILED_BY_USER":
                self.current_run_metrics["failed_by_user_count"] += 1
        if status in ["DONE", "FAILED", "FAILED_BY_USER"]:
            rospy.loginfo(f"📥 收到执行器反馈: {status}，准备派发下一个任务。")
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

        self.current_run_metrics["executed_joint_cost_rad"] += cost
        self.current_run_metrics["executed_task_count"] += 1
        status = str(payload.get("status", "")).strip().upper()
        if status == "DONE":
            self.current_run_metrics["executed_done_joint_cost_rad"] += cost
            self.current_run_metrics["executed_done_task_count"] += 1

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

    def _build_ik_seed_candidates(self):
        seeds = []
        for candidate in [self._ik_seed_base, self.last_successful_ik, self.home_joint_seed]:
            if candidate is None:
                continue
            c = [float(v) for v in candidate[:7]]
            if len(c) < 7:
                continue
            if not any(np.allclose(np.array(c), np.array(s), atol=1e-4) for s in seeds):
                seeds.append(c)
        return seeds[: self.ik_seed_trials]

    def call_ik(self, pose, frame_id):
        """调用 MoveIt 服务计算真实关节角 (Inverse Kinematics)"""
        seeds = self._build_ik_seed_candidates()
        if not seeds:
            seeds = [None]

        last_error_code = -1
        for trial_idx, seed in enumerate(seeds):
            req = PositionIKRequest()
            req.group_name = "panda_manipulator"
            req.robot_state = self.robot.get_current_state()
            if seed is not None:
                self._apply_seed_to_robot_state(req.robot_state, seed)
            req.pose_stamped = PoseStamped()
            req.pose_stamped.header.frame_id = frame_id
            req.pose_stamped.pose = pose
            req.timeout = rospy.Duration(self.ik_timeout_sec)
            try:
                res = self.compute_ik(req)
            except Exception as exc:
                rospy.logdebug("IK 调用异常 (trial=%d): %s", trial_idx + 1, str(exc))
                continue
            last_error_code = int(res.error_code.val)
            if res.error_code.val == 1: # 求解成功
                q = list(res.solution.joint_state.position[:7])
                self.last_successful_ik = q
                return q
        rospy.logdebug(
            "IK 全部失败: pos=(%.3f,%.3f,%.3f) frame=%s last_error=%d trials=%d timeout=%.2fs",
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            str(frame_id),
            last_error_code,
            len(seeds),
            self.ik_timeout_sec,
        )
        return None

    def get_ik(self, pose, frame_id):
        """按 demo.py 实际执行的观察位姿态做 IK，而不是直接判 raw GraspNet pose。"""
        planning_frame = self.robot.get_planning_frame()
        projected = self.execution_ik_projector.build_observation_poses(
            pose,
            frame_id,
            planning_frame,
        )
        for observation_pose, meta in projected:
            q = self.call_ik(observation_pose, planning_frame)
            if q is not None:
                rospy.logdebug(
                    "scheduler execution-style IK ok: mode=%s back=%.3f lift=%.3f target_z=%.3f observe_z=%.3f",
                    str(meta.get("mode", "?")),
                    float(meta.get("back_distance", -1.0)),
                    float(meta.get("obs_lift", -1.0)),
                    float(meta.get("target_z", -1.0)),
                    float(meta.get("observe_z", -1.0)),
                )
                return q
        return None

    def adjust_transition_cost(self, joint_cost, semantic_score=0.0):
        semantic_score = max(0.0, float(semantic_score))
        if self.semantic_weight <= 1e-9 or semantic_score <= 1e-9:
            return float(joint_cost)
        return float(joint_cost / (1.0 + self.semantic_weight * semantic_score))

    def build_unreachable_direct_tasks(self, raw_clusters_data, cluster_ids):
        """IK 预过滤失败时仍下发原始候选，让执行器的直接抓取兜底接管。"""
        tasks = []
        for cid in cluster_ids:
            cluster_candidates = raw_clusters_data.get(cid, [])
            if not cluster_candidates:
                continue
            sorted_indices = sorted(
                range(len(cluster_candidates)),
                key=lambda idx: (
                    float(cluster_candidates[idx][2]) if len(cluster_candidates[idx]) > 2 else 0.0,
                    float(cluster_candidates[idx][1][1]) if len(cluster_candidates[idx][1]) > 1 else 0.0,
                ),
                reverse=True,
            )
            candidate_indices = sorted_indices[: self.per_task_candidate_count]
            candidate_poses = []
            candidate_infos = []
            for idx in candidate_indices:
                pose, info, _semantic_score = cluster_candidates[idx]
                candidate_poses.append(pose)
                candidate_infos.append(list(info))
            if not candidate_poses:
                continue
            tasks.append(
                {
                    "poses": candidate_poses,
                    "infos": candidate_infos,
                    # 给执行器一个稳定种子；真正的目标位 IK/规划交给直接抓取兜底再试。
                    "seed_q": list(self._ik_seed_base or self.last_successful_ik or self.home_joint_seed),
                    "unreachable_direct": True,
                    "cluster_id": cid,
                }
            )
        return tasks

    def process_and_schedule(self, pose_msg, info_values, candidate_source="primary"):
        if pose_msg is None or info_values is None:
            return
        schedule_start_wall = time.time()
        rospy.loginfo("="*50)
        rospy.loginfo(f"🧠 [阶段 1] 开始多目标 GTSP 调度规划... (source={candidate_source})")
        try:
            self._ik_seed_base = list(self.arm.get_current_joint_values()[:7])
        except Exception:
            self._ik_seed_base = None
        
        num_poses = len(pose_msg.poses)
        info_stride = int(len(info_values) / num_poses) if num_poses > 0 and len(info_values) % num_poses == 0 else 0
        if info_stride < 4:
            rospy.logerr(
                "❌ 当前调度器要求 grasp_info 至少包含 [object_id, width, score, depth]。"
                "收到长度=%d, num_poses=%d, info_stride=%d。请检查上游 grasp_from_sam / semantic_reranker 输出。",
                len(info_values),
                num_poses,
                info_stride,
            )
            return

        labels = np.array([int(info_values[i * info_stride]) for i in range(num_poses)], dtype=np.int32)
        unique_labels = sorted(set(labels.tolist()))
        rospy.loginfo(f"📊 发现 {len(unique_labels)} 个独立物体目标簇（基于 object_id 显式分组）。")
        # ---------------------------------------------------------
        # IK 过滤模块: 提前剔除机械臂够不到的姿态
        # ---------------------------------------------------------
        clusters_data = {cid: [] for cid in unique_labels}
        raw_clusters_data = {cid: [] for cid in unique_labels}
        raw_cluster_sizes = {cid: 0 for cid in unique_labels}
        for i, label in enumerate(labels):
            raw_cluster_sizes[label] += 1
            pose = pose_msg.poses[i]
            semantic_score = 0.0
            base = i * info_stride
            info = info_values[base + 1 : base + 4]
            if info_stride >= 5 and (base + 4) < len(info_values):
                semantic_score = float(info_values[base + 4])
            raw_clusters_data[label].append((pose, list(info), semantic_score))
            
            q = self.get_ik(pose, pose_msg.header.frame_id)
            if q is not None:
                clusters_data[label].append((pose, info, q, semantic_score))

        for cid in unique_labels:
            raw_count = int(raw_cluster_sizes.get(cid, 0))
            reachable_count = int(len(clusters_data.get(cid, [])))
            ratio_str = f"{reachable_count}/{raw_count}" if raw_count > 0 else "0/0"
            if raw_count > 0:
                ratio_pct = 100.0 * reachable_count / raw_count
                level = rospy.logwarn if ratio_pct < 10.0 else rospy.loginfo
                level(
                    "  ↳ 目标簇 %s: IK 可达 %s (%.1f%%)",
                    str(cid),
                    ratio_str,
                    ratio_pct,
                )
            else:
                rospy.loginfo("  ↳ 目标簇 %s: 原始候选 0 个", str(cid))

        # 计算每个簇的 IK 可达率，传递给 GA 作为可达性奖励
        cluster_reachability = {}
        for cid in unique_labels:
            raw = int(raw_cluster_sizes.get(cid, 0))
            if raw > 0:
                cluster_reachability[cid] = float(len(clusters_data.get(cid, []))) / float(raw)
            else:
                cluster_reachability[cid] = 0.0

        valid_clusters = {cid: data for cid, data in clusters_data.items() if len(data) > 0}
        dropped_clusters = [cid for cid in unique_labels if len(clusters_data.get(cid, [])) == 0]
        if dropped_clusters:
            rospy.logwarn(
                "⚠️ 以下目标簇在调度前被 IK/可达性过滤完全剔除: %s",
                ", ".join(str(cid) for cid in dropped_clusters),
            )
            if not self.allow_unreachable_candidate_dispatch:
                rospy.logwarn(
                    "⏭️ 工程兜底已关闭，这些 IK=0 目标本轮将被跳过，不会下发给执行器。"
                )
        
        if not valid_clusters:
            rospy.logwarn("所有生成的姿态均超出机械臂工作空间限制！")
            if self.allow_unreachable_candidate_dispatch:
                self.task_queue = self.build_unreachable_direct_tasks(raw_clusters_data, unique_labels)
                if self.task_queue:
                    scheduler_time_sec = time.time() - schedule_start_wall
                    self.current_run_metrics = {
                        "task_count": len(self.task_queue),
                        "cluster_count": 0,
                        "unreachable_direct_task_count": len(self.task_queue),
                        "candidate_pose_count": len(pose_msg.poses),
                        "joint_cost_rad": 0.0,
                        "planned_joint_cost_rad": 0.0,
                        "scheduler_time_sec": float(scheduler_time_sec),
                        "semantic_weight": float(self.semantic_weight),
                        "candidate_source": str(candidate_source),
                        "run_start_sim_time": rospy.Time.now().to_sec(),
                        "done_count": 0,
                        "failed_count": 0,
                        "failed_by_user_count": 0,
                        "executed_joint_cost_rad": 0.0,
                        "executed_done_joint_cost_rad": 0.0,
                        "executed_task_count": 0,
                        "executed_done_task_count": 0,
                    }
                    self.pub_schedule_metrics.publish(String(data=json.dumps(self.current_run_metrics)))
                    rospy.logwarn(
                        "⚠️ IK 预过滤为 0，但已下发 %d 个工程兜底任务给执行器直接尝试。",
                        len(self.task_queue),
                    )
                    self.is_executing = True
                    self.saved_header = pose_msg.header
                    self.dispatch_next_task()
                    self.fallback_pose_msg = None
                    self.fallback_info_msg = None
                    self._ik_seed_base = None
                    return
            return

        # ---------------------------------------------------------
        # 遗传算法寻优模块: 求解最优顺序
        # ---------------------------------------------------------
        current_joints = self.arm.get_current_joint_values()
        rospy.loginfo("🧬 正在运行遗传算法 (Genetic Algorithm) 优化抓取序列...")
        
        best_sequence = self.run_genetic_algorithm(valid_clusters, current_joints, cluster_reachability)
        total_joint_cost = self.calc_sequence_cost(best_sequence, valid_clusters, current_joints)
        scheduler_time_sec = time.time() - schedule_start_wall
        
        # 将最优序列转化为任务队列（每个目标下发多个备选姿态，首个为 GA 选中姿态）
        self.task_queue = []
        for cid, pose_idx in best_sequence:
            cluster_candidates = valid_clusters[cid]
            # 先按语义分和抓取得分做一个排序，再把 GA 选中索引提前到首位
            sorted_indices = sorted(
                range(len(cluster_candidates)),
                key=lambda idx: (
                    float(cluster_candidates[idx][3]) if len(cluster_candidates[idx]) > 3 else 0.0,
                    float(cluster_candidates[idx][1][1]) if len(cluster_candidates[idx][1]) > 1 else 0.0,
                ),
                reverse=True,
            )
            candidate_indices = [pose_idx] + [idx for idx in sorted_indices if idx != pose_idx]
            candidate_indices = candidate_indices[: self.per_task_candidate_count]

            candidate_poses = []
            candidate_infos = []
            for idx in candidate_indices:
                pose, info, _q, _semantic_score = cluster_candidates[idx]
                candidate_poses.append(pose)
                candidate_infos.append(list(info))

            # 仍保留 GA 选中的关节角作为首选构型种子（给 demo seeded IK 用）
            selected_q = list(cluster_candidates[pose_idx][2])
            self.task_queue.append(
                {
                    "poses": candidate_poses,
                    "infos": candidate_infos,
                    "seed_q": selected_q,
                    "cluster_id": cid,
                }
            )

        unreachable_direct_tasks = []
        if self.allow_unreachable_candidate_dispatch and dropped_clusters:
            unreachable_direct_tasks = self.build_unreachable_direct_tasks(raw_clusters_data, dropped_clusters)
            if unreachable_direct_tasks:
                rospy.logwarn(
                    "⚠️ %d 个目标簇 IK 预过滤为 0，已追加为工程兜底直接抓取任务。",
                    len(unreachable_direct_tasks),
                )
                self.task_queue.extend(unreachable_direct_tasks)

        metrics = {
            "task_count": len(self.task_queue),
            "cluster_count": len(valid_clusters),
            "unreachable_direct_task_count": len(unreachable_direct_tasks),
            "skipped_ik_zero_count": len(dropped_clusters) if not self.allow_unreachable_candidate_dispatch else 0,
            "skipped_ik_zero_clusters": [int(cid) for cid in dropped_clusters]
            if not self.allow_unreachable_candidate_dispatch
            else [],
            "candidate_pose_count": len(pose_msg.poses),
            "joint_cost_rad": float(total_joint_cost),
            "planned_joint_cost_rad": float(total_joint_cost),
            "scheduler_time_sec": float(scheduler_time_sec),
            "semantic_weight": float(self.semantic_weight),
            "candidate_source": str(candidate_source),
            "run_start_sim_time": rospy.Time.now().to_sec(),
            "done_count": 0,
            "failed_count": 0,
            "failed_by_user_count": 0,
            "executed_joint_cost_rad": 0.0,
            "executed_done_joint_cost_rad": 0.0,
            "executed_task_count": 0,
            "executed_done_task_count": 0,
        }
        self.current_run_metrics = metrics
        self.pub_schedule_metrics.publish(String(data=json.dumps(metrics)))
            
        rospy.loginfo(f"🏆 最优序列已生成 (共 {len(self.task_queue)} 步)。准备分发指令！")
        self.is_executing = True

        # 💥 修复 1：把 header 提前存下来，再分发指令！
        self.saved_header = pose_msg.header

        self.dispatch_next_task()
        self.fallback_pose_msg = None
        self.fallback_info_msg = None
        self._ik_seed_base = None

    def run_genetic_algorithm(self, clusters, start_joints, cluster_reachability=None):
        """核心算法：基于遗传算法的 GTSP 路径规划"""
        POP_SIZE = 50       # 种群大小
        GENERATIONS = 30    # 迭代代数
        MUTATION_RATE = 0.2 # 变异率

        cluster_ids = list(clusters.keys())
        num_targets = len(cluster_ids)

        if cluster_reachability is None:
            cluster_reachability = {cid: 1.0 for cid in cluster_ids}
        # 可达性奖励：可达率低于 20% 的簇会被显著惩罚
        def reachability_penalty(cid):
            ratio = float(cluster_reachability.get(cid, 1.0))
            ratio = max(ratio, 1e-6)
            if ratio >= 1.0:
                return 0.0
            return float(self.reachability_weight * (1.0 - ratio))

        # 初始化种群：随机生成抓取顺序和每个物体的姿态选择
        population = []
        for _ in range(POP_SIZE):
            seq = copy.deepcopy(cluster_ids)
            random.shuffle(seq)
            choices = {cid: random.randint(0, len(clusters[cid])-1) for cid in cluster_ids}
            population.append((seq, choices))

        def calc_cost(chrom):
            """适应度评估：计算该序列的【关节转移总代价 + 可达性惩罚】"""
            seq, choices = chrom
            total_cost = 0.0
            curr_q = start_joints
            for cid in seq:
                pose_idx = choices[cid]
                target_q = clusters[cid][pose_idx][2]
                semantic_score = clusters[cid][pose_idx][3] if len(clusters[cid][pose_idx]) > 3 else 0.0
                # 切比雪夫距离：取转动幅度最大的那个关节作为该动作的耗时代价
                cost = np.max(np.abs(np.array(target_q) - np.array(curr_q)))
                cost = self.adjust_transition_cost(cost, semantic_score)
                # 添加可达性惩罚：低 IK 可达率的簇会被 GA 自然规避
                cost = cost * (1.0 + reachability_penalty(cid))
                total_cost += cost
                curr_q = target_q
            return total_cost

        # 进化迭代
        for gen in range(GENERATIONS):
            fitness_scores = [1.0 / (calc_cost(chrom) + 1e-6) for chrom in population]
            total_fitness = sum(fitness_scores)
            probs = [f / total_fitness for f in fitness_scores]
            
            new_population = []
            # 精英保留策略
            best_idx = np.argmax(fitness_scores)
            new_population.append(copy.deepcopy(population[best_idx]))
            
            while len(new_population) < POP_SIZE:
                # 轮盘赌选择
                p1 = population[np.random.choice(POP_SIZE, p=probs)]
                p2 = population[np.random.choice(POP_SIZE, p=probs)]
                
                # 交叉 (随机继承父母的姿态选择)
                child_seq = copy.deepcopy(p1[0]) 
                child_choices = {cid: (p1[1][cid] if random.random() > 0.5 else p2[1][cid]) for cid in cluster_ids}
                    
                # 变异 (随机交换抓取顺序)
                if random.random() < MUTATION_RATE and num_targets > 1:
                    idx1, idx2 = random.sample(range(num_targets), 2)
                    child_seq[idx1], child_seq[idx2] = child_seq[idx2], child_seq[idx1]
                    
                # 变异 (随机更换抓取姿态)
                if random.random() < MUTATION_RATE:
                    mut_cid = random.choice(cluster_ids)
                    child_choices[mut_cid] = random.randint(0, len(clusters[mut_cid])-1)
                    
                new_population.append((child_seq, child_choices))
            population = new_population

        # 选出这几十代里进化出的最优个体
        final_costs = [calc_cost(chrom) for chrom in population]
        best_chrom = population[np.argmin(final_costs)]
        
        rospy.loginfo(f"📉 算法收敛！最小总关节代价: {np.min(final_costs):.4f} rad")
        return [(cid, best_chrom[1][cid]) for cid in best_chrom[0]]

    def calc_sequence_cost(self, best_sequence, clusters, start_joints):
        total_cost = 0.0
        curr_q = start_joints
        for cid, pose_idx in best_sequence:
            target_q = clusters[cid][pose_idx][2]
            semantic_score = clusters[cid][pose_idx][3] if len(clusters[cid][pose_idx]) > 3 else 0.0
            joint_cost = float(np.max(np.abs(np.array(target_q) - np.array(curr_q))))
            total_cost += self.adjust_transition_cost(joint_cost, semantic_score)
            curr_q = target_q
        return total_cost

    def dispatch_next_task(self):
        """派发任务给 demo.py 机械臂执行器"""
        if len(self.task_queue) > 0:
            rospy.loginfo(f"📦 正在下发队列中的下一个目标... (剩余 {len(self.task_queue)-1} 个)")
            task = self.task_queue.pop(0)
            poses = task.get("poses", [])
            infos = task.get("infos", [])
            seed_q = task.get("seed_q", [])
            cluster_id = task.get("cluster_id", None)

            # 下发该目标的多候选姿态（首个为 GA 选中）
            pa = PoseArray()
            pa.header = self.saved_header  
            for pose in poses:
                pa.poses.append(pose)
            self.pub_pose.publish(pa)
            
            # info payload 约定：
            # [pose1(width,score,depth), pose2(...), ... , seed_q(7)]
            info_data = []
            for info in infos:
                if len(info) >= 3:
                    info_data.extend([float(info[0]), float(info[1]), float(info[2])])
            if len(seed_q) >= 7:
                info_data.extend([float(v) for v in seed_q[:7]])
            info_msg = Float32MultiArray(data=info_data)
            self.pub_info.publish(info_msg)
            if task.get("unreachable_direct", False):
                rospy.logwarn(
                    "🧯 当前目标来自 IK=0 工程兜底队列：cluster=%s，执行器将直接尝试抓取候选。",
                    str(task.get("cluster_id", "?")),
                )
            elif cluster_id is not None:
                rospy.loginfo(
                    "📌 当前下发目标簇: cluster=%s，候选数=%d。",
                    str(cluster_id),
                    len(pa.poses),
                )
            rospy.loginfo(
                "📨 已下发目标候选数: %d（seed_q=%s）",
                len(pa.poses),
                "yes" if len(seed_q) >= 7 else "no",
            )
        else:
            if self.current_run_metrics is not None:
                total_time = rospy.Time.now().to_sec() - self.current_run_metrics["run_start_sim_time"]
                self.current_run_metrics["total_time_sec"] = float(total_time)
                # 兼容下游旧字段：将最终执行总代价写入 joint_cost_rad。
                # 同时保留 planned_joint_cost_rad 作为规划阶段总代价。
                self.current_run_metrics["joint_cost_rad"] = float(
                    self.current_run_metrics.get("executed_joint_cost_rad", 0.0)
                )
                self.pub_schedule_metrics.publish(String(data=json.dumps(self.current_run_metrics)))
                rospy.loginfo(
                    f"📊 本轮任务统计: total_cost={self.current_run_metrics['executed_joint_cost_rad']:.4f} rad, "
                    f"done_cost={self.current_run_metrics['executed_done_joint_cost_rad']:.4f} rad, "
                    f"planned_cost={self.current_run_metrics.get('planned_joint_cost_rad', 0.0):.4f} rad, "
                    f"scheduler_time={self.current_run_metrics['scheduler_time_sec']:.4f} s, "
                    f"total_time={total_time:.3f} s, "
                    f"done={self.current_run_metrics['done_count']}, "
                    f"failed={self.current_run_metrics['failed_count']}, "
                    f"failed_by_user={self.current_run_metrics['failed_by_user_count']}, "
                    f"skipped_ik0={self.current_run_metrics.get('skipped_ik_zero_count', 0)}"
                )
            skipped_count = 0
            skipped_clusters = []
            if self.current_run_metrics is not None:
                skipped_count = int(self.current_run_metrics.get("skipped_ik_zero_count", 0))
                skipped_clusters = list(self.current_run_metrics.get("skipped_ik_zero_clusters", []))
            if skipped_count > 0:
                rospy.logwarn(
                    "🏁 所有可执行目标已完成；另有 %d 个目标因 IK=0 被跳过: %s",
                    skipped_count,
                    ", ".join(str(cid) for cid in skipped_clusters),
                )
            else:
                rospy.loginfo("🎉 所有多目标拣选任务已全部完成！调度中枢进入空闲等待。")
            self.is_executing = False
            self.current_run_metrics = None
            self.pub_demo_cmd.publish("all_done")
            # 检查执行期间是否有新消息到达，有则自动触发新一轮调度
            if self.pending_pose_msg is not None and self.pending_info_msg is not None:
                rospy.loginfo("📥 执行期间收到新抓取候选，自动触发下一轮调度。")
                self.primary_pose_msg = self.pending_pose_msg
                self.primary_info_msg = self.pending_info_msg
                self.pending_pose_msg = None
                self.pending_info_msg = None
                self.try_process_primary()

if __name__ == '__main__':
    try:
        scheduler = GTSPScheduler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
