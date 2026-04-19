#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
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

        # 内部任务队列与状态
        self.task_queue = [] 
        self.is_executing = False
        self.primary_pose_msg = None
        self.primary_info_msg = None
        self.fallback_pose_msg = None
        self.fallback_info_msg = None
        self.fallback_timer = None
        self.processing_schedule = False
        self.schedule_lock = threading.Lock()
        self.current_run_metrics = None
        self.pose_topic = rospy.get_param('~pose_topic', '/graspnet/grasp_pose_array_raw')
        self.info_topic = rospy.get_param('~info_topic', '/graspnet/grasp_info_raw')
        self.fallback_pose_topic = rospy.get_param('~fallback_pose_topic', '/graspnet/grasp_pose_array_raw')
        self.fallback_info_topic = rospy.get_param('~fallback_info_topic', '/graspnet/grasp_info_raw')
        self.fallback_timeout_sec = float(rospy.get_param('~fallback_timeout_sec', 1.5))
        self.enable_fallback = bool(
            rospy.get_param('~enable_fallback', False if 'semantic' in self.pose_topic else True)
        )
        self.semantic_weight = float(rospy.get_param('~semantic_weight', 0.0))

        # 3. 订阅主抓取候选话题，并在需要时订阅 raw 兜底话题
        rospy.Subscriber(self.pose_topic, PoseArray, self.pose_cb)
        rospy.Subscriber(self.info_topic, Float32MultiArray, self.info_cb)
        if self.enable_fallback and self.fallback_pose_topic != self.pose_topic:
            rospy.Subscriber(self.fallback_pose_topic, PoseArray, self.fallback_pose_cb)
        if self.enable_fallback and self.fallback_info_topic != self.info_topic:
            rospy.Subscriber(self.fallback_info_topic, Float32MultiArray, self.fallback_info_cb)
        
        # 4. 订阅底层执行器 (demo.py) 的状态反馈
        rospy.Subscriber('/demo/task_status', String, self.status_cb)

        # 5. 把优选后的单体任务发布给执行器
        self.pub_pose = rospy.Publisher('/graspnet/grasp_pose_array', PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)
        self.pub_demo_cmd = rospy.Publisher('/demo/command', String, queue_size=1)
        self.pub_schedule_metrics = rospy.Publisher('/scheduler/run_metrics', String, queue_size=10)

        rospy.loginfo(
            "🚀 多目标 GTSP 任务调度中枢已全面启动并待命！pose_topic=%s info_topic=%s fallback=(%s,%s) semantic_weight=%.3f",
            self.pose_topic,
            self.info_topic,
            self.fallback_pose_topic,
            self.fallback_info_topic,
            self.semantic_weight,
        )

# ==========================================
    # 💥 修改后：双重触发机制，彻底解决异步卡死！
    # ==========================================
    def pose_cb(self, msg):
        if self.is_executing:
            return
        self.primary_pose_msg = msg
        self.try_process_primary()

    def info_cb(self, msg):
        if self.is_executing:
            return
        self.primary_info_msg = msg.data
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

    def get_ik(self, pose, frame_id):
        """调用 MoveIt 服务计算真实关节角 (Inverse Kinematics)"""
        req = PositionIKRequest()
        req.group_name = "panda_manipulator"
        req.robot_state = self.robot.get_current_state()
        req.pose_stamped = PoseStamped()
        req.pose_stamped.header.frame_id = frame_id
        req.pose_stamped.pose = pose
        req.timeout = rospy.Duration(0.1)
        
        res = self.compute_ik(req)
        if res.error_code.val == 1: # 求解成功
            return list(res.solution.joint_state.position[:7])
        return None

    def adjust_transition_cost(self, joint_cost, semantic_score=0.0):
        semantic_score = max(0.0, float(semantic_score))
        if self.semantic_weight <= 1e-9 or semantic_score <= 1e-9:
            return float(joint_cost)
        return float(joint_cost / (1.0 + self.semantic_weight * semantic_score))

    def process_and_schedule(self, pose_msg, info_values, candidate_source="primary"):
        if pose_msg is None or info_values is None:
            return
        schedule_start_wall = time.time()
        rospy.loginfo("="*50)
        rospy.loginfo(f"🧠 [阶段 1] 开始多目标 GTSP 调度规划... (source={candidate_source})")
        
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
        raw_cluster_sizes = {cid: 0 for cid in unique_labels}
        for i, label in enumerate(labels):
            raw_cluster_sizes[label] += 1
            pose = pose_msg.poses[i]
            semantic_score = 0.0
            base = i * info_stride
            info = info_values[base + 1 : base + 4]
            if info_stride >= 5 and (base + 4) < len(info_values):
                semantic_score = float(info_values[base + 4])
            
            q = self.get_ik(pose, pose_msg.header.frame_id)
            if q is not None:
                clusters_data[label].append((pose, info, q, semantic_score))

        for cid in unique_labels:
            rospy.loginfo(
                "  ↳ 目标簇 %s: 原始候选 %d 个, IK 可达 %d 个",
                str(cid),
                int(raw_cluster_sizes.get(cid, 0)),
                int(len(clusters_data.get(cid, []))),
            )

        valid_clusters = {cid: data for cid, data in clusters_data.items() if len(data) > 0}
        dropped_clusters = [cid for cid in unique_labels if len(clusters_data.get(cid, [])) == 0]
        if dropped_clusters:
            rospy.logwarn(
                "⚠️ 以下目标簇在调度前被 IK/可达性过滤完全剔除: %s",
                ", ".join(str(cid) for cid in dropped_clusters),
            )
        
        if not valid_clusters:
            rospy.logwarn("所有生成的姿态均超出机械臂工作空间限制！")
            return

        # ---------------------------------------------------------
        # 遗传算法寻优模块: 求解最优顺序
        # ---------------------------------------------------------
        current_joints = self.arm.get_current_joint_values()
        rospy.loginfo("🧬 正在运行遗传算法 (Genetic Algorithm) 优化抓取序列...")
        
        best_sequence = self.run_genetic_algorithm(valid_clusters, current_joints)
        total_joint_cost = self.calc_sequence_cost(best_sequence, valid_clusters, current_joints)
        scheduler_time_sec = time.time() - schedule_start_wall
        
        # 将最优序列转化为任务队列
        self.task_queue = []
        for cid, pose_idx in best_sequence:
            pose, info, q, _semantic_score = valid_clusters[cid][pose_idx]
            self.task_queue.append((pose, info, q))

        metrics = {
            "task_count": len(self.task_queue),
            "cluster_count": len(valid_clusters),
            "candidate_pose_count": len(pose_msg.poses),
            "joint_cost_rad": float(total_joint_cost),
            "scheduler_time_sec": float(scheduler_time_sec),
            "semantic_weight": float(self.semantic_weight),
            "candidate_source": str(candidate_source),
            "run_start_sim_time": rospy.Time.now().to_sec(),
            "done_count": 0,
            "failed_count": 0,
            "failed_by_user_count": 0,
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

    def run_genetic_algorithm(self, clusters, start_joints):
        """核心算法：基于遗传算法的 GTSP 路径规划"""
        POP_SIZE = 50       # 种群大小
        GENERATIONS = 30    # 迭代代数
        MUTATION_RATE = 0.2 # 变异率
        
        cluster_ids = list(clusters.keys())
        num_targets = len(cluster_ids)
        
        # 初始化种群：随机生成抓取顺序和每个物体的姿态选择
        population = []
        for _ in range(POP_SIZE):
            seq = copy.deepcopy(cluster_ids)
            random.shuffle(seq)
            choices = {cid: random.randint(0, len(clusters[cid])-1) for cid in cluster_ids}
            population.append((seq, choices))
            
        def calc_cost(chrom):
            """适应度评估：计算该序列的【关节转移总代价】"""
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
            pose, info, q = self.task_queue.pop(0)

            # 把这一个被 GA 选中的“天命姿态”包装成数组下发
            pa = PoseArray()
            # 💥 修复 2：使用存下来的 saved_header
            pa.header = self.saved_header  
            pa.poses.append(pose)
            self.pub_pose.publish(pa)
            
            # info(3个: width/score/depth) + q(7个关节角) 一起下发
            info_msg = Float32MultiArray(data=list(info) + list(q))
            self.pub_info.publish(info_msg)
        else:
            if self.current_run_metrics is not None:
                total_time = rospy.Time.now().to_sec() - self.current_run_metrics["run_start_sim_time"]
                rospy.loginfo(
                    f"📊 本轮任务统计: joint_cost={self.current_run_metrics['joint_cost_rad']:.4f} rad, "
                    f"scheduler_time={self.current_run_metrics['scheduler_time_sec']:.4f} s, "
                    f"total_time={total_time:.3f} s, "
                    f"done={self.current_run_metrics['done_count']}, "
                    f"failed={self.current_run_metrics['failed_count']}, "
                    f"failed_by_user={self.current_run_metrics['failed_by_user_count']}"
                )
            rospy.loginfo("🎉 所有多目标拣选任务已全部完成！调度中枢进入空闲等待。")
            self.is_executing = False
            self.current_run_metrics = None
            self.pub_demo_cmd.publish("all_done")

if __name__ == '__main__':
    try:
        scheduler = GTSPScheduler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
