#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import time
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from std_msgs.msg import Float32MultiArray, String
from sklearn.cluster import DBSCAN
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
        self.raw_poses = None
        self.raw_infos = None
        self.current_run_metrics = None

        # 3. 订阅 GraspNet 的“原始”姿态数据
        rospy.Subscriber('/graspnet/grasp_pose_array_raw', PoseArray, self.pose_cb)
        rospy.Subscriber('/graspnet/grasp_info_raw', Float32MultiArray, self.info_cb)
        
        # 4. 订阅底层执行器 (demo.py) 的状态反馈
        rospy.Subscriber('/demo/task_status', String, self.status_cb)

        # 5. 把优选后的单体任务发布给执行器
        self.pub_pose = rospy.Publisher('/graspnet/grasp_pose_array', PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)
        self.pub_demo_cmd = rospy.Publisher('/demo/command', String, queue_size=1)
        self.pub_schedule_metrics = rospy.Publisher('/scheduler/run_metrics', String, queue_size=10)

        rospy.loginfo("🚀 多目标 GTSP 任务调度中枢已全面启动并待命！")

# ==========================================
    # 💥 修改后：双重触发机制，彻底解决异步卡死！
    # ==========================================
    def pose_cb(self, msg):
        if self.is_executing: return
        self.raw_poses = msg
        self.process_and_schedule() # 💥 新增：拿到姿态后，也试着去触发一下

    def info_cb(self, msg):
        if self.is_executing: return
        self.raw_infos = msg.data
        self.process_and_schedule() # 拿到信息后，也试着去触发一下
        
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

    def get_ik(self, pose):
        """调用 MoveIt 服务计算真实关节角 (Inverse Kinematics)"""
        req = PositionIKRequest()
        req.group_name = "panda_manipulator"
        req.robot_state = self.robot.get_current_state()
        req.pose_stamped = PoseStamped()
        req.pose_stamped.header.frame_id = self.raw_poses.header.frame_id
        req.pose_stamped.pose = pose
        req.timeout = rospy.Duration(0.1)
        
        res = self.compute_ik(req)
        if res.error_code.val == 1: # 求解成功
            return list(res.solution.joint_state.position[:7])
        return None

    def process_and_schedule(self):
        if not self.raw_poses or not self.raw_infos: return
        schedule_start_wall = time.time()
        rospy.loginfo("="*50)
        rospy.loginfo("🧠 [阶段 1] 开始多目标 GTSP 调度规划...")
        
        num_poses = len(self.raw_poses.poses)
        info_stride = int(len(self.raw_infos) / num_poses) if num_poses > 0 and len(self.raw_infos) % num_poses == 0 else 0
        use_object_ids = info_stride >= 4

        if use_object_ids:
            labels = np.array([int(self.raw_infos[i * info_stride]) for i in range(num_poses)], dtype=np.int32)
            unique_labels = sorted(set(labels.tolist()))
            rospy.loginfo(f"📊 发现 {len(unique_labels)} 个独立物体目标簇（直接使用 object_id 分组）。")
        else:
            # ---------------------------------------------------------
            # 聚类模块: 使用 DBSCAN 区分不同的物体簇
            # ---------------------------------------------------------
            positions = [[p.position.x, p.position.y, p.position.z] for p in self.raw_poses.poses]
            X = np.array(positions)
            # =======================================================
            # 💥 工业级“空间拉伸”黑科技：防止跨层货架误连
            # =======================================================
            X_scaled = np.copy(X)
            # 将所有抓取点的 Z 轴坐标放大 3 倍（XY 保持不变）
            # 这样物理上相差 5cm 的跨层间隙，在算法眼里就变成了 15cm 的鸿沟！
            X_scaled[:, 2] *= 3.0 
            
            clustering = DBSCAN(eps=0.08, min_samples=2).fit(X_scaled)
            labels = clustering.labels_
            unique_labels = set(labels) - {-1} # 剔除离群噪点
                
            if len(unique_labels) == 0:
                rospy.logwarn("未能形成有效的目标簇！")
                self.raw_poses = None; self.raw_infos = None
                return
                
            rospy.loginfo(f"📊 发现 {len(unique_labels)} 个独立物体目标簇。")
        # =========================================================
        # 💥 新增：使用 Open3D 3D球体可视化聚类结果
        # =========================================================
        # try:
        #     import open3d as o3d
        #     rospy.loginfo("👀 正在打开 Open3D 窗口显示【抓取姿态聚类分布】...")
        #     rospy.loginfo("💡 提示：不同颜色的球体代表不同的物体，灰色代表被舍弃的孤立噪点。")
        #     rospy.loginfo("⚠️ 请手动关闭 3D 弹窗，以便调度器继续执行遗传算法！")
            
        #     # 为每个独立的簇随机分配一种明亮的颜色
        #     color_map = {label: [random.random(), random.random(), random.random()] for label in unique_labels}
        #     color_map[-1] = [0.5, 0.5, 0.5]  # -1 代表离群噪点，统一涂成灰色
            
        #     geometries = []
        #     # 在原点添加一个世界坐标系 (XYZ 分别对应红绿蓝，大小 0.1 米)
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #     geometries.append(frame)
            
        #     # 为每一个抓取坐标生成一个半径为 5 毫米的 3D 小球
        #     for i, pos in enumerate(X):
        #         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        #         sphere.translate(pos)
        #         sphere.paint_uniform_color(color_map[labels[i]])
        #         geometries.append(sphere)
                
        #     # 渲染并阻塞，直到用户手动关闭窗口
        #     o3d.visualization.draw_geometries(geometries, window_name="DBSCAN Grasp Clustering (Close to continue)")
        # except ImportError:
        #     rospy.logwarn("未安装 open3d 库，跳过聚类可视化。")
        # =========================================================
        # ---------------------------------------------------------
        # IK 过滤模块: 提前剔除机械臂够不到的姿态
        # ---------------------------------------------------------
        clusters_data = {cid: [] for cid in unique_labels}
        for i, label in enumerate(labels):
            if not use_object_ids and label == -1:
                continue
            pose = self.raw_poses.poses[i]
            if use_object_ids:
                info = self.raw_infos[i * info_stride + 1 : i * info_stride + 4]
            else:
                info = self.raw_infos[i*3 : i*3+3]
            
            q = self.get_ik(pose)
            if q is not None:
                clusters_data[label].append((pose, info, q))
                
        valid_clusters = {cid: data for cid, data in clusters_data.items() if len(data) > 0}
        
        if not valid_clusters:
            rospy.logwarn("所有生成的姿态均超出机械臂工作空间限制！")
            self.raw_poses = None; self.raw_infos = None
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
            pose, info, q = valid_clusters[cid][pose_idx]
            self.task_queue.append((pose, info, q))

        metrics = {
            "task_count": len(self.task_queue),
            "cluster_count": len(valid_clusters),
            "candidate_pose_count": len(self.raw_poses.poses),
            "joint_cost_rad": float(total_joint_cost),
            "scheduler_time_sec": float(scheduler_time_sec),
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
        self.saved_header = self.raw_poses.header

        self.dispatch_next_task()
        
        self.raw_poses = None; self.raw_infos = None

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
                # 切比雪夫距离：取转动幅度最大的那个关节作为该动作的耗时代价
                cost = np.max(np.abs(np.array(target_q) - np.array(curr_q)))
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
            total_cost += float(np.max(np.abs(np.array(target_q) - np.array(curr_q))))
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
