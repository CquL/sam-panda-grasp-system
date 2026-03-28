#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import rospy
import numpy as np
import moveit_commander
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft 

try:
    from gazebo_interface import GazeboCubeManager
    from motion_controller import ArmController, GripperActionController
except ImportError:
    rospy.logwarn("无法导入自定义控制器，请确保相关文件在搜索路径中。")

class PickPlaceDemo:
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        # =======================================================
        # 🛡️ 强制安全层：将绝对物理地面加入碰撞空间，防止砸桌子！
        # =======================================================
        rospy.sleep(1.0) # 等待场景接口初始化完成
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        # 将地面放在 z=0 的位置（或者稍微往下沉一点点 z=-0.02，避免和底座模型摩擦）
        ground_pose.pose.position.z = -0.02 
        ground_pose.pose.orientation.w = 1.0
        
        # 添加一个 2米 x 2米宽，4厘米厚的隐形碰撞盒子作为地面
        self.scene.add_box("absolute_ground", ground_pose, size=(2.0, 2.0, 0.04))
        rospy.loginfo("🌍 已将隐形刚体地面 [absolute_ground] 强制加入 MoveIt 碰撞空间！")
        # =======================================================

        self.is_running_task = False
        self.stop_requested = False
        # =======================================================
        # 🚀 进阶优化：更改底层路径规划算法
        # =======================================================
        # 1. 增加规划时间（默认只有 5 秒，高级算法需要更多时间来优化路径）
        self.arm.move_group.set_planning_time(5.0) 
        # =======================================================
        # 💥 核心修复：放宽关节执行的容忍度，防止 Gazebo 物理引擎报错中止
        # =======================================================
        self.arm.move_group.set_goal_position_tolerance(0.01)  # 末端位置容忍 1cm 误差
        self.arm.move_group.set_goal_orientation_tolerance(0.05) # 姿态容忍 0.05 弧度误差
        self.arm.move_group.set_goal_joint_tolerance(0.05)     # 关节角度容忍 0.05 弧度误差
        # =======================================================

        # 2. 更改规划器 ID (下面列出了几种常用的，你可以取消注释你想用的那个)
        # self.arm.move_group.set_planner_id("RRTConnectkConfigDefault") # MoveIt默认，速度极快，但路径随机且扭曲
        # self.arm.move_group.set_planner_id("RRTstarkConfigDefault")    # RRT* (推荐)：寻找最短/最优路径，动作最像人类，但计算稍慢
        # self.arm.move_group.set_planner_id("PRMstarkConfigDefault")  # PRM*：构建路线图，适合在复杂狭窄的货架内穿梭
        # self.arm.move_group.set_planner_id("CHOMP")                  # CHOMP：专注于轨迹平滑和避障优化的算法
        # =======================================================


        # =======================================================
        # 📦 改进 1：定义模拟周转箱 (Basket) 的位置
        # =======================================================
        self.basket_x = 0.45  
        self.basket_y = -0.1
        self.basket_z = 0.23
        self.drop_counter = 0  # 放置计数器，用于偏移计算

        self.grasp_pose_array_received = None 
        self.grasp_infos = []

        # 订阅/发布话题
        self.grasp_sub = rospy.Subscriber('/graspnet/grasp_pose_array', PoseArray, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber('/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)
        self.cmd_sub = rospy.Subscriber('/demo/command', String, self.ui_command_callback)

        # =======================================================
        # 📡 改进 2：新增任务状态汇报话题，与未来的调度器进行闭环握手
        # =======================================================
        self.status_pub = rospy.Publisher('/demo/task_status', String, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo(">> 机械臂后台执行器已就绪！")
        rospy.loginfo(">> 等待调度器下发任务...")
        rospy.loginfo("=" * 50)

    def ui_command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd in ['home', 'stop', 'h']:
            rospy.logwarn("!!! 收到干预指令：紧急刹车并回 Home !!!")
            self.stop_requested = True
            self.arm.move_group.stop()  
            self.is_running_task = False
            self.arm.go_to_home()
            self.status_pub.publish("FAILED_BY_USER") # 告知调度器任务被人工打断
        elif cmd == 'quit':
            rospy.logwarn("收到退出指令，正在关闭后台服务...")
            rospy.signal_shutdown("UI 请求退出")

    def grasp_pose_callback(self, msg):
        if not self.is_running_task:
            self.grasp_pose_array_received = msg

    def grasp_info_callback(self, msg):
        if not self.is_running_task:
            self.grasp_infos = msg.data

    def transform_pose(self, input_pose, target_frame="world"):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            return tf2_geometry_msgs.do_transform_pose(input_pose, transform)
        except Exception as e:
            return None

# ---------------- 核心逻辑：强制水平抓取修正 ----------------
    def pick_object(self):
        rospy.loginfo("[1/6] 正在等待 GTSP 下发的抓取姿态...")
        wait_count = 0
        while self.grasp_pose_array_received is None:
            rospy.sleep(0.5)
            wait_count += 1
            if wait_count > 60:
                rospy.logerr("等待超时！未收到候选抓取姿态。")
                return False

        planning_frame = self.robot.get_planning_frame()

        rospy.loginfo("🔍 开始候选抓取姿态评估（保留原始方向 + 二次评分）...")

        candidate_pool = []

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
                target_pose_stamped.pose.orientation.w
            ]
            mat_orig = tft.quaternion_matrix(q_orig)

            # GraspNet 默认 X 轴朝向物体
            raw_grasp_x = mat_orig[:3, 0]

            # ==========================================================
            # 关键改动 1：不再粗暴清零 z 分量，只做“轻度水平化”
            # 做法：保留原始方向，但限制其俯仰太大时才跳过
            # ==========================================================
            raw_norm = np.linalg.norm(raw_grasp_x)
            if raw_norm < 1e-6:
                rospy.logwarn(f"候选姿态 {i+1}: 原始抓取方向异常，跳过")
                continue

            raw_grasp_x = raw_grasp_x / raw_norm

            # 与水平面的夹角，z 分量越大表示越“扎上扎下”
            tilt_abs = abs(raw_grasp_x[2])

            # 太竖直的抓取不要
            if tilt_abs > 0.45:
                rospy.logwarn(f"候选姿态 {i+1}: 抓取方向俯仰过大(z={raw_grasp_x[2]:.3f})，跳过")
                continue

            # 只做轻微水平化：保留一部分 z 信息，而不是直接置 0
            softened = raw_grasp_x.copy()
            softened[2] *= 0.35
            norm = np.linalg.norm(softened)
            if norm < 1e-6:
                rospy.logwarn(f"候选姿态 {i+1}: 轻度水平化后方向退化，跳过")
                continue

            robot_z = softened / norm

            # 固定手背朝上，但保留 robot_z 的真实方向
            world_down = np.array([0.0, 0.0, -1.0])
            robot_y = np.cross(robot_z, world_down)
            y_norm = np.linalg.norm(robot_y)
            if y_norm < 1e-6:
                rospy.logwarn(f"候选姿态 {i+1}: robot_y 退化，跳过")
                continue
            robot_y = robot_y / y_norm
            robot_x = np.cross(robot_y, robot_z)
            robot_x = robot_x / np.linalg.norm(robot_x)

            new_rot_mat = np.eye(4)
            new_rot_mat[:3, 0] = robot_x
            new_rot_mat[:3, 1] = robot_y
            new_rot_mat[:3, 2] = robot_z
            q_new = tft.quaternion_from_matrix(new_rot_mat)

            target_pose = copy.deepcopy(target_pose_stamped.pose)
            target_pose.orientation.x = q_new[0]
            target_pose.orientation.y = q_new[1]
            target_pose.orientation.z = q_new[2]
            target_pose.orientation.w = q_new[3]

            # 目标点不要太低
            if target_pose.position.z < 0.08:
                target_pose.position.z = 0.08

            # ==========================================================
            # 关键改动 2：预抓取距离保守一点，先别后退太猛
            # ==========================================================
            back_distance = 0.10
            pre_grasp_pose = copy.deepcopy(target_pose)
            pre_grasp_pose.position.x -= robot_z[0] * back_distance
            pre_grasp_pose.position.y -= robot_z[1] * back_distance
            pre_grasp_pose.position.z -= robot_z[2] * back_distance

            # ==========================================================
            # IK/规划可达性验证
            # ==========================================================
            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.set_pose_target(pre_grasp_pose)

            plan_result = self.arm.move_group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                success = len(plan_result.joint_trajectory.points) > 0
                plan = plan_result

            self.arm.move_group.clear_pose_targets()

            if not success:
                rospy.logwarn(f"❌ 候选姿态 {i+1} 不可达或会撞货架，被淘汰")
                continue

            # ==========================================================
            # 关键改动 3：不再第一个成功就 break，先存起来再二次评分
            # ==========================================================
            if len(self.grasp_infos) > i * 3:
                raw_width = self.grasp_infos[i * 3]
            else:
                raw_width = 0.04

            # 宽度太离谱的先弱过滤
            # if raw_width < 0.015 or raw_width > 0.075:
            #     rospy.logwarn(f"候选姿态 {i+1}: width={raw_width:.3f} 不合理，跳过")
            #     continue

            # 评分思路：
            # 1) 偏好 z 分量更小的“更平稳”方向
            # 2) 偏好宽度更接近常见圆柱抓取区间（这里用 0.05 当参考）
            # 3) 偏好预抓取点离目标不要太远
            width_ref = 0.05
            score = 0.0
            score -= 2.0 * abs(robot_z[2])              # 俯仰越大越扣分
            score -= 3.0 * abs(raw_width - width_ref)   # 宽度偏离参考值越多越扣分
            score -= 0.5 * back_distance                # 预抓取过远略扣分

            candidate_pool.append({
                "idx": i,
                "score": score,
                "plan": plan,
                "pre_grasp_pose": pre_grasp_pose,
                "target_pose": target_pose,
                "robot_z": robot_z.copy(),
                "chosen_width": raw_width
            })

            rospy.loginfo(
                f"✅ 候选 {i+1} 可达: score={score:.3f}, "
                f"width={raw_width:.3f}, robot_z=({robot_z[0]:.3f},{robot_z[1]:.3f},{robot_z[2]:.3f})"
            )

        if not candidate_pool:
            rospy.logerr("所有候选姿态都不可达或质量过差！")
            self.grasp_pose_array_received = None
            return False

        # ==========================================================
        # 关键改动 4：选“评分最高”的，而不是第一个成功的
        # ==========================================================
        candidate_pool.sort(key=lambda c: c["score"], reverse=True)
        best = candidate_pool[0]

        best_plan = best["plan"]
        best_pre_grasp_pose = best["pre_grasp_pose"]
        best_target_pose = best["target_pose"]
        best_robot_z = best["robot_z"]
        chosen_width = best["chosen_width"]

        rospy.loginfo(
            f"🏆 选中候选 {best['idx']+1}: "
            f"score={best['score']:.3f}, width={chosen_width:.3f}"
        )

        # ==========================================================
        # 正式执行
        # ==========================================================
        self.gripper.open(width=0.08)

        rospy.loginfo("[4/6] 移动到柜外预抓取点...")
        if not self.arm.move_group.execute(best_plan, wait=True):
            rospy.logerr("移动到预抓取点失败！")
            return False

        self.arm.move_group.stop()
        rospy.sleep(0.5)

        rospy.loginfo("[5/6] 沿当前选中方向平稳插入...")

        # 先不要插太深，减少“带偏插入”
        extra_depth = 0.01
        in_dx = best_robot_z[0] * (back_distance + extra_depth)
        in_dy = best_robot_z[1] * (back_distance + extra_depth)
        in_dz = best_robot_z[2] * (back_distance + extra_depth)

        if not self.arm.cartesian_move_relative(
            dx=in_dx, dy=in_dy, dz=in_dz, avoid_collisions=False
        ):
            rospy.logwarn("🛑 直线插入失败！")
            return False

        rospy.sleep(0.5)

        # 这一步先保留你现在逻辑，先观察“是否更居中”
        target_width = max(0.02, min(chosen_width - 0.013, 0.065))
        rospy.loginfo(f"[6/6] 闭合夹爪 (目标宽度: {target_width:.3f}m)")
        self.gripper.close(width=0.03, force=35.0)
        rospy.sleep(1.0)

        rospy.loginfo(">>> 保持当前方向原路撤出...")
        if not self.arm.cartesian_move_relative(
            dx=-in_dx, dy=-in_dy, dz=-in_dz, avoid_collisions=False
        ):
            rospy.logwarn("🛑 撤出失败！")
            return False

        rospy.sleep(0.5)

        rospy.loginfo(">>> 完全离开柜子，安全抬高...")
        self.arm.cartesian_move_relative(dx=0.0, dy=0.0, dz=0.12)

        self.grasp_pose_array_received = None
        self.grasp_infos = []

        return True

    def place_object(self):
        if self.stop_requested: return False
        
        current_pose = self.arm.move_group.get_current_pose().pose
        target = copy.deepcopy(current_pose)
        
        # =======================================================
        # 📦 改进 3：基于用户指定坐标 (0.45, -0.1, 0.08) 的 Y 轴偏移逻辑
        # =======================================================
        # self.basket_x = 0.45
        # self.basket_y = -0.1
        # self.basket_z = 0.08
        
        # 每次放置，y 轴向左偏移 10cm (你也可以根据物体宽度调整为 0.08)
        # 第一次在 -0.1, 第二次在 0.0, 第三次在 0.1 ...
        offset_y = self.drop_counter * 0.10 
        
        target.position.x = self.basket_x
        target.position.y = self.basket_y + offset_y
        target.position.z = self.basket_z + 0.15 # 保持一个 15cm 的安全上方高度进行对准
        
        rospy.loginfo(f"正在规划移动到放置点: x={target.position.x:.2f}, y={target.position.y:.2f}, z={target.position.z:.2f}")
        
        # 1. 先移动到放置点的上方
        self.arm.move_group.set_pose_target(target)
        if not self.arm.move_group.go(wait=True):
            rospy.logerr("❌ 无法到达放置点上方，请检查是否撞到货架或超出限位")
            return False
            
        # 2. 垂直下降到最终高度 (0.08)
        self.arm.cartesian_move_relative(dz=-0.15)
        
        # 3. 松开夹爪
        self.gripper.open()
        rospy.sleep(0.5)
        
        # 4. 放置完成后计数器+1，并向上抬起撤出
        self.drop_counter += 1
        self.arm.cartesian_move_relative(dz=0.20)
        return True

    # =======================================================
    # 💥 缩进修复：必须确保这个函数严格缩进在 PickPlaceDemo 类内部！
    # =======================================================
    def run_service(self):
        self.arm.go_to_home()
        while not rospy.is_shutdown():
            self.stop_requested = False
            
            if self.grasp_pose_array_received is None:
                rospy.sleep(0.1)
                continue
            
            rospy.loginfo(">>> 开始执行物理抓取序列...")
            self.is_running_task = True
            
            # 💡 新增一个变量来记录任务到底成没成功
            task_success = False 
            
            try:
                if self.pick_object() and self.place_object():
                    rospy.loginfo("✅ 单次抓取-放置物理动作执行完毕！")
                    task_success = True
                else:
                    rospy.logwarn("⚠️ 任务失败或中止。")
            except Exception as e:
                rospy.logerr(f"异常: {e}")
            
            # 💥 必须先回 Home，先清空变量，彻底打扫完战场！
            self.arm.go_to_home()
            self.grasp_pose_array_received = None
            self.grasp_infos = []
            self.is_running_task = False
            rospy.loginfo(">>> 已彻底复位，准备向调度器请求/接收新任务...")
            
            # 💥 最后一步：安全地发送反馈信号！
            # 此时发送 DONE，调度器瞬间发来的新任务就绝对不会被前面的代码误删了。
            if task_success:
                self.status_pub.publish("DONE")
            else:
                self.status_pub.publish("FAILED")

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
        except RuntimeError as e:
            rospy.sleep(2.0)
        except Exception as e:
            rospy.logerr(f"💥 main循环发生致命崩溃！错误信息: {e}")
            break

if __name__ == "__main__":
    main()