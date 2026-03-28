#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import rospy
import numpy as np
import moveit_commander
import moveit_commander.planning_scene_interface
import moveit_commander.robot
from gazebo_interface import GazeboCubeManager
from motion_controller import ArmController, GripperActionController

import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft 
from geometry_msgs.msg import PoseArray

class PickPlaceDemo:
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper

        # 初始化 MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene.remove_world_object() 

        # 添加地面 (防止规划路径钻到地底)
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()
        ground_pose.pose.orientation.w = 1.0
        ground_pose.pose.position.z = -0.01 
        self.scene.add_box("ground", ground_pose, size=(3, 3, 0.01))

        # 放置位置参数
        self.place_x = 0.5
        self.place_y = 0.0
        self.place_z = 0.13

        # 数据缓存
        # 数据缓存
        self.grasp_pose_array_received = None  # <--- 改成带 array 的新名字！
        self.grasp_width = 0.04 
        self.grasp_infos = []

        # ============ 订阅 GraspNet 话题 ============
        # 1. 抓取姿态
        self.grasp_sub = rospy.Subscriber(
            '/graspnet/grasp_pose_array', PoseArray, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber(
            '/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)
        rospy.loginfo(">>> 已订阅 /graspnet 相关话题，等待检测结果...")
        # ==========================================================
        # 【新增】RViz 可视化专用发布者
        # ==========================================================
        self.best_target_pub = rospy.Publisher('/graspnet/best_target_pose', PoseStamped, queue_size=1)
        self.best_pre_pub = rospy.Publisher('/graspnet/best_pre_grasp_pose', PoseStamped, queue_size=1)
        
        rospy.loginfo(">>> 已订阅 /graspnet 相关话题，等待检测结果...")


        # 初始化 TF 监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)

        rospy.loginfo("=" * 50)
        rospy.loginfo("系统初始化完成，准备执行【强制水平抓取】模式！")
        rospy.loginfo("=" * 50)

    def grasp_pose_callback(self, msg):
        self.grasp_pose_array_received = msg

    def grasp_info_callback(self, msg):
        self.grasp_infos = msg.data

    def transform_pose(self, input_pose, target_frame="world"):
        """坐标系转换"""
        try:
            if input_pose.header.frame_id == target_frame:
                return input_pose
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose_transformed = tf2_geometry_msgs.do_transform_pose(input_pose, transform)
            return pose_transformed
        except Exception as e:
            rospy.logerr(f"坐标转换失败: {e}")
            return None

    # ---------------- 核心逻辑：强制水平抓取修正 ----------------
    def pick_object(self):
        rospy.loginfo("[1/6] 正在等待 GraspNet 的候选姿态序列...")
        wait_count = 0
        while self.grasp_pose_array_received is None:
            rospy.sleep(0.5)
            wait_count += 1
            if wait_count > 60: 
                rospy.logerr("等待超时！未收到候选抓取姿态。")
                return False
        
        planning_frame = self.robot.get_planning_frame()
        
        best_plan = None
        best_pre_grasp_pose = None
        best_target_pose = None
        chosen_width = 0.04

        rospy.loginfo("🔍 开始对候选姿态进行运动学与避障评估 (面试开始)...")
        
        # 遍历接收到的 Top-10 姿态
        for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
            # 将 Pose 包装成 PoseStamped 用于坐标转换
            ps = PoseStamped()
            ps.header = self.grasp_pose_array_received.header
            ps.pose = raw_pose
            
            target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
            if target_pose_stamped is None: continue

            # =========================================================================
            # [步骤 1] 提取并强制水平抓取修正 (保留你的原逻辑)
            # =========================================================================
            q_orig = [
                target_pose_stamped.pose.orientation.x,
                target_pose_stamped.pose.orientation.y,
                target_pose_stamped.pose.orientation.z,
                target_pose_stamped.pose.orientation.w
            ]
            mat_orig = tft.quaternion_matrix(q_orig) 
            raw_grasp_x = mat_orig[:3, 0] 
            
            flat_approach = np.array([raw_grasp_x[0], raw_grasp_x[1], 0.0])
            norm = np.linalg.norm(flat_approach)
            
            if norm < 0.001:
                rospy.logwarn(f"候选姿态 {i+1}: 抓取方向几乎垂直，无法修正，淘汰！")
                continue
                
            robot_z = flat_approach / norm
            robot_x = np.array([0.0, 0.0, -1.0]) 
            robot_y = np.cross(robot_z, robot_x) 
            robot_y = robot_y / np.linalg.norm(robot_y) 
            robot_x = np.cross(robot_y, robot_z)
            
            new_rot_mat = np.eye(4)
            new_rot_mat[:3, 0] = robot_x
            new_rot_mat[:3, 1] = robot_y
            new_rot_mat[:3, 2] = robot_z
            new_rot_mat[:3, 3] = [
                target_pose_stamped.pose.position.x,
                target_pose_stamped.pose.position.y,
                target_pose_stamped.pose.position.z
            ]
            
            q_new = tft.quaternion_from_matrix(new_rot_mat)
            target_pose_stamped.pose.orientation.x = q_new[0]
            target_pose_stamped.pose.orientation.y = q_new[1]
            target_pose_stamped.pose.orientation.z = q_new[2]
            target_pose_stamped.pose.orientation.w = q_new[3]

            # =========================================================================
            # [步骤 2] 计算预抓取点 (保留你的微调参数)
            # =========================================================================
            back_distance = 0.08 
            pre_grasp_pose = copy.deepcopy(target_pose_stamped.pose)
            
            pre_grasp_pose.position.x -= robot_z[0] * back_distance
            pre_grasp_pose.position.y -= robot_z[1] * back_distance + 0.03
            
            if pre_grasp_pose.position.z < 0.05: pre_grasp_pose.position.z = 0.05
            pre_grasp_pose.position.z += 0.06

            # =========================================================================
            # [步骤 3] 全新核心逻辑：MoveIt 碰撞检测与逆向运动学(IK)验证
            # =========================================================================
            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.set_pose_target(pre_grasp_pose)
            
            # 使用 plan() 进行计算，此时机械臂不会动！
            plan_result = self.arm.move_group.plan()
            
            # 兼容不同 ROS/MoveIt 版本的 plan() 返回值
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                success = len(plan_result.joint_trajectory.points) > 0
                plan = plan_result
            
            if success:
                rospy.loginfo(f"🎉 候选姿态 {i+1} 验证通过！IK有解且完美避开环境障碍物！")
                best_plan = plan
                best_pre_grasp_pose = pre_grasp_pose
                best_target_pose = target_pose_stamped.pose
                
                # 提取匹配的 width 宽度 (防止数组越界)
                if len(self.grasp_infos) > i * 3:
                    chosen_width = self.grasp_infos[i * 3]
                else:
                    chosen_width = 0.04
                break # 找到了第一个安全的姿态，直接跳出循环！
            else:
                rospy.logwarn(f"❌ 候选姿态 {i+1} 被淘汰 (会导致撞货架或关节扭曲)")

        # === 遍历结束，检查录取结果 ===
        if best_plan is None:
            rospy.logerr("所有候选姿态都会发生碰撞或不可达，请尝试换个角度看目标！")
            self.grasp_pose_array_received = None # 清理缓存
            return False
        # =========================================================================
        # 【新增】发布最终选定的完美姿态，供 RViz 可视化调试
        # =========================================================================
        target_msg = PoseStamped()
        target_msg.header.frame_id = planning_frame
        target_msg.header.stamp = rospy.Time.now()
        target_msg.pose = best_target_pose
        self.best_target_pub.publish(target_msg)

        pre_msg = PoseStamped()           
        pre_msg.header.frame_id = planning_frame
        pre_msg.header.stamp = rospy.Time.now()
        pre_msg.pose = best_pre_grasp_pose
        self.best_pre_pub.publish(pre_msg)
        rospy.loginfo("👀 已发布最终修正姿态到 RViz，话题: /graspnet/best_target_pose")

        # =========================================================================
        # [步骤 4] 验证通过，正式驱动机械臂执行！
        # =========================================================================
        self.gripper.open()
        
        rospy.loginfo("[4/6] 正在执行无碰撞安全轨迹，移动到预抓取点...")
        exec_success = self.arm.move_group.execute(best_plan, wait=True)
        if not exec_success:
            rospy.logerr("轨迹物理执行失败！")
            return False
            
        rospy.sleep(0.5)
        
        # === 6. 直线插入 (Cartesian Insertion) ===
        best_target_pose.position.z = best_pre_grasp_pose.position.z
        dx = best_target_pose.position.x - best_pre_grasp_pose.position.x
        dy = best_target_pose.position.y - best_pre_grasp_pose.position.y + 0.04
        dz = best_target_pose.position.z - best_pre_grasp_pose.position.z
        
        rospy.loginfo(f"[5/6] 沿笛卡尔空间直线插入 (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})")
        if not self.arm.cartesian_move_relative(dx, dy, dz):
            rospy.logerr("直线插入失败！")
            return False
        
        # === 7. 闭合夹爪 ===
        target_width = max(0.0, chosen_width - 0.01)
        rospy.loginfo(f"[6/6] 闭合夹爪 (目标宽度: {target_width:.3f}m)")
        self.gripper.close(width=0.025, force=50.0) 
        rospy.sleep(0.5)

        # === 8. 抬起撤出 ===
        # === 8. 撤出并抬起 ===
        rospy.loginfo(">>> 像抽屉一样平直撤出货架...")
        
        # 2. 完全离开货架后，再往上抬起，动作才足够优雅且安全
        rospy.loginfo(">>> 离开货架，向上抬起物体...")
        self.arm.cartesian_move_relative(dz=0.12)
        # 1. 先沿着刚才伸进去的 Y 轴原路退出来 (先退够 25cm 确保离开货架本体)
        if not self.arm.cartesian_move_relative(dy=-0.25):
            rospy.logwarn("撤出时检测到碰撞风险，尝试直接回到 Home！")
            return False
            
        rospy.sleep(0.5)
        
        # 抓取结束后，清除这一轮的接收数据，准备下一次抓取
        self.grasp_pose_array_received = None
        self.grasp_infos = []
        
        return True 
    
    def place_object(self):
        rospy.loginfo(f"\n>>> 开始放置 -> ({self.place_x}, {self.place_y}, {self.place_z})")
        
        current_pose = self.arm.move_group.get_current_pose().pose
        target_place_pose = copy.deepcopy(current_pose)
        target_place_pose.position.x = self.place_x
        target_place_pose.position.y = self.place_y
        target_place_pose.position.z = self.place_z + 0.15 
        
        if not self.arm.move_to_pose_target(target_place_pose):
            return False
        
        self.arm.cartesian_move_relative(dz=-0.10)
        self.gripper.open()
        rospy.sleep(0.5)
        self.arm.cartesian_move_relative(dz=0.20)
        return True

    def run(self):
        self.arm.go_to_home()
        rospy.sleep(1.0)
        if self.pick_object():
            rospy.sleep(1.0)
            if self.place_object():
                rospy.loginfo("\n✅ 抓取放置任务圆满完成！\n")
                self.arm.go_to_home()

                return True
            else:
                rospy.logerr("放置失败")
        else:
            rospy.logerr("抓取失败")

def main():
    try:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("pick_place_demo_v2", anonymous=True)
        arm = ArmController("panda_manipulator")
        gripper = GripperActionController()
        demo = PickPlaceDemo(arm, gripper)
        demo.run()
    except Exception as e:
        rospy.logerr(f"运行出错: {e}")

if __name__ == "__main__":
    main()