#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import rospy
import numpy as np
import moveit_commander
import moveit_commander.planning_scene_interface
import moveit_commander.robot
from motion_controller import ArmController, GripperActionController

import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft 

class PickPlaceDemo:
    def __init__(self, arm, gripper):
        self.arm = arm
        self.gripper = gripper

        # 初始化 MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene.remove_world_object() 

        # 添加地面
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()
        ground_pose.pose.orientation.w = 1.0
        ground_pose.pose.position.z = -0.01 
        self.scene.add_box("ground", ground_pose, size=(3, 3, 0.01))

        # 放置位置参数
        self.place_x = 0.5
        self.place_y = 0.0
        self.place_z = 0.2

        # 数据缓存
        self.grasp_pose_received = None 
        self.grasp_width = 0.04 

        # 订阅话题
        self.grasp_sub = rospy.Subscriber(
            '/graspnet/grasp_pose', PoseStamped, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber(
            '/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)

        rospy.loginfo(">>> 已订阅 /graspnet 相关话题，等待检测结果...")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)

        rospy.loginfo("=" * 50)
        rospy.loginfo("系统初始化完成，准备执行【稳健重构抓取】模式！")
        rospy.loginfo("=" * 50)

    def grasp_pose_callback(self, msg):
        self.grasp_pose_received = msg

    def grasp_info_callback(self, msg):
        if len(msg.data) >= 1:
            self.grasp_width = msg.data[0]

    def transform_pose(self, input_pose, target_frame="world"):
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

    # ---------------- 核心逻辑：姿态重构 (Reconstruct Pose) ----------------
    def pick_object(self):
        rospy.loginfo("[1/6] 正在等待 GraspNet 信号...")
        wait_count = 0
        while self.grasp_pose_received is None:
            rospy.sleep(0.5)
            wait_count += 1
            if wait_count > 60: 
                rospy.logerr("等待超时！")
                return False
        
        # 1. 坐标系转换
        raw_pose = self.grasp_pose_received
        planning_frame = self.robot.get_planning_frame()
        target_pose_stamped = self.transform_pose(raw_pose, target_frame=planning_frame)
        if target_pose_stamped is None: return False

        # =========================================================================
        # [步骤 1] 提取原始信息
        # =========================================================================
        q_orig = [
            target_pose_stamped.pose.orientation.x,
            target_pose_stamped.pose.orientation.y,
            target_pose_stamped.pose.orientation.z,
            target_pose_stamped.pose.orientation.w
        ]
        mat_orig = tft.quaternion_matrix(q_orig) 
        
        # 提取 GraspNet 的进给方向 (X轴)
        raw_grasp_x = mat_orig[:3, 0] 
        # 归一化，确保它是单位向量
        raw_grasp_x = raw_grasp_x / np.linalg.norm(raw_grasp_x)

        # =========================================================================
        # [步骤 2] 姿态重构：保留 3D 进给方向，强制手腕水平
        # =========================================================================
        # 1. 设定机械臂的进给方向 (Z轴) = GraspNet 的 X 轴
        # 这里我们【不】拍扁它，保留 3D 角度，这样更灵活
        robot_z = raw_grasp_x 

        # 2. 策略：强制 Robot X (侧面/手背) 指向下方 [0, 0, -1]
        # 这样可以保证夹爪是水平夹持的 (Side Grasp)，且手腕不会扭曲
        force_robot_x = np.array([0.0, 0.0, -1.0]) 
        
        # (保护逻辑) 如果进给方向也是垂直的，X 就不能向下，改为向前
        if abs(robot_z[2]) > 0.95:
            force_robot_x = np.array([1.0, 0.0, 0.0])

        # 3. 计算 Robot Y (叉乘: Z x X)
        robot_y = np.cross(robot_z, force_robot_x) 
        robot_y = robot_y / np.linalg.norm(robot_y)
        
        # 4. 重新计算 Robot X (Y cross Z) 保证严格正交
        robot_x = np.cross(robot_y, robot_z)
        robot_x = robot_x / np.linalg.norm(robot_x)
        
        # 5. 组装旋转矩阵
        new_rot_mat = np.eye(4)
        new_rot_mat[:3, 0] = robot_x
        new_rot_mat[:3, 1] = robot_y
        new_rot_mat[:3, 2] = robot_z
        # 位置保持不变
        new_rot_mat[:3, 3] = [
            target_pose_stamped.pose.position.x,
            target_pose_stamped.pose.position.y,
            target_pose_stamped.pose.position.z
        ]

        # 6. 更新 Pose
        q_new = tft.quaternion_from_matrix(new_rot_mat)
        target_pose_stamped.pose.orientation.x = q_new[0]
        target_pose_stamped.pose.orientation.y = q_new[1]
        target_pose_stamped.pose.orientation.z = q_new[2]
        target_pose_stamped.pose.orientation.w = q_new[3]

        rospy.loginfo(f"[2/6] 姿态重构完成: Z轴进给，手腕水平。")

        # =========================================================================
        # [步骤 3] 计算预抓取点 (纯向量计算，无魔法数字)
        # =========================================================================
        approach_vector = robot_z 
        back_distance = 0.12 # 后退 12cm
        
        pre_grasp_pose = copy.deepcopy(target_pose_stamped.pose)
        
        # 严格沿着进给向量后退
        # pre_grasp_pose.position.x -= approach_vector[0] * back_distance
        pre_grasp_pose.position.y -= approach_vector[1] * back_distance
        # pre_grasp_pose.position.z -= approach_vector[2] * back_distance
        pre_grasp_pose.position.z += 0.03
        # 最低高度保护 (防止预抓取点钻地)
        if pre_grasp_pose.position.z < 0.05: pre_grasp_pose.position.z = 0.05
        
        rospy.loginfo(f"[3/6] 计算预抓取点: 沿向量后退 {back_distance*100:.0f}cm")

        # 4. 打开夹爪
        self.gripper.open()

        # 5. 移动到预抓取点
        rospy.loginfo("[4/6] 移动到预抓取点...")
        # 增加一点规划时间给复杂路径
        self.arm.move_group.set_planning_time(5.0)
        
        if not self.arm.move_to_pose_target(pre_grasp_pose):
            rospy.logerr("预抓取点不可达 (可能是超出工作空间或碰撞)")
            return False
        
        rospy.sleep(0.5)
        
        # 6. 直线插入 (Cartesian Insertion)
        rospy.loginfo(f"[5/6] 直线插入抓取...")
        target_pose_stamped.pose.position.y += 0.04
        target_pose_stamped.pose.position.z = pre_grasp_pose.position.z
        # 使用 MoveIt 的笛卡尔路径规划，比 cartesian_move_relative 更稳
        waypoints = [target_pose_stamped.pose]

            # 移除 0.0，只保留 步长(0.01)。因为你的环境不支持 float 类型的 jump_threshold 参数
        (plan, fraction) = self.arm.move_group.compute_cartesian_path(
            waypoints, 0.01)
        
        if fraction < 0.9:
            rospy.logwarn(f"笛卡尔规划不完整 ({fraction*100}%)，尝试PTP移动...")
            if not self.arm.move_to_pose_target(target_pose_stamped.pose):
                rospy.logerr("最终抓取点不可达！")
                return False
        else:
            self.arm.move_group.execute(plan, wait=True)
        
        # 7. 闭合夹爪
        target_width = max(0.01, self.grasp_width - 0.05)
        rospy.loginfo(f"[6/6] 闭合夹爪 (目标: {target_width:.3f}m)")
        self.gripper.close(width=target_width, force=40.0) 
        rospy.sleep(0.5)

        # 8. 抬起
        rospy.loginfo(">>> 抬起物体")
        self.arm.cartesian_move_relative(dz=0.15)
        return True

    def place_object(self):
        rospy.loginfo(f"\n>>> 开始放置 -> ({self.place_x}, {self.place_y}, {self.place_z})")
        
        current_pose = self.arm.move_group.get_current_pose().pose
        target_place_pose = copy.deepcopy(current_pose)
        target_place_pose.position.x = self.place_x
        target_place_pose.position.y = self.place_y
        target_place_pose.position.z = self.place_z + 0.15 
        
        if not self.arm.move_to_pose_target(target_place_pose):
            rospy.logerr("放置点上方不可达")
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
            else:
                rospy.logerr("放置失败")
        else:
            rospy.logerr("抓取失败")

def main():
    try:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("pick_place_demo_robust", anonymous=True)
        arm = ArmController("panda_manipulator")
        gripper = GripperActionController()
        demo = PickPlaceDemo(arm, gripper)
        demo.run()
    except Exception as e:
        rospy.logerr(f"运行出错: {e}")

if __name__ == "__main__":
    main()