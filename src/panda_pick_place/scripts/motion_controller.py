#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tft
import actionlib
from sensor_msgs.msg import JointState
from franka_gripper.msg import (
    MoveAction, MoveGoal,
    GraspAction, GraspGoal, GraspEpsilon
)

class ArmController:
    """封装机械臂的 MoveIt 控制"""

    def __init__(self, group_name="panda_arm"):
        self.group_name = group_name
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # ====== 新增：规划器相关配置 ======
        # 从参数服务器读取 planner_id（例如：RRTConnectkConfigDefault）
        # 如果不设，则使用 MoveIt 默认规划器
        self.planner_id = rospy.get_param("~planner_id", "")
        if self.planner_id:
            self.move_group.set_planner_id(self.planner_id)
            rospy.loginfo(f"[ArmController] 使用规划器: {self.planner_id}")
        else:
            rospy.loginfo("[ArmController] 使用 MoveIt 默认规划器")
        # =================================

        # 参数配置与原代码保持一致
        self.move_group.set_planning_time(10.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.allow_replanning(True)
        self.move_group.set_max_velocity_scaling_factor(0.5)
        self.move_group.set_max_acceleration_scaling_factor(0.5)

        self.eef_link = self.move_group.get_end_effector_link()

    # ====== 新增：对外提供更换规划器的接口 ======
    def set_planner(self, planner_id: str):
        """
        动态切换规划器，例如:
            arm.set_planner("RRTConnectkConfigDefault")
        """
        if not planner_id:
            rospy.logwarn("[ArmController] planner_id 为空，忽略切换")
            return False

        self.move_group.set_planner_id(planner_id)
        self.planner_id = planner_id
        rospy.loginfo(f"[ArmController] 已切换规划器为: {planner_id}")
        return True
    # ==========================================

# ====== 【新增】直接接收 Pose 对象的方法 (用于 GraspNet) ======
    def move_to_pose_target(self, pose):
        """
        直接移动到 geometry_msgs/Pose 目标
        支持四元数输入，适配 GraspNet 的输出
        """
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose)

        rospy.loginfo(f">>> 规划移动到目标 Pose: {pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f}")
        
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            rospy.loginfo(" ✓ 机械臂移动成功")
        else:
            rospy.logerr(" ✗ 机械臂移动失败")
        return success
    # ========================================================
    
    def move_to_pose(self, x, y, z, roll=0.0, pitch=3.14, yaw=0.0):
        """移动到指定位姿（默认末端朝下）"""
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        q = tft.quaternion_from_euler(roll, pitch, yaw)
        pose_goal.orientation.x = q[0]
        pose_goal.orientation.y = q[1]
        pose_goal.orientation.z = q[2]
        pose_goal.orientation.w = q[3]

        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose_goal)

        rospy.loginfo(
            f">>> 规划移动到: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            f"planner={self.planner_id if self.planner_id else 'default'}"
        )
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            rospy.loginfo(" ✓ 机械臂移动成功")
        else:
            rospy.logerr(" ✗ 机械臂移动失败")
        return success
    def cartesian_move_relative(self, dx=0.0, dy=0.0, dz=0.0,
                                eef_step=0.01, avoid_collisions=True):
        """cartesian_move_relative
        从当前位置走一条笛卡尔直线轨迹（相对移动）
        dx, dy, dz: 相对于当前末端位置的偏移量（米）
        eef_step: 轨迹插值步长（米），越小越平滑
        avoid_collisions: 是否在规划时考虑碰撞
        """
        waypoints = []

        # 当前 pose
        start_pose = self.move_group.get_current_pose().pose

        # 目标 pose = 当前 + 相对位移
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = start_pose.position.x + dx
        target_pose.position.y = start_pose.position.y + dy
        target_pose.position.z = start_pose.position.z + dz
        target_pose.orientation = start_pose.orientation  # 姿态保持不变

        waypoints.append(target_pose)

        # 注意：第三个参数必须是 bool（你这个 MoveIt 版本）
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step,
            avoid_collisions  # 这里传 True / False，而不是 0.0
        )

        # ==========================================
        # 【核心修复】严格拒绝不完整的危险路径
        # ==========================================
        # 考虑到浮点数精度，我们要求 fraction 必须大于 0.95 (即95%以上) 才算成功
        if fraction < 0.95:
            rospy.logerr(f"[ArmController] 🛑 致命安全拦截：直线路径被障碍物阻挡！只规划了 {fraction*100:.1f}%。为防止碰撞，拒绝执行插入！")
            return False

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        return True



    def go_to_home(self):
        """回到初始位置"""
        rospy.loginfo("\n>>> 回到初始位置 "
                      f"(planner={self.planner_id if self.planner_id else 'default'})")
        joint_goal = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        self.move_group.set_start_state_to_current_state()
        success = self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        rospy.loginfo("go_to_home success? %s", success)
        rospy.loginfo("✓ 已回到初始位置\n")
        return success


class GripperController:
    """封装夹爪的 MoveIt 控制"""

    def __init__(self, group_name="panda_hand"):
        self.group_name = group_name
        self.hand_group = moveit_commander.MoveGroupCommander(group_name)

    def open(self):
        """打开夹爪"""
        rospy.loginfo(">>> 打开夹爪")
        joint_goal = self.hand_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            joint_goal[0] = 0.04
            joint_goal[1] = 0.04
        success = self.hand_group.go(joint_goal, wait=True)
        self.hand_group.stop()
        rospy.sleep(0.3)
        rospy.loginfo("open_gripper success? %s", success)
        return success

    def close(self):
        """关闭夹爪"""
        rospy.loginfo(">>> 关闭夹爪")
        joint_goal = self.hand_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            joint_goal[0] = 0.03
            joint_goal[1] = 0.03
        success = self.hand_group.go(joint_goal, wait=True)
        self.hand_group.stop()
        rospy.sleep(0.3)
        rospy.loginfo("close_gripper success? %s", success)
        return success


class GripperActionController:
    """
    使用 franka_gripper 的 Move / Grasp Action 控制夹爪
    open() -> /franka_gripper/move
    close() -> /franka_gripper/grasp
    """

    def __init__(self,
                 move_action_name="/franka_gripper/move",
                 grasp_action_name="/franka_gripper/grasp"):

        rospy.loginfo("[GripperActionController] 等待 franka_gripper action server ...")
        self.move_client = actionlib.SimpleActionClient(
            move_action_name, MoveAction
        )
        self.grasp_client = actionlib.SimpleActionClient(
            grasp_action_name, GraspAction
        )

        self.move_client.wait_for_server()
        self.grasp_client.wait_for_server()
        rospy.loginfo("[GripperActionController] ✓ 已连接到 move / grasp action server")
        self.grasp_epsilon_inner = float(rospy.get_param("~grasp_epsilon_inner", 0.020))
        self.grasp_epsilon_outer = float(rospy.get_param("~grasp_epsilon_outer", 0.020))
        self.grasp_timeout_sec = float(rospy.get_param("~grasp_timeout_sec", 4.0))
        self.soft_grasp_width_tolerance = float(rospy.get_param("~soft_grasp_width_tolerance", 0.030))
        self.soft_grasp_min_width = float(rospy.get_param("~soft_grasp_min_width", 0.035))
        self.soft_grasp_max_width = float(rospy.get_param("~soft_grasp_max_width", 0.078))
        self.latest_joint_state = None
        self.last_grasp_was_soft_success = False
        self.last_grasp_hard_success = False
        self.last_observed_gripper_width = None
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb, queue_size=10)

    def joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def get_current_gripper_width(self):
        msg = self.latest_joint_state
        if msg is None:
            return None
        pos_map = {}
        try:
            for name, pos in zip(msg.name, msg.position):
                pos_map[name] = float(pos)
        except Exception:
            return None

        left = pos_map.get("panda_finger_joint1")
        right = pos_map.get("panda_finger_joint2")
        if left is not None and right is not None:
            return max(0.0, left + right)
        if left is not None:
            return max(0.0, 2.0 * left)
        if right is not None:
            return max(0.0, 2.0 * right)
        return None

    def extract_width_from_error(self, error_text):
        if not error_text:
            return None
        match = re.search(r"but at\s+([0-9]*\.?[0-9]+)m", str(error_text))
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    def soft_grasp_succeeded(self, requested_width, observed_width=None):
        current_width = observed_width if observed_width is not None else self.get_current_gripper_width()
        if current_width is None:
            return False, None
        width_error = abs(float(current_width) - float(requested_width))
        occupied = self.soft_grasp_min_width <= current_width <= self.soft_grasp_max_width
        close_enough = width_error <= self.soft_grasp_width_tolerance
        return bool(occupied and close_enough), float(current_width)

    def open(self, width=0.08, speed=0.1):
        """
        张开夹爪到指定宽度（单位：米）
        默认 0.08 m = 8 cm
        """
        rospy.loginfo(f"[GripperActionController] 打开夹爪: width={width:.3f}")
        goal = MoveGoal()
        goal.width = width
        goal.speed = speed

        self.move_client.send_goal(goal)
        finished = self.move_client.wait_for_result(rospy.Duration(5.0))

        if not finished:
            rospy.logwarn("[GripperActionController] move 超时")
            return False

        result = self.move_client.get_result()
        rospy.loginfo(f"[GripperActionController] move 结果: {result}")
        if result is None:
            rospy.logwarn("[GripperActionController] move 未返回有效结果")
            return False
        if hasattr(result, "success") and not bool(result.success):
            rospy.logwarn(
                f"[GripperActionController] move 动作执行完成，但夹爪未成功张开到目标宽度: error={getattr(result, 'error', '')}"
            )
            return False
        return True

    def close(self, width=0.033, speed=0.05, force=40.0):
        """
        使用 grasp 动作关闭夹爪并尝试抓取
        width: 最终期望的爪间距
        force: 抓取力 (N)
        """
        rospy.loginfo(
            f"[GripperActionController] grasp: width={width:.3f}, "
            f"speed={speed:.3f}, force={force:.1f}"
        )
        self.last_grasp_was_soft_success = False
        self.last_grasp_hard_success = False
        self.last_observed_gripper_width = None

        goal = GraspGoal()
        goal.width = width
        goal.speed = speed
        goal.force = force

        eps = GraspEpsilon()
        eps.inner = self.grasp_epsilon_inner
        eps.outer = self.grasp_epsilon_outer
        goal.epsilon = eps

        self.grasp_client.send_goal(goal)
        finished = self.grasp_client.wait_for_result(rospy.Duration(self.grasp_timeout_sec))

        if not finished:
            soft_ok, current_width = self.soft_grasp_succeeded(width)
            self.last_observed_gripper_width = current_width
            if soft_ok:
                self.last_grasp_was_soft_success = True
                rospy.logwarn(
                    f"[GripperActionController] grasp 超时，但当前夹爪宽度 {current_width:.3f} m "
                    f"接近目标宽度 {width:.3f} m，按软成功处理并保留当前夹爪状态。"
                )
                return True
            self.grasp_client.cancel_goal()
            rospy.logwarn("[GripperActionController] grasp 超时")
            return False

        result = self.grasp_client.get_result()
        rospy.loginfo(f"[GripperActionController] grasp 结果: {result}")
        if result is None:
            rospy.logwarn("[GripperActionController] grasp 未返回有效结果")
            return False

        if hasattr(result, "success") and bool(result.success):
            self.last_grasp_hard_success = True
            self.last_observed_gripper_width = self.get_current_gripper_width()
            return True

        if hasattr(result, "success") and not bool(result.success):
            observed_width = self.extract_width_from_error(getattr(result, "error", ""))
            soft_ok, current_width = self.soft_grasp_succeeded(width, observed_width=observed_width)
            self.last_observed_gripper_width = current_width
            if soft_ok:
                self.last_grasp_was_soft_success = True
                rospy.logwarn(
                    f"[GripperActionController] grasp 返回失败，但当前夹爪宽度 {current_width:.3f} m "
                    f"接近目标宽度 {width:.3f} m，按软成功处理。"
                )
                return True
            rospy.logwarn(
                f"[GripperActionController] grasp 动作执行完成，但未形成稳定抓取: error={getattr(result, 'error', '')}"
            )
            return False

        return True
