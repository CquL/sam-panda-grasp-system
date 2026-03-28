#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import tf.transformations

class GraspClient:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_grasp_client', anonymous=True)

        # 1. 初始化机械臂控制组 (根据您的配置，可能是 'panda_arm' 或 'arm')
        self.group_name = "panda_arm" 
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        
        # 2. 初始化夹爪控制组 (通常是 'hand' 或 'panda_hand')
        self.hand_group_name = "hand"
        self.hand_group = moveit_commander.MoveGroupCommander(self.hand_group_name)

        # 订阅 GraspNet 发布的最佳抓取姿态
        self.grasp_sub = rospy.Subscriber('/graspnet/grasp_pose', PoseStamped, self.grasp_cb)
        self.received_pose = None
        
        rospy.loginfo("等待 GraspNet 发布抓取姿态...")

    def grasp_cb(self, msg):
        # 只接收一次姿态，避免重复执行
        if self.received_pose is None:
            rospy.loginfo("收到抓取姿态！准备执行...")
            self.received_pose = msg
            self.execute_grasp(msg)

    def execute_grasp(self, grasp_pose):
        # A. 打开夹爪
        self.hand_group.set_named_target("open")
        self.hand_group.go(wait=True)

        # B. 移动到预抓取位置 (抓取点上方 10cm)
        pre_grasp_pose = geometry_msgs.msg.Pose()
        pre_grasp_pose.orientation = grasp_pose.pose.orientation
        pre_grasp_pose.position.x = grasp_pose.pose.position.x
        pre_grasp_pose.position.y = grasp_pose.pose.position.y
        pre_grasp_pose.position.z = grasp_pose.pose.position.z + 0.15 # 抬高 15cm

        rospy.loginfo("正在前往预抓取位置...")
        self.move_group.set_pose_target(pre_grasp_pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
        if not success:
            rospy.logerr("预抓取移动失败！")
            return

        # C. 直线下降抓取
        rospy.loginfo("正在执行抓取下降...")
        waypoints = []
        wpose = self.move_group.get_current_pose().pose
        wpose.position.z -= 0.15 # 下降 15cm
        waypoints.append(wpose)

        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        self.move_group.execute(plan, wait=True)

        # D. 关闭夹爪
        self.hand_group.set_named_target("close")
        self.hand_group.go(wait=True)
        rospy.sleep(1.0)

        # E. 抬起物体
        rospy.loginfo("抓取完成，抬起物体...")
        wpose.position.z += 0.2
        waypoints = [wpose]
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        self.move_group.execute(plan, wait=True)

        rospy.loginfo("任务完成！")
        # 重置接收状态，如果想连续抓取可以删掉这行
        # self.received_pose = None 

if __name__ == '__main__':
    try:
        client = GraspClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass