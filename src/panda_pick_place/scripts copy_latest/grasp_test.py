#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib

from franka_gripper.msg import (
    MoveAction,
    MoveGoal,
    GraspAction,
    GraspGoal,
    GraspEpsilon
)


def main():
    rospy.init_node("test_grasp_action")

    # 1. 连接两个 action server：move（张开）和 grasp（抓取）
    rospy.loginfo("等待 /franka_gripper/move 和 /franka_gripper/grasp 服务器...")

    move_client = actionlib.SimpleActionClient(
        "/franka_gripper/move", MoveAction
    )
    grasp_client = actionlib.SimpleActionClient(
        "/franka_gripper/grasp", GraspAction
    )

    move_client.wait_for_server()
    grasp_client.wait_for_server()
    rospy.loginfo("✓ 已连接到 gripper action 服务器")

    # 2. 先把爪子张开一点（比如 8cm）
    open_goal = MoveGoal()
    open_goal.width = 0.08       # 8 cm 张开
    open_goal.speed = 0.1        # 张开的速度

    rospy.loginfo("-> 发送张开命令: width=%.3f" % open_goal.width)
    move_client.send_goal(open_goal)
    finished = move_client.wait_for_result(rospy.Duration(5.0))
    if not finished:
        rospy.logwarn("move action 超时/失败")
    else:
        rospy.loginfo("move 结果: %s", str(move_client.get_result()))

    rospy.sleep(2.0)

    # 3. 再发送 grasp 动作（假设中间你手动把一个方块放到爪子中间）
    grasp_goal = GraspGoal()
    grasp_goal.width = 0.03          # 最终闭合的宽度（3cm 左右）
    grasp_goal.speed = 0.05          # 闭合速度
    grasp_goal.force = 40.0          # 抓取力（N）

    # 允许的宽度误差
    grasp_goal.epsilon = GraspEpsilon()
    grasp_goal.epsilon.inner = 0.005
    grasp_goal.epsilon.outer = 0.005

    rospy.loginfo(
        "-> 发送 grasp 命令: width=%.3f, force=%.1f"
        % (grasp_goal.width, grasp_goal.force)
    )
    grasp_client.send_goal(grasp_goal)
    finished = grasp_client.wait_for_result(rospy.Duration(10.0))

    if not finished:
        rospy.logwarn("grasp action 超时/失败")
    else:
        result = grasp_client.get_result()
        rospy.loginfo("grasp 结果: %s", str(result))

    rospy.loginfo("测试结束，退出节点。")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
