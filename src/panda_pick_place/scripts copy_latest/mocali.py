#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import rospy
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Pose
import tf.transformations as tft

# Gazebo 相关
import rospkg
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_ros_link_attacher.srv import Attach, AttachRequest

from gazebo_msgs.srv import DeleteModel, DeleteModelRequest


class PickPlaceDemo:
    """Franka Panda 抓取放置类（在代码中生成方块 + link_attacher 粘块）"""

    def __init__(self):
        # 初始化 MoveIt Commander 和 ROS 节点
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pick_place_node', anonymous=True)

        # 初始化机器人接口、场景与运动规划组
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()  # 目前不往场景里加物体

        # 定义机械臂和夹爪规划组
        self.group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.hand_group = moveit_commander.MoveGroupCommander("panda_hand")

        # 获取末端执行器的链接名称
        self.eef_link = self.move_group.get_end_effector_link()

        # 设置规划参数
        self.move_group.set_planning_time(10.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.allow_replanning(True)
        self.move_group.set_max_velocity_scaling_factor(0.5)
        self.move_group.set_max_acceleration_scaling_factor(0.5)

        # ================== 从参数服务器读取位置参数 ==================
        # 建议：pick_z 和 cube_z 保持一致（都是方块中心高度）
        self.pick_x = rospy.get_param('~pick_x', 0.5)
        self.pick_y = rospy.get_param('~pick_y', 0.0)
        self.pick_z = rospy.get_param('~pick_z', 0.13)  # ★ 改成和 cube_z 一致的默认值

        self.place_x = rospy.get_param('~place_x', 0.3)
        self.place_y = rospy.get_param('~place_y', 0.3)
        self.place_z = rospy.get_param('~place_z', 0.3)

        # Gazebo 中方块生成位置（默认 = 抓取位置）
        self.cube_x = rospy.get_param('~cube_x', self.pick_x)
        self.cube_y = rospy.get_param('~cube_y', self.pick_y)
        self.cube_z = rospy.get_param('~cube_z', self.pick_z)

        # ================== Gazebo link attacher 服务 ==================
        rospy.loginfo("等待 Gazebo link attacher 服务 /link_attacher_node/attach & detach ...")
        rospy.wait_for_service('/link_attacher_node/attach')
        rospy.wait_for_service('/link_attacher_node/detach')
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        rospy.loginfo("Gazebo link attacher 服务已连接")

        # ================== Gazebo 中生成方块 ==================
        self.spawn_cube_in_gazebo()

        rospy.sleep(2.0)  # 等待一切准备好

        rospy.loginfo("=" * 50)
        rospy.loginfo("初始化完成（方块由代码生成，使用 Gazebo link_attacher 粘块）")
        rospy.loginfo(f"规划参考坐标系: {self.robot.get_planning_frame()}")
        rospy.loginfo(f"末端执行器: {self.eef_link}")
        rospy.loginfo(f"规划组: {self.group_name}")
        rospy.loginfo(f"抓取位置: ({self.pick_x}, {self.pick_y}, {self.pick_z})")
        rospy.loginfo(f"放置位置: ({self.place_x}, {self.place_y}, {self.place_z})")
        rospy.loginfo(f"方块生成位置: ({self.cube_x}, {self.cube_y}, {self.cube_z})")
        rospy.loginfo("=" * 50)

    # ---------------------------------------------------------
    # 在 Gazebo 中生成方块
    # ---------------------------------------------------------
    def spawn_cube_in_gazebo(self):
        """调用 /gazebo/spawn_sdf_model 在 Gazebo 里面生成方块"""
        rospy.loginfo("等待 /gazebo/spawn_sdf_model 服务...")
        try:
            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=20.0)
        except rospy.ROSException:
            rospy.logerr("等待 /gazebo/spawn_sdf_model 超时，请确认 Gazebo 已经启动并加载 gazebo_ros。")
            return

        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # 读取 SDF 模型文件
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('panda_pick_place')
            sdf_path = os.path.join(pkg_path, 'models', 'cube.sdf')
            rospy.loginfo(f"方块 SDF 路径: {sdf_path}")

            with open(sdf_path, 'r') as f:
                sdf_xml = f.read()
        except Exception as e:
            rospy.logerr(f"读取 cube.sdf 失败: {e}")
            return

        # 设置方块初始位姿
        initial_pose = Pose()
        initial_pose.position.x = self.cube_x
        initial_pose.position.y = self.cube_y
        initial_pose.position.z = self.cube_z
        initial_pose.orientation.x = 0.0
        initial_pose.orientation.y = 0.0
        initial_pose.orientation.z = 0.0
        initial_pose.orientation.w = 1.0

        req = SpawnModelRequest()
        req.model_name = "cube"   # Gazebo 中的模型名字，和 link_attacher 里保持一致
        req.model_xml = sdf_xml
        req.robot_namespace = ""
        req.initial_pose = initial_pose
        req.reference_frame = "world"

        try:
            resp = spawn_srv(req)
            if resp.success:
                rospy.loginfo("✓ 已在 Gazebo 中生成方块 cube")
            else:
                rospy.logwarn(f"生成方块 cube 失败，可能已存在：{resp.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"调用 /gazebo/spawn_sdf_model 出错: {e}")

    # ---------------------------------------------------------
    # Gazebo 中的附着 / 分离
    # ---------------------------------------------------------
    def attach_in_gazebo(self):
        """在 Gazebo 中把方块粘到夹爪上"""
        rospy.loginfo(">>> Gazebo: attach 物体 到夹爪")

        req = AttachRequest()

        # ★★ 确认自：/gazebo/link_states
        # 例如 name: panda::panda_leftfinger  => model_name_1="panda", link_name_1="panda_leftfinger"
        req.model_name_1 = "panda"
        req.link_name_1  = "panda_leftfinger"

        # cube.sdf: <model name="cube"> <link name="link">
        req.model_name_2 = "cube"
        req.link_name_2  = "link"

        try:
            resp = self.attach_srv.call(req)
            if resp.ok:
                rospy.loginfo("Gazebo attach 成功")
                return True
            else:
                rospy.logwarn("Gazebo attach 返回失败")
                return False
        except rospy.ServiceException as e:
            rospy.logerr("Gazebo attach 服务调用出错: %s", e)
            return False

    def detach_in_gazebo(self):
        """在 Gazebo 中把方块从夹爪上分离"""
        rospy.loginfo(">>> Gazebo: detach 物体 和夹爪")

        req = AttachRequest()
        req.model_name_1 = "panda"
        req.link_name_1  = "panda_leftfinger"
        req.model_name_2 = "cube"
        req.link_name_2  = "link"

        try:
            resp = self.detach_srv.call(req)
            if resp.ok:
                rospy.loginfo("Gazebo detach 成功")
                return True
            else:
                rospy.logwarn("Gazebo detach 返回失败")
                return False
        except rospy.ServiceException as e:
            rospy.logerr("Gazebo detach 服务调用出错: %s", e)
            return False

    # ---------------------------------------------------------
    # 夹爪控制部分
    # ---------------------------------------------------------
    def open_gripper(self):
        """打开夹爪"""
        rospy.loginfo(">>> 打开夹爪")
        joint_goal = self.hand_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            # 这个值和 URDF 里的限制配合使用，0.04 一般就是张开
            joint_goal[0] = 0.04
            joint_goal[1] = 0.04
        success = self.hand_group.go(joint_goal, wait=True)
        self.hand_group.stop()
        rospy.sleep(0.3)
        rospy.loginfo("open_gripper success? %s", success)
        return success

    def close_gripper(self):
        """关闭夹爪"""
        rospy.loginfo(">>> 关闭夹爪")
        joint_goal = self.hand_group.get_current_joint_values()
        if len(joint_goal) >= 2:
            joint_goal[0] = 0.03  # 完全闭合
            joint_goal[1] = 0.03
        success = self.hand_group.go(joint_goal, wait=True)
        self.hand_group.stop()
        rospy.sleep(0.3)
        rospy.loginfo("close_gripper success? %s", success)
        return success

    # ---------------------------------------------------------
    # 机械臂移动控制部分
    # ---------------------------------------------------------
    def move_to_pose(self, x, y, z, roll=0.0, pitch=3.14, yaw=0.0):
        """移动到指定位姿（默认末端朝下）"""
        pose_goal = geometry_msgs.msg.Pose()

        # 设置位置
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        # 设置姿态（欧拉角 → 四元数）
        quaternion = tft.quaternion_from_euler(roll, pitch, yaw)
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]

        # 每次规划前都用当前状态作为起点
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose_goal)

        rospy.loginfo(f">>> 规划移动到: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if success:
            rospy.loginfo(" ✓ 机械臂移动成功")
        else:
            rospy.logerr(" ✗ 机械臂移动失败")
        return success

    # ---------------------------------------------------------
    # 抓取与放置逻辑（MoveIt + Gazebo 粘块）
    # ---------------------------------------------------------
    def pick_object(self, x, y, z):
        """抓取物体"""
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("开始抓取流程")
        rospy.loginfo("=" * 50)

        # 1. 打开夹爪并移动到物体上方
        rospy.loginfo("[1/4] 移动到物体上方")
        if not self.open_gripper():
            rospy.logerr("✗ 打开夹爪失败，取消抓取")
            return False

        if not self.move_to_pose(x, y, z + 0.25):
            rospy.logerr("✗ 移动到预抓取位置失败")
            return False
        rospy.sleep(0.3)

        # 2. 下降至物体上方一点点（z + 0.03，保证和 cube 高度接近）
        rospy.loginfo("[2/4] 下降到物体附近")
        if not self.move_to_pose(x, y, z + 0.03):
            rospy.logerr("✗ 下降失败")
            return False
        rospy.sleep(0.3)

        # 3. 关闭夹爪抓取
        rospy.loginfo("[3/4] 关闭夹爪抓取物体")
        self.close_gripper()
        # if not self.close_gripper():
        #     rospy.logerr("✗ 夹爪执行失败，取消本次抓取")
        #     return False
        rospy.sleep(0.3)

        # 4. 在 Gazebo 里把方块粘到夹爪上
        if not self.attach_in_gazebo():
            rospy.logerr("✗ Gazebo attach 失败，取消本次抓取")
            return False

        # 5. 抬起物体
        rospy.loginfo("[4/4] 抬起物体")
        if not self.move_to_pose(x, y, z + 0.20):
            rospy.logerr("✗ 抬起失败")
            return False

        rospy.loginfo("✓ 抓取完成!")
        rospy.loginfo("=" * 50 + "\n")
        return True

    def place_object(self, x, y, z):
        """放置物体"""
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("开始放置流程")
        rospy.loginfo("=" * 50)

        # 1. 移动到放置位置上方
        rospy.loginfo("[1/4] 移动到放置位置上方")
        if not self.move_to_pose(x, y, z + 0.20):
            rospy.logerr("✗ 移动到放置位置上方失败")
            return False
        rospy.sleep(0.3)

        # 2. 下降到放置位置附近
        rospy.loginfo("[2/4] 下降到放置位置附近")
        if not self.move_to_pose(x, y, z + 0.03):
            rospy.logerr("✗ 下降失败")
            return False
        rospy.sleep(0.3)

        # 3. 打开夹爪释放物体
        rospy.loginfo("[3/4] 打开夹爪释放物体")
        if not self.open_gripper():
            rospy.logerr("✗ 打开夹爪失败，取消放置")
            return False
        rospy.sleep(0.3)

        # 4. 在 Gazebo 里解除约束，让物块留在桌上
        if not self.detach_in_gazebo():
            rospy.logwarn("⚠ Gazebo detach 失败，可能约束已不存在")

        # 5. 抬起末端执行器
        rospy.loginfo("[4/4] 抬起末端执行器")
        if not self.move_to_pose(x, y, z + 0.20):
            rospy.logerr("✗ 抬起失败")
            return False

        rospy.loginfo("✓ 放置完成!")
        rospy.loginfo("=" * 50 + "\n")
        self.delete_cube_in_gazebo()
        return True

    # ---------------------------------------------------------
    # 回到初始位置
    # ---------------------------------------------------------
    def go_to_home(self):
        """回到初始位置"""
        rospy.loginfo("\n>>> 回到初始位置")
        joint_goal = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        self.move_group.set_start_state_to_current_state()
        success = self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        rospy.loginfo("go_to_home success? %s", success)
        rospy.loginfo("✓ 已回到初始位置\n")
        return success

    # ---------------------------------------------------------
    # 主演示函数
    # ---------------------------------------------------------
    def run_demo(self):
        """运行完整的抓取放置演示"""
        try:
            rospy.loginfo("\n" + "#" * 50)
            rospy.loginfo("# 开始 Franka Panda 抓取放置演示（Gazebo link attacher）")
            rospy.loginfo("#" * 50 + "\n")

            # 回到初始位置
            self.go_to_home()
            rospy.sleep(1.0)

            # 执行抓取
            if self.pick_object(self.pick_x, self.pick_y, self.pick_z):
                rospy.sleep(0.5)
                # 执行放置
                if self.place_object(self.place_x, self.place_y, self.place_z):
                    rospy.loginfo("\n" + "#" * 50)
                    rospy.loginfo("# ✓✓✓ 抓取放置演示完成! ✓✓✓")
                    rospy.loginfo("#" * 50 + "\n")
                    self.go_to_home()
                else:
                    rospy.logerr("\n✗✗✗ 放置失败 ✗✗✗\n")
            else:
                rospy.logerr("\n✗✗✗ 抓取失败 ✗✗✗\n")

        except Exception as e:
            rospy.logerr(f"\n✗✗✗ 发生错误: {e} ✗✗✗\n")
            import traceback
            traceback.print_exc()

    def delete_cube_in_gazebo(self):
        """删除 Gazebo 中的 cube 模型"""
        rospy.loginfo(">>> 尝试删除 Gazebo 中的方块 cube ...")
        try:
            rospy.wait_for_service('/gazebo/delete_model', timeout=10.0)
            delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            req = DeleteModelRequest()
            req.model_name = "cube"
            resp = delete_srv(req)
            if resp.success:
                rospy.loginfo("✓ 已删除 Gazebo 中的方块 cube")
            else:
                rospy.logwarn(f"⚠ 删除方块 cube 失败: {resp.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"调用 /gazebo/delete_model 出错: {e}")
        except rospy.ROSException:
            rospy.logerr("等待 /gazebo/delete_model 超时，请确认 Gazebo 正在运行。")

# ---------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------
def main():
    try:
        rospy.loginfo("正在初始化抓取放置演示节点...")
        demo = PickPlaceDemo()
        rospy.sleep(1.0)  # 确保组件准备完毕
        demo.run_demo()
        rospy.loginfo("演示完成，节点保持运行...")
        
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("收到中断信号，退出...")
        return
    except KeyboardInterrupt:
        rospy.loginfo("用户中断，退出...")
        return
    except Exception as e:
        rospy.logerr(f"主函数错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
