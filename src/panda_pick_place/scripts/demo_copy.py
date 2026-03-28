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
        self.arm.move_group.set_planner_id("RRTConnectkConfigDefault") # MoveIt默认，速度极快，但路径随机且扭曲
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
        
        best_plan = None
        best_pre_grasp_pose = None
        best_target_pose = None
        chosen_width = 0.04
        best_robot_z = None # 保存我们算出来的绝对朝向

        rospy.loginfo("🔍 开始强制水平修正与避障评估...")
        
        for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
            ps = PoseStamped()
            ps.header = self.grasp_pose_array_received.header
            ps.pose = raw_pose
            
            target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
            if target_pose_stamped is None: continue

            # =========================================================================
            # [步骤 1] 终极约束：保留 GraspNet 目标方向，强制水平对齐 TCP！
            # =========================================================================
            q_orig = [
                target_pose_stamped.pose.orientation.x,
                target_pose_stamped.pose.orientation.y,
                target_pose_stamped.pose.orientation.z,
                target_pose_stamped.pose.orientation.w
            ]
            mat_orig = tft.quaternion_matrix(q_orig) 
            raw_grasp_x = mat_orig[:3, 0] # GraspNet 默认 X 轴朝向物体
            
            # 将朝向投影到 XY 平面，抹除所有的上下倾斜
            flat_approach = np.array([raw_grasp_x[0], raw_grasp_x[1], 0.0])
            norm = np.linalg.norm(flat_approach)
            
            if norm < 0.001:
                rospy.logwarn(f"候选姿态 {i+1}: 抓取方向几乎垂直，跳过！")
                continue
                
            # 重建机械臂坐标系：Z 轴朝目标，X 轴朝下(手背朝上)，Y 轴平行地面(两指平行)
            robot_z = flat_approach / norm
            robot_x = np.array([0.0, 0.0, -1.0]) 
            robot_y = np.cross(robot_z, robot_x) 
            robot_y = robot_y / np.linalg.norm(robot_y) 
            robot_x = np.cross(robot_y, robot_z)
            
            new_rot_mat = np.eye(4)
            new_rot_mat[:3, 0] = robot_x
            new_rot_mat[:3, 1] = robot_y
            new_rot_mat[:3, 2] = robot_z
            
            q_new = tft.quaternion_from_matrix(new_rot_mat)
            target_pose_stamped.pose.orientation.x = q_new[0]
            target_pose_stamped.pose.orientation.y = q_new[1]
            target_pose_stamped.pose.orientation.z = q_new[2]
            target_pose_stamped.pose.orientation.w = q_new[3]

            # 🛡️ 强制底盘防撞保护：目标和预抓取点双双抬高，确保路径绝对水平且不蹭底
            if target_pose_stamped.pose.position.z < 0.08:
                target_pose_stamped.pose.position.z = 0.08

            # =========================================================================
            # [步骤 2] 计算预抓取点 (彻底剔除所有魔改偏移)
            # =========================================================================
            back_distance = 0.12 
            pre_grasp_pose = copy.deepcopy(target_pose_stamped.pose)
            
            # 严格沿着计算出的 TCP Z 轴后退，绝不触碰 Z 坐标！
            pre_grasp_pose.position.x -= robot_z[0] * back_distance
            pre_grasp_pose.position.y -= robot_z[1] * back_distance
            pre_grasp_pose.position.z -= robot_z[2] * back_distance
            # pre_grasp_pose.position.z += 0.03 # 轻微抬高，避免碰撞
            # pre_grasp_pose.position.x  # 轻微侧移，增加避障空间
            # [步骤 3] MoveIt 碰撞检测与逆向运动学(IK)验证
            # =========================================================================
            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.set_pose_target(pre_grasp_pose)
            
            plan_result = self.arm.move_group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                success = len(plan_result.joint_trajectory.points) > 0
                plan = plan_result
            
            if success:
                rospy.loginfo(f"🎉 成功锁定纯平行抓取姿态！")
                best_plan = plan
                best_pre_grasp_pose = pre_grasp_pose
                best_target_pose = target_pose_stamped.pose
                best_robot_z = robot_z # 存下完美的 Z 轴方向用于拔插
                
                if len(self.grasp_infos) > i * 3:
                    chosen_width = self.grasp_infos[i * 3]
                else:
                    chosen_width = 0.04
                break 
            else:
                rospy.logwarn(f"❌ 姿态 {i+1} 会撞货架，被淘汰！")

        if best_plan is None:
            rospy.logerr("所有候选姿态都不可达！")
            self.grasp_pose_array_received = None 
            return False

        # # 发布 RViz 可视化
        # target_msg = PoseStamped()
        # target_msg.header.frame_id = planning_frame
        # target_msg.header.stamp = rospy.Time.now()
        # target_msg.pose = best_target_pose
        # self.best_target_pub.publish(target_msg)

        # pre_msg = PoseStamped()           
        # pre_msg.header.frame_id = planning_frame
        # pre_msg.header.stamp = rospy.Time.now()
        # pre_msg.pose = best_pre_grasp_pose
        # self.best_pre_pub.publish(pre_msg)

        # =========================================================================
        # [步骤 4] 验证通过，正式驱动机械臂执行！
        # =========================================================================
        self.gripper.open(width=0.08)
        
        rospy.loginfo("[4/6] 移动到柜外预抓取点 (保持姿态平行)...")
        if not self.arm.move_group.execute(best_plan, wait=True):
            rospy.logerr("移动失败！")
            return False
            
        rospy.sleep(0.5)
        
        # =========================================================================
        # [步骤 5] 绝对平行直线插入！
        # =========================================================================
        rospy.loginfo(f"[5/6] 像抽屉一样水平滑入柜子...")
        
        # 💥 核心修改：在原本进去的距离上，强行多往里插 5 厘米 (彻底包住可乐)
        extra_depth = 0.02  # 你可以根据可乐的粗细微调这个值，比如 0.04 到 0.06
        total_insert_distance = back_distance + extra_depth
        
        in_dx = robot_z[0] * total_insert_distance
        in_dy = robot_z[1] * total_insert_distance
        in_dz = robot_z[2] * total_insert_distance
        
        # 显式传入 avoid_collisions=False，闭着眼睛大胆往里插！
        if not self.arm.cartesian_move_relative(dx=in_dx, dy=in_dy, dz=in_dz, avoid_collisions=False):
            rospy.logwarn("🛑 拦截：直线插入控制失败！")
            return False
            
        rospy.sleep(0.5)
        
        # === 6. 闭合夹爪 ===
        target_width = max(0.0, chosen_width - 0.013)
        rospy.loginfo(f"[6/6] 闭合夹爪 (目标宽度: {target_width:.3f}m)")
        self.gripper.close(width=target_width, force=35.0) 
        rospy.sleep(1.0)

        # =========================================================================
        # [步骤 7] 拔出与抬起
        # =========================================================================
        rospy.loginfo(">>> 保持水平原路撤出...")
        
        # 💥 怎么进去的就怎么原路退出（对刚才的 in_dx, in_dy, in_dz 取反）
        if not self.arm.cartesian_move_relative(dx=-in_dx, dy=-in_dy, dz=-in_dz, avoid_collisions=False):
            rospy.logwarn("🛑 撤出时发生异常，尝试强行回 Home！")
            return False
            
        rospy.sleep(0.5)
        
        rospy.loginfo(">>> 完全离开柜子，安全抬高...")
        # 离开柜子后，只改 dz 向上抬起 12 厘米
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