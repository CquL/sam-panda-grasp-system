#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import copy
import rospy
import numpy as np
import cv2
import moveit_commander
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Vector3Stamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
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
        self.bridge = CvBridge()

        rospy.sleep(1.0)
        ground_pose = PoseStamped()
        ground_pose.header.frame_id = self.robot.get_planning_frame()
        ground_pose.pose.position.x = 0.0
        ground_pose.pose.position.y = 0.0
        ground_pose.pose.position.z = -0.02
        ground_pose.pose.orientation.w = 1.0
        self.scene.add_box("absolute_ground", ground_pose, size=(2.0, 2.0, 0.04))
        rospy.loginfo("🌍 已将隐形刚体地面 [absolute_ground] 强制加入 MoveIt 碰撞空间！")

        self.is_running_task = False
        self.stop_requested = False
        self.arm.move_group.set_planning_time(5.0)
        self.arm.move_group.set_goal_position_tolerance(0.01)
        self.arm.move_group.set_goal_orientation_tolerance(0.05)
        self.arm.move_group.set_goal_joint_tolerance(0.05)

        self.basket_x = 0.45
        self.basket_y = -0.1
        self.basket_z = 0.23
        self.drop_counter = 0

        self.grasp_pose_array_received = None
        self.grasp_infos = []

        self.debug_rgb = None
        self.debug_K = None
        self.show_grasp_debug = rospy.get_param("~show_grasp_debug", True)
        # self.camera_frame = rospy.get_param("~camera_frame", "depth_camera_link")
        self.gripper_draw_half_width = rospy.get_param("~gripper_draw_half_width", 0.04)
        self.camera_frame = rospy.get_param("~camera_frame", "wrist_camera_optical_link")
        self.grasp_sub = rospy.Subscriber('/graspnet/grasp_pose_array', PoseArray, self.grasp_pose_callback)
        self.info_sub = rospy.Subscriber('/graspnet/grasp_info', Float32MultiArray, self.grasp_info_callback)
        self.cmd_sub = rospy.Subscriber('/demo/command', String, self.ui_command_callback)
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.debug_rgb_cb, queue_size=1)
        self.info_cam_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.debug_info_cb, queue_size=1)

        self.wrist_rgb = None
        self.wrist_depth = None
        rospy.Subscriber('/wrist_camera/color/image_raw', Image, self.wrist_rgb_cb)
        rospy.Subscriber('/wrist_camera/depth/image_raw', Image, self.wrist_depth_cb)

        self.status_pub = rospy.Publisher('/demo/task_status', String, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 增加触发 VLM 的发布器
        self.pub_wrist_trigger = rospy.Publisher('/wrist_vlm/trigger', String, queue_size=1)
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
            self.status_pub.publish("FAILED_BY_USER")
        elif cmd == 'quit':
            rospy.logwarn("收到退出指令，正在关闭后台服务...")
            rospy.signal_shutdown("UI 请求退出")

    def grasp_pose_callback(self, msg):
        if not self.is_running_task:
            self.grasp_pose_array_received = msg

    def grasp_info_callback(self, msg):
        if not self.is_running_task:
            self.grasp_infos = msg.data

    def debug_rgb_cb(self, msg):
        try:
            self.debug_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def debug_info_cb(self, msg):
        try:
            self.debug_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        except Exception:
            self.debug_K = None

    def wrist_rgb_cb(self, msg):
        try:
            self.wrist_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def wrist_depth_cb(self, msg):
        try:
            self.wrist_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            pass

    def transform_pose(self, input_pose, target_frame="world"):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            return tf2_geometry_msgs.do_transform_pose(input_pose, transform)
        except Exception:
            return None

    # ===================================================================================
    # 🧠 方案 B: VLM 语义微调 (纯相机局部坐标系版)
    # ===================================================================================
    def calculate_world_correction(self):
        """
        全自动天地融合版：VLM 给范围 + 深度图抠边缘
        返回：【世界坐标系】下，机械臂为了对准目标需要平移的 dx, dy, dz
        """
        if self.wrist_depth is None or self.wrist_rgb is None:
            return 0.0, 0.0, 0.0

        # 1. 触发 VLM 获取边界框
        self.pub_wrist_trigger.publish("trigger")
        rospy.loginfo("等待 VLM 节点返回边框数据...")

        try:
            msg = rospy.wait_for_message('/wrist_vlm/bbox', Float32MultiArray, timeout=8.0)
            if len(msg.data) != 4:
                return 0.0, 0.0, 0.0
            xmin, ymin, xmax, ymax = map(int, msg.data)
        except rospy.ROSException:
            rospy.logwarn("VLM 节点超时未回复，放弃微调！")
            return 0.0, 0.0, 0.0

        h, w = self.wrist_depth.shape
        cx, cy = w // 2, h // 2  
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        # 2. X轴(左右)：通过深度图抠出物理边缘中点
        roi_depth = self.wrist_depth[ymin:ymax, xmin:xmax]
        valid_mask = (roi_depth > 0.05) & (roi_depth < 0.4)
        if not np.any(valid_mask): return 0.0, 0.0, 0.0
        
        min_depth = np.min(roi_depth[valid_mask])
        target_mask = (roi_depth >= min_depth) & (roi_depth < min_depth + 0.05)
        
        _, local_x_indices = np.where(target_mask)
        if len(local_x_indices) == 0: return 0.0, 0.0, 0.0
        
        global_x_indices = local_x_indices + xmin
        target_x = int((np.min(global_x_indices) + np.max(global_x_indices)) / 2.0)

        # 3. Y轴(上下)：直接信任 VLM 给出的绝对中点
        target_y = int((ymin + ymax) / 2.0)

        # --- 渲染调试画面 ---
        img_cv = self.wrist_rgb.copy()
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2) 
        cv2.circle(img_cv, (cx, cy), 5, (255, 0, 0), -1)       
        cv2.circle(img_cv, (target_x, target_y), 5, (0, 255, 0), -1) 
        cv2.line(img_cv, (cx, cy), (target_x, target_y), (0, 255, 255), 2) 
        cv2.imshow("Auto VLM Eye-in-Hand", img_cv)
        cv2.waitKey(1)

        # =========================================================================
        # 🚀 核心工业级刚体逆解：绝对杜绝向量方向错误和视差补偿错误！
        # =========================================================================
        try:
            # 步骤A：构建物体在【相机坐标系】下的绝对 3D 点 (Pose)
            # 因为是点(Pose)而不是向量，所以在转换时物理高度差(0.04m)会被TF树完美计算在内！
            p_cam = PoseStamped()
            p_cam.header.frame_id = self.camera_frame
            p_cam.header.stamp = rospy.Time(0)
            fx, fy = 500.0, 500.0
            
            p_cam.pose.position.x = (target_x - cx) * min_depth / fx
            p_cam.pose.position.y = (target_y - cy) * min_depth / fy
            p_cam.pose.position.z = min_depth
            p_cam.pose.orientation.w = 1.0

            # 抓取夹爪的当前坐标系名称
            ee_link = self.arm.move_group.get_end_effector_link()
            if not ee_link: ee_link = "panda_hand"

            # 步骤B：问 TF 树，这个 3D 物体点在【夹爪坐标系】里在哪儿？
            trans_cam_to_hand = self.tf_buffer.lookup_transform(ee_link, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            p_hand = tf2_geometry_msgs.do_transform_pose(p_cam, trans_cam_to_hand)

            # 步骤C：得出夹爪需要移动的局部距离
            # 如果物体完美对准，它在夹爪坐标系下的 x 和 y 必定是 0。
            # 所以 p_hand 的 x 和 y，正是夹爪自身需要去消除的位移！
            vec_hand = Vector3Stamped()
            vec_hand.header.frame_id = ee_link
            vec_hand.header.stamp = rospy.Time(0)
            vec_hand.vector.x = p_hand.pose.position.x
            vec_hand.vector.y = p_hand.pose.position.y
            vec_hand.vector.z = 0.0 # 绝对不改变深度，深度交给 GraspNet 的初始规划

            # 步骤D：把夹爪的局部平移，转化为【世界坐标系】的位移
            trans_hand_to_world = self.tf_buffer.lookup_transform("world", ee_link, rospy.Time(0), rospy.Duration(1.0))
            vec_world = tf2_geometry_msgs.do_transform_vector3(vec_hand, trans_hand_to_world)

            return vec_world.vector.x, vec_world.vector.y, vec_world.vector.z

        except Exception as e:
            rospy.logerr(f"TF 逆解空间映射失败: {e}")
            return 0.0, 0.0, 0.0
    # ===================================================================================

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
        best_plan, best_pre_grasp_pose, best_target_pose = None, None, None
        chosen_width = 0.04
        best_robot_z, best_robot_x, best_robot_y = None, None, None

        rospy.loginfo("🔍 开始强制水平修正与避障评估...")

        for i, raw_pose in enumerate(self.grasp_pose_array_received.poses):
            ps = PoseStamped()
            ps.header = self.grasp_pose_array_received.header
            ps.pose = raw_pose

            target_pose_stamped = self.transform_pose(ps, target_frame=planning_frame)
            if target_pose_stamped is None:
                continue

            q_orig = [
                target_pose_stamped.pose.orientation.x, target_pose_stamped.pose.orientation.y,
                target_pose_stamped.pose.orientation.z, target_pose_stamped.pose.orientation.w
            ]
            mat_orig = tft.quaternion_matrix(q_orig)
            raw_grasp_x = mat_orig[:3, 0]

            flat_approach = np.array([raw_grasp_x[0], raw_grasp_x[1], 0.0])
            norm = np.linalg.norm(flat_approach)

            if norm < 0.001: continue

            robot_z = flat_approach / norm
            robot_x = np.array([0.0, 0.0, -1.0])
            robot_y = np.cross(robot_z, robot_x)
            robot_y = robot_y / np.linalg.norm(robot_y)
            robot_x = np.cross(robot_y, robot_z)

            new_rot_mat = np.eye(4)
            new_rot_mat[:3, 0], new_rot_mat[:3, 1], new_rot_mat[:3, 2] = robot_x, robot_y, robot_z
            q_new = tft.quaternion_from_matrix(new_rot_mat)
            target_pose_stamped.pose.orientation.x = q_new[0]
            target_pose_stamped.pose.orientation.y = q_new[1]
            target_pose_stamped.pose.orientation.z = q_new[2]
            target_pose_stamped.pose.orientation.w = q_new[3]

            if target_pose_stamped.pose.position.z < 0.08:
                target_pose_stamped.pose.position.z = 0.08

            back_distance = 0.12
            pre_grasp_pose = copy.deepcopy(target_pose_stamped.pose)
            pre_grasp_pose.position.x -= robot_z[0] * back_distance
            pre_grasp_pose.position.y -= robot_z[1] * back_distance
            pre_grasp_pose.position.z -= robot_z[2] * back_distance

            self.arm.move_group.set_start_state_to_current_state()
            self.arm.move_group.set_pose_target(pre_grasp_pose)

            plan_result = self.arm.move_group.plan()
            if isinstance(plan_result, tuple):
                success, plan = plan_result[0], plan_result[1]
            else:
                success, plan = len(plan_result.joint_trajectory.points) > 0, plan_result

            self.arm.move_group.clear_pose_targets()

            if success:
                rospy.loginfo("🎉 成功锁定纯平行抓取姿态！")
                best_plan = plan
                best_pre_grasp_pose = pre_grasp_pose
                best_target_pose = target_pose_stamped.pose
                best_robot_z, best_robot_x, best_robot_y = robot_z, robot_x, robot_y
                chosen_width = self.grasp_infos[i * 3] if len(self.grasp_infos) > i * 3 else 0.04
                break
            else:
                rospy.logwarn(f"❌ 姿态 {i+1} 会撞货架，被淘汰！")

        if best_plan is None:
            rospy.logerr("所有候选姿态都不可达！")
            self.grasp_pose_array_received = None
            return False

        self.gripper.open(width=0.08)

        rospy.loginfo("[4/6] 移动到柜外预抓取点 (保持姿态平行)...")
        if not self.arm.move_group.execute(best_plan, wait=True):
            rospy.logerr("移动失败！")
            return False

        rospy.sleep(1.0) 
        
        # =========================================================================
        # 🌟🌟🌟 [步骤 4.5] VLM 语义微调 (TF 树空间刚体逆解版) 🌟🌟🌟
        # =========================================================================
        rospy.loginfo("[4.5/6] 📸 启动腕部相机，执行 3D 刚体逆解对齐...")
        
        # 直接拿到底盘(世界坐标系)应该平移的距离，完全不需要手动算矩阵了
        fine_dx, fine_dy, fine_dz = self.calculate_world_correction()

        # 安全限幅：单次最多纠正 10 厘米
        fine_dx = np.clip(fine_dx, -0.10, 0.10)
        fine_dy = np.clip(fine_dy, -0.10, 0.10)
        fine_dz = np.clip(fine_dz, -0.10, 0.10)
        
        if abs(fine_dx) > 0.002 or abs(fine_dy) > 0.002 or abs(fine_dz) > 0.002:
            rospy.loginfo(f"🔧 TF 刚体平移修正: dx={fine_dx:.4f}, dy={fine_dy:.4f}, dz={fine_dz:.4f}")
            self.arm.cartesian_move_relative(dx=fine_dx, dy=fine_dy, dz=fine_dz, avoid_collisions=False)
            rospy.sleep(0.5)
        else:
            rospy.loginfo("🎯 目标居中完美，无需微调！")
        # =========================================================================
        
        rospy.loginfo(f"[5/6] 像抽屉一样水平滑入柜子...")
        extra_depth = 0.05
        total_insert_distance = back_distance + extra_depth

        in_dx = best_robot_z[0] * total_insert_distance
        in_dy = best_robot_z[1] * total_insert_distance
        in_dz = best_robot_z[2] * total_insert_distance

        if not self.arm.cartesian_move_relative(dx=in_dx, dy=in_dy, dz=in_dz, avoid_collisions=False):
            rospy.logwarn("🛑 拦截：直线插入控制失败！")
            return False

        rospy.sleep(0.5)

        target_width = max(0.0, chosen_width - 0.01)
        rospy.loginfo(f"[6/6] 闭合夹爪 (目标宽度: {target_width:.3f}m)")
        self.gripper.close(width=0.025, force=50.0)
        rospy.sleep(1.0)

        rospy.loginfo(">>> 保持水平原路撤出...")
        self.arm.cartesian_move_relative(dx=0.0, dy=0.0, dz=0.03)

        if not self.arm.cartesian_move_relative(dx=-in_dx, dy=-in_dy, dz=-in_dz, avoid_collisions=False):
            rospy.logwarn("🛑 撤出时发生异常，尝试强行回 Home！")
            return False

        rospy.sleep(0.5)

        rospy.loginfo(">>> 完全离开柜子，安全抬高...")
        # self.arm.cartesian_move_relative(dx=0.0, dy=0.0, dz=0.12)

        self.grasp_pose_array_received = None
        self.grasp_infos = []

        return True

    def place_object(self):
        if self.stop_requested: return False
        target = copy.deepcopy(self.arm.move_group.get_current_pose().pose)
        
        offset_y = self.drop_counter * 0.10
        target.position.x = self.basket_x
        target.position.y = self.basket_y + offset_y
        target.position.z = self.basket_z + 0.15

        rospy.loginfo(f"正在规划移动到放置点: x={target.position.x:.2f}, y={target.position.y:.2f}, z={target.position.z:.2f}")
        self.arm.move_group.set_pose_target(target)
        if not self.arm.move_group.go(wait=True): return False

        self.arm.cartesian_move_relative(dz=-0.15)
        self.gripper.open()
        rospy.sleep(0.5)
        self.drop_counter += 1
        self.arm.cartesian_move_relative(dz=0.20)
        return True

    def run_service(self):
        self.arm.go_to_home()
        while not rospy.is_shutdown():
            self.stop_requested = False
            if self.grasp_pose_array_received is None:
                if self.show_grasp_debug and self.debug_rgb is not None: cv2.waitKey(1)
                rospy.sleep(0.1)
                continue

            rospy.loginfo(">>> 开始执行物理抓取序列...")
            self.is_running_task = True
            task_success = False

            try:
                if self.pick_object() and self.place_object():
                    rospy.loginfo("✅ 单次抓取-放置物理动作执行完毕！")
                    task_success = True
                else: rospy.logwarn("⚠️ 任务失败或中止。")
            except Exception as e:
                rospy.logerr(f"异常: {e}")

            self.arm.go_to_home()
            self.grasp_pose_array_received = None
            self.grasp_infos = []
            self.is_running_task = False
            rospy.loginfo(">>> 已彻底复位，准备向调度器请求/接收新任务...")
            self.status_pub.publish("DONE" if task_success else "FAILED")

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
        except RuntimeError: rospy.sleep(2.0)
        except Exception as e:
            rospy.logerr(f"💥 main循环发生致命崩溃！错误信息: {e}")
            break
        finally:
            if rospy.is_shutdown():
                try: cv2.destroyAllWindows()
                except Exception: pass

if __name__ == "__main__":
    main()