#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters
import numpy as np
import os
import sys
import torch
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from detection_msgs.msg import BoundingBoxes
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray # 【新增】用于发送宽度、分数等
from tf.transformations import quaternion_from_matrix
# ==========================================
# 1. 环境配置
# ==========================================
GRASPNET_ROOT = os.path.join(os.path.expanduser('~'), 'grasp_robot_ws', 'graspnet-baseline')
sys.path.append(GRASPNET_ROOT)
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'pointnet2'))
try:
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup
except ImportError as e:
    rospy.logerr(f"GraspNet 模块导入失败: {e}")
    sys.exit(1)

class GraspGeneratorNode:
    def __init__(self):
        rospy.init_node('grasp_generator_node', anonymous=True)
        
        # ---------------------------------------
        # A. 模型加载
        # ---------------------------------------
        checkpoint_path = os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar') 
        if not os.path.exists(checkpoint_path):
            rospy.logerr(f"❌ 找不到权重文件: {checkpoint_path}")
            sys.exit(1)

        rospy.loginfo(f"正在加载 GraspNet 模型 (GPU模式)...")
        self.net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        rospy.loginfo("✅ 模型加载完成！")

        # ---------------------------------------
        # B. ROS 设置
        # ---------------------------------------
        self.sub_yolo = message_filters.Subscriber('/yolov5/detections', BoundingBoxes)
        self.sub_pc = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        
        # 发布 1: 抓取位姿 (位置 + 旋转)
        self.pub_pose = rospy.Publisher('/graspnet/grasp_pose', PoseStamped, queue_size=1)
        # 发布 2: 抓取参数 (宽度, 分数, 深度) -> [width, score, depth]
        self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_yolo, self.sub_pc], queue_size=10, slop=1.0)
        self.ts.registerCallback(self.callback)
        
        self.num_point = 20000 
        
        # 可视化缓存
        self.vis_full_cloud = None    
        self.vis_target_cloud = None  
        self.vis_best_grasp_array = None 
        self.has_new_data = False

        rospy.loginfo(">>> 系统就绪，等待数据...")

    def callback(self, yolo_msg, pc_msg):
        # --- 1. 获取全场景点云 ---
        gen = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=False)
        try:
            pc_array_full = np.array(list(gen)).reshape(pc_msg.height, pc_msg.width, 3)
        except ValueError: return

        full_cloud_flat = pc_array_full.reshape(-1, 3)
        full_cloud_flat = full_cloud_flat[~np.isnan(full_cloud_flat).any(axis=1)]
        vis_bg_cloud = full_cloud_flat[::10] 

        # --- 2. 寻找 YOLO 目标 ---
        target_box = None
        max_score = 0
        for box in yolo_msg.bounding_boxes:
            if box.probability > max_score:
                max_score = box.probability
                target_box = box
        if target_box is None: return

        u_min = max(0, int(target_box.xmin))
        v_min = max(0, int(target_box.ymin))
        u_max = min(pc_msg.width, int(target_box.xmax))
        v_max = min(pc_msg.height, int(target_box.ymax))

        cropped_cloud = pc_array_full[v_min:v_max, u_min:u_max, :]
        flat_cloud = cropped_cloud.reshape(-1, 3)
        flat_cloud = flat_cloud[~np.isnan(flat_cloud).any(axis=1)]

        if len(flat_cloud) < 50: return

        # --- 3. 采样 ---
        if len(flat_cloud) >= self.num_point:
            idxs = np.random.choice(len(flat_cloud), self.num_point, replace=False)
        else:
            idxs = np.random.choice(len(flat_cloud), self.num_point, replace=True)
        sampled_cloud = flat_cloud[idxs]

        # --- 4. 推理 ---
        cloud_tensor = torch.from_numpy(sampled_cloud[np.newaxis].astype(np.float32)).to(self.device)
        end_points = {'point_clouds': cloud_tensor, 'cloud_colors': cloud_tensor}

        try:
            with torch.no_grad():
                end_points = self.net(end_points)
                grasp_preds = pred_decode(end_points)
        except Exception: return

        # --- 5. 解析最佳抓取 ---
        preds_top_grasps = grasp_preds[0].detach().cpu().numpy()
        scores = preds_top_grasps[:, 0]
        
        if np.max(scores) < 0.05: return 

        best_idx = np.argmax(scores)
        best_grasp = preds_top_grasps[best_idx]
        
        # GraspNet Output Format: [Score, Width, Height, Depth, Rot(9), Trans(3), ObjMsg]
        score = best_grasp[0]
        width = best_grasp[1]
        height = best_grasp[2]
        depth = best_grasp[3]
        rotation_matrix = best_grasp[4:13].reshape(3, 3)
        translation = best_grasp[13:16]

        # --- 6. 打印详细信息 (如你所愿) ---
        print("\n" + "="*40)
        rospy.loginfo(f"⚡ 最佳抓取检测成功!")
        print(f"  > 目标: {target_box.Class} (YOLO Conf: {max_score:.2f})")
        print(f"  > 抓取评分 (Score): {score:.4f}")
        print(f"  > 建议夹爪宽度 (Width): {width:.4f} m")
        print(f"  > 抓取深度 (Depth): {depth:.4f} m")
        print(f"  > 抓取位置 (XYZ): [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        print("="*40 + "\n")

        # --- 7. 发布信息 ---
        # A. 发布 Pose
        self.publish_pose(rotation_matrix, translation, pc_msg.header.frame_id)
        
        # B. 发布 Info (Width, Score, Depth)
        info_msg = Float32MultiArray()
        info_msg.data = [width, score, depth]
        self.pub_info.publish(info_msg)

        # --- 8. 更新可视化数据 ---
        self.vis_full_cloud = vis_bg_cloud     
        self.vis_target_cloud = sampled_cloud  
        self.vis_best_grasp_array = best_grasp 
        self.has_new_data = True

    def publish_pose(self, rot, trans, frame_id):
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = trans
        q = quaternion_from_matrix(matrix)
        
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.pose.position.x = trans[0]
        msg.pose.position.y = trans[1]
        msg.pose.position.z = trans[2]
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pub_pose.publish(msg)

    def run_visualization(self):
        """主线程：Open3D 渲染"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GraspNet Monitor", width=1024, height=768)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        pcd_bg = o3d.geometry.PointCloud()     
        pcd_target = o3d.geometry.PointCloud() 
        current_gripper_geoms = [] 
        
        is_first_frame = True

        while not rospy.is_shutdown():
            if self.has_new_data and self.vis_full_cloud is not None:
                pcd_bg.points = o3d.utility.Vector3dVector(self.vis_full_cloud)
                pcd_bg.paint_uniform_color([0.5, 0.5, 0.5]) 
                pcd_target.points = o3d.utility.Vector3dVector(self.vis_target_cloud)
                pcd_target.paint_uniform_color([1, 0, 0]) 

                for geom in current_gripper_geoms:
                    vis.remove_geometry(geom, reset_bounding_box=False)
                current_gripper_geoms = []

                if self.vis_best_grasp_array is not None:
                    gg = GraspGroup(self.vis_best_grasp_array.reshape(1, 17))
                    gripper_geometry_list = gg.to_open3d_geometry_list()
                    for geom in gripper_geometry_list:
                        geom.paint_uniform_color([0, 1, 0]) 
                        vis.add_geometry(geom, reset_bounding_box=False)
                        current_gripper_geoms.append(geom)

                if is_first_frame:
                    vis.add_geometry(pcd_bg)
                    vis.add_geometry(pcd_target)
                    vis.reset_view_point(True)
                    is_first_frame = False
                else:
                    vis.update_geometry(pcd_bg)
                    vis.update_geometry(pcd_target)
                
                self.has_new_data = False
                vis.poll_events()
                vis.update_renderer()
            else:
                vis.poll_events()
                vis.update_renderer()

if __name__ == '__main__':
    try:
        node = GraspGeneratorNode()
        node.run_visualization()
    except rospy.ROSInterruptException:
        pass