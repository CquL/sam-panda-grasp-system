#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
import torch
import open3d as o3d  # 【新增】Open3D 可视化库
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix

# ==========================================
# 1. 路径配置与导入
# ==========================================
# 获取 graspnet-baseline 的绝对路径
GRASPNET_ROOT = os.path.join(os.path.expanduser('~'), 'grasp_robot_ws', 'graspnet-baseline')

# 将相关目录加入路径
sys.path.append(GRASPNET_ROOT) 
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'pointnet2'))

try:
    # 尝试导入 GraspNet 相关模块
    try:
        from graspnet import GraspNet, pred_decode 
    except ImportError:
        from models.graspnet import GraspNet, pred_decode

    from graspnetAPI import GraspGroup  # 【新增】用于处理抓取组数据
    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
except ImportError as e:
    rospy.logerr(f"导入失败! 错误信息: {e}")
    rospy.logerr(f"当前 sys.path: {sys.path}")
    sys.exit(1)

class GraspNetROSDetector:
    def __init__(self):
        rospy.init_node('graspnet_detector_node', anonymous=True)
        
        # ==========================================
        # 2. 模型初始化
        # ==========================================
        checkpoint_path = os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar')
        
        if not os.path.exists(checkpoint_path):
            rospy.logerr(f"找不到权重文件: {checkpoint_path}")
            sys.exit(1)

        rospy.loginfo(f"正在加载模型: {checkpoint_path} ...")
        
        # 初始化网络结构
        self.net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
        # 加载权重到 GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.net.to(device)
        
        # 加载参数
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        
        rospy.loginfo("✓ 模型加载成功！等待点云数据...")

        # ==========================================
        # 3. ROS 通信设置
        # ==========================================
        self.pub_pose = rospy.Publisher('/graspnet/grasp_pose', PoseStamped, queue_size=1)
        
        # 订阅点云
        self.sub_pc = rospy.Subscriber('/camera/depth/points', PointCloud2, self.callback, queue_size=1)
        
        self.num_point = 20000 
        self.collision_thresh = 0.01
        self.voxel_size = 0.005

    def callback(self, ros_data):
        rospy.loginfo("收到点云，开始推理...")
        
        # 1. ROS PointCloud2 -> Numpy
        gen = read_points(ros_data, field_names=("x", "y", "z"), skip_nans=True)
        raw_cloud = np.array(list(gen)) 
        
        if raw_cloud.shape[0] < 1000:
            rospy.logwarn("点云点数太少，跳过")
            return

        # 2. 数据采样 
        cloud_masked = raw_cloud
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=True)
        
        sampled_cloud = cloud_masked[idxs] 

        # 3. 准备模型输入 
        end_points = dict()
        cloud_tensor = torch.from_numpy(sampled_cloud[np.newaxis].astype(np.float32)).to(self.device)
        
        end_points['point_clouds'] = cloud_tensor
        end_points['cloud_colors'] = cloud_tensor 
        
        # 4. 执行推理 
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        
        # =========================================================
        # 【新增功能】 Open3D 可视化模块 (在这里弹窗)
        # =========================================================
        try:
            # A. 转换抓取结果
            gg_array = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(gg_array)
            
            # 过滤一下，只显示分数最高的 50 个，防止太乱
            gg.nms()              # 去重
            gg.sort_by_score()    # 排序
            gg_vis = gg[:50]      # 取前50
            grippers = gg_vis.to_open3d_geometry_list() # 转为几何体

            # B. 转换点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sampled_cloud.astype(np.float32))
            pcd.paint_uniform_color([0.6, 0.6, 0.6]) # 给点云涂成灰色

            # C. 弹出窗口
            rospy.loginfo(">>> 正在显示 3D 窗口。请【关闭窗口】以发送 Pose 并继续运行！ <<<")
            o3d.visualization.draw_geometries([pcd, *grippers], window_name="GraspNet ROS Debug")
            
        except Exception as e:
            rospy.logwarn(f"可视化出错 (不影响运行): {e}")
        # =========================================================

        # 5. 解析结果并发布
        preds_top_grasps = grasp_preds[0].detach().cpu().numpy() 
        
        # 取最高分的抓取
        scores = preds_top_grasps[:, 0]
        best_idx = np.argmax(scores)
        best_grasp = preds_top_grasps[best_idx]
        
        score = best_grasp[0]
        rotation = best_grasp[4:13].reshape(3, 3) 
        translation = best_grasp[13:16] 

        rospy.loginfo(f"检测到最佳抓取: Score={score:.3f}, Pos={translation}")

        # 6. 发布 PoseStamped 给 MoveIt
        self.publish_grasp_pose(rotation, translation, ros_data.header.frame_id)

    def publish_grasp_pose(self, rot_matrix, trans_vector, frame_id):
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = trans_vector
        
        q = quaternion_from_matrix(matrix)
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = frame_id
        
        pose_msg.pose.position.x = trans_vector[0]
        pose_msg.pose.position.y = trans_vector[1]
        pose_msg.pose.position.z = trans_vector[2]
        
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pub_pose.publish(pose_msg)
        rospy.loginfo(f"已发布 Pose 到 /graspnet/grasp_pose (frame: {frame_id})")

if __name__ == '__main__':
    try:
        GraspNetROSDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass