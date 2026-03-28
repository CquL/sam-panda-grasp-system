#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
import torch
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_matrix

# ==========================================
# 1. 路径配置与导入 (关键修改区域!)
# ==========================================
# 获取 graspnet-baseline 的绝对路径
GRASPNET_ROOT = os.path.join(os.path.expanduser('~'), 'grasp_robot_ws', 'graspnet-baseline')

# 1. 将根目录加入路径
sys.path.append(GRASPNET_ROOT) 

# 2. 【新增】将 models 目录也加入路径！解决 'No module named backbone'
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))

# 3. 【新增】将 utils 目录加入路径 (防止 utils 内部互相引用报错)
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

# 4. 【新增】将 pointnet2 目录加入路径 (以防万一)
sys.path.append(os.path.join(GRASPNET_ROOT, 'pointnet2'))

try:
    # --- 移花接木：直接使用 demo.py 用到的模块 ---
    # 注意：现在 models 在路径里了，可以直接 import graspnet，也可以 models.graspnet
    # 为了保险，我们尝试两种方式
    try:
        from graspnet import GraspNet, pred_decode # 如果 models 在 path 里，直接 import
    except ImportError:
        from models.graspnet import GraspNet, pred_decode

    from collision_detector import ModelFreeCollisionDetector
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
except ImportError as e:
    rospy.logerr(f"导入失败! 错误信息: {e}")
    # 打印一下当前的路径帮助调试
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
        
        rospy.loginfo("✓ 模型加载成功！")

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
        rospy.loginfo("收到点云，开始处理...")
        
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
        
        # 5. 解析结果 
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

if __name__ == '__main__':
    try:
        GraspNetROSDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass