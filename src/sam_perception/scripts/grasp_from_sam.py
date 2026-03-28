#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
import torch
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float32MultiArray
from tf.transformations import quaternion_from_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRASPNET_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', '..', 'third_party', 'graspnet-baseline')
)
GRASPNET_ROOT = os.environ.get('GRASPNET_ROOT', DEFAULT_GRASPNET_ROOT)
sys.path.append(GRASPNET_ROOT)
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'pointnet2'))

try:
    from graspnet import GraspNet, pred_decode
except ImportError as e:
    rospy.logerr(f"GraspNet 模块导入失败: {e}")
    sys.exit(1)

class SamGraspNode:
    def __init__(self):
        rospy.init_node('sam_grasp_node')

        if not os.path.isdir(GRASPNET_ROOT):
            rospy.logerr(f"找不到 GraspNet 目录: {GRASPNET_ROOT}")
            sys.exit(1)
        
        # 1. 模型加载 (使用你之前的 rs.tar 权重)
        checkpoint_path = os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar') 
        if not os.path.exists(checkpoint_path):
            rospy.logerr(f"找不到 GraspNet 权重文件: {checkpoint_path}")
            sys.exit(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.net.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        rospy.loginfo("✅ GraspNet 已准备好处理 SAM 点云")

        # 2. 核心订阅：听 SAM 抠出来的物体点云
        self.cloud_sub = rospy.Subscriber("/sam_perception/object_cloud", PointCloud2, self.callback)
        
        # 3. 发布：抓取位姿 和 抓取参数 (为了无缝对接 demo.py)
        # 【修改】使用 PoseArray 发布多个姿态
        # self.pub_pose_array = rospy.Publisher('/graspnet/grasp_pose_array', PoseArray, queue_size=1)
        # self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)

        self.pub_pose_array = rospy.Publisher('/graspnet/grasp_pose_array_raw', PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher('/graspnet/grasp_info_raw', Float32MultiArray, queue_size=1)

    def callback(self, pc_msg):
        # --- 1. 转换点云 ---
        gen = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        sampled_cloud = np.array(list(gen))
        
        if len(sampled_cloud) < 50:
            rospy.logwarn("点云点数太少，无法计算抓取")
            return

        # --- 2. 采样至模型要求的点数 (防显存溢出，保持 8192) ---
        num_point = 8192
        if len(sampled_cloud) >= num_point:
            idxs = np.random.choice(len(sampled_cloud), num_point, replace=False)
        else:
            idxs = np.random.choice(len(sampled_cloud), num_point, replace=True)
        sampled_cloud = sampled_cloud[idxs]

        # --- 3. 模型推理 ---
        cloud_tensor = torch.from_numpy(sampled_cloud[np.newaxis].astype(np.float32)).to(self.device)
        end_points = {'point_clouds': cloud_tensor, 'cloud_colors': cloud_tensor}
        torch.cuda.empty_cache()
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)

        # =======================================================
        # --- 4. 核心解析与三大物理修复 ---
        # =======================================================
        # grasp_preds[0] 是形状为 [N, 17] 的矩阵
        # 内部排序 0:分数, 1:宽度, 2:高度, 3:深度, 4~12:旋转矩阵, 13~15:平移坐标
        grasps_array = grasp_preds[0].detach().cpu().numpy()
        
        # 将矩阵拆解成对应的属性向量，方便进行向量化数学计算
        scores = grasps_array[:, 0]
        widths = grasps_array[:, 1]
        depths = grasps_array[:, 3]
        rotations = grasps_array[:, 4:13].reshape(-1, 3, 3)
        translations = grasps_array[:, 13:16]


        if len(scores) == 0:
            rospy.logwarn("没有找到符合夹爪宽度的抓取姿态！")
            return
        
        
        if np.max(scores) < 0.05: 
            return 

        # # 【修复一】：基础宽度过滤（剔除太宽抓不住的，防推倒）
        # mask = (widths > 0.02) & (widths <= 0.075)
        
        # # 同步过滤所有属性
        # scores = scores[mask]
        # widths = widths[mask]
        # depths = depths[mask]
        # rotations = rotations[mask]
        # translations = translations[mask]
        

        # 【修复二】：人为制造“穿透深度”（往物体中心推入 6 厘米）
        # 沿着机械臂要夹取的 Z 轴方向（forward_vector）深入
        forward_vector = rotations[:, :, 2] 
        approach_offset = rospy.get_param("~approach_offset", 0.02)
        translations = translations + forward_vector * approach_offset

        # 【修复三】：“向心力”惩罚机制（强迫抓点对准物体正中央）
        object_center = np.mean(sampled_cloud, axis=0) # 算出物体几何中心
        distances = np.linalg.norm(translations - object_center, axis=1)
        
        # 归一化距离并扣分
        max_dist = np.max(distances) + 1e-5
        normalized_distances = distances / max_dist
        penalty_weight = 1.5 
        combined_scores = scores - (penalty_weight * normalized_distances)
        
        # --- 5. 按照引入向心力的“新得分”进行排序，取 Top-60 ---
        top_k = min(60, len(combined_scores)) 
        top_indices = np.argsort(-combined_scores)[:top_k]

        # =======================================================
        # --- 6. 组装 ROS 消息发送给调度器 ---
        # =======================================================
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = pc_msg.header.frame_id
        
        info_data = []

        for idx in top_indices:
            score = combined_scores[idx]  # 使用新分数
            width = widths[idx]
            depth = depths[idx]
            rot = rotations[idx]
            trans = translations[idx]

            # 旋转矩阵转四元数
            matrix = np.eye(4)
            matrix[:3, :3] = rot
            matrix[:3, 3] = trans
            q = quaternion_from_matrix(matrix)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = trans
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
            
            pose_array_msg.poses.append(pose)
            info_data.extend([width, score, depth])

        self.pub_pose_array.publish(pose_array_msg)
        
        info_msg = Float32MultiArray(data=info_data)
        self.pub_info.publish(info_msg)
        
        rospy.loginfo(f"⚡ 已批量发布 Top-{top_k} 个姿态供 MoveIt 筛选！最优融合得分: {combined_scores[top_indices[0]]:.4f}")

        # =======================================================
        # --- 7. 下班清扫战场，防止内存溢出 ---
        # =======================================================
        del cloud_tensor
        del end_points
        torch.cuda.empty_cache()

        
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

if __name__ == '__main__':
    node = SamGraspNode()
    rospy.spin()
