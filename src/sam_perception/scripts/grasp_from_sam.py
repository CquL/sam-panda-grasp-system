#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import sys
import torch
import cv2
import threading
import traceback
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from std_msgs.msg import Float32MultiArray
from tf.transformations import quaternion_from_matrix
import tf.transformations as tft
import tf2_ros
from gazebo_msgs.srv import GetModelState
from cv_bridge import CvBridge
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray

# 环境配置 (保持你原有的路径)
GRASPNET_ROOT = os.path.join(os.path.expanduser('~'), 'grasp_robot_ws', 'graspnet-baseline')
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
        self.save_thesis_figures = rospy.get_param("~save_thesis_figures", False)
        self.figure_output_dir = os.path.expanduser(
            rospy.get_param("~figure_output_dir", "~/grasp_robot_ws/thesis_figures")
        )
        self.figure_index = 0
        self.visualize_grasp_on_depth = rospy.get_param("~visualize_grasp_on_depth", True)
        self.visualize_grasp_o3d = rospy.get_param("~visualize_grasp_o3d", False)
        self.top_k_per_object = rospy.get_param("~top_k_per_object", 15)
        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_rgb = None
        self.intrinsics = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.get_model_state = None
        self.shelf_model_name = str(rospy.get_param("~shelf_model_name", "narrow_supermarket_shelf_enclosed_0"))
        self.shelf_pose_fallback = {
            "x": float(rospy.get_param("~shelf_pose_fallback_x", 0.737098)),
            "y": float(rospy.get_param("~shelf_pose_fallback_y", -0.148598)),
            "z": float(rospy.get_param("~shelf_pose_fallback_z", 0.205537)),
            "yaw": float(rospy.get_param("~shelf_pose_fallback_yaw", 0.0)),
        }
        self.o3d_lock = threading.Lock()
        self.o3d_raw_items = None
        self.o3d_corrected_items = None
        self.o3d_thread = None

        if self.visualize_grasp_on_depth:
            cv2.namedWindow("GraspNet Raw Poses", cv2.WINDOW_NORMAL)
            cv2.namedWindow("GraspNet Corrected Poses", cv2.WINDOW_NORMAL)
        if self.visualize_grasp_o3d:
            self.o3d_thread = threading.Thread(target=self.o3d_visualization_loop, daemon=True)
            self.o3d_thread.start()
        
        # 1. 模型加载 (使用你之前的 rs.tar 权重)
        checkpoint_path = os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar') 
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
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback, queue_size=1)
        
        # 3. 发布：抓取位姿 和 抓取参数 (为了无缝对接 demo.py)
        # 【修改】使用 PoseArray 发布多个姿态
        # self.pub_pose_array = rospy.Publisher('/graspnet/grasp_pose_array', PoseArray, queue_size=1)
        # self.pub_info = rospy.Publisher('/graspnet/grasp_info', Float32MultiArray, queue_size=1)

        self.pub_pose_array = rospy.Publisher('/graspnet/grasp_pose_array_raw', PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher('/graspnet/grasp_info_raw', Float32MultiArray, queue_size=1)
        self.pub_raw_markers = rospy.Publisher('/graspnet/raw_grasp_markers', MarkerArray, queue_size=1)
        self.pub_corrected_markers = rospy.Publisher('/graspnet/corrected_grasp_markers', MarkerArray, queue_size=1)
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=0.5)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState, persistent=True)
        except Exception:
            rospy.logwarn("⚠️ GraspNet 节点未连接到 /gazebo/get_model_state，将使用货架 fallback 位姿。")

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            pass

    def info_callback(self, msg):
        self.intrinsics = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def get_shelf_world_pose(self):
        if self.get_model_state is not None:
            try:
                resp = self.get_model_state(self.shelf_model_name, "world")
                if getattr(resp, "success", False):
                    q = resp.pose.orientation
                    yaw = float(tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2])
                    return {
                        "x": float(resp.pose.position.x),
                        "y": float(resp.pose.position.y),
                        "z": float(resp.pose.position.z),
                        "yaw": yaw,
                    }
            except Exception as e:
                rospy.logwarn(f"⚠️ 获取货架位姿失败，改用 fallback: {e}")
        return dict(self.shelf_pose_fallback)

    def get_sensor_to_world_matrix(self, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                "world", source_frame, rospy.Time(0), rospy.Duration(0.5)
            )
            q = transform.transform.rotation
            mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            mat[:3, 3] = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                dtype=np.float64,
            )
            return mat
        except Exception as e:
            rospy.logwarn(f"⚠️ 获取 {source_frame} -> world TF 失败: {e}")
            return None

    def get_shelf_inner_regions_local(self):
        return [
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.235, "zmax": 0.385},
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.435, "zmax": 0.585},
            {"xmin": -0.12, "xmax": 0.12, "ymin": -0.27, "ymax": 0.27, "zmin": 0.635, "zmax": 0.775},
        ]

    def estimate_box_front_fallback_grasp(self, object_cloud, source_frame, width_limit, approach_offset):
        if len(object_cloud) < 30:
            return None

        sensor_to_world = self.get_sensor_to_world_matrix(source_frame)
        if sensor_to_world is None:
            return None
        world_to_sensor = np.linalg.inv(sensor_to_world)

        shelf_pose = self.get_shelf_world_pose()
        shelf_tf = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        shelf_tf[:3, 3] = np.array([shelf_pose["x"], shelf_pose["y"], shelf_pose["z"]], dtype=np.float64)
        world_to_shelf = np.linalg.inv(shelf_tf)

        points_h = np.hstack([object_cloud.astype(np.float64), np.ones((len(object_cloud), 1), dtype=np.float64)])
        points_world = (sensor_to_world @ points_h.T).T
        points_local = (world_to_shelf @ points_world.T).T[:, :3]

        mins = np.min(points_local, axis=0)
        maxs = np.max(points_local, axis=0)
        size_local = maxs - mins
        width_y = float(size_local[1])
        depth_x = float(size_local[0])
        height_z = float(size_local[2])

        fallback_width = float(width_y + 0.004)
        if fallback_width > width_limit:
            return None

        center_local = np.array(
            [
                0.5 * (mins[0] + maxs[0]),
                0.5 * (mins[1] + maxs[1]),
                mins[2] + max(0.04, min(0.5 * height_z, height_z - 0.02)),
            ],
            dtype=np.float64,
        )

        center_world_h = shelf_tf @ np.array([center_local[0], center_local[1], center_local[2], 1.0], dtype=np.float64)
        center_world = center_world_h[:3]

        approach_world = shelf_tf[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        approach_world = approach_world / max(np.linalg.norm(approach_world), 1e-6)
        down_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        lateral_world = np.cross(approach_world, down_world)
        lateral_norm = np.linalg.norm(lateral_world)
        if lateral_norm < 1e-6:
            return None
        lateral_world = lateral_world / lateral_norm

        rot_world = np.column_stack([approach_world, down_world, lateral_world])
        rot_sensor = world_to_sensor[:3, :3] @ rot_world
        translation_sensor = (world_to_sensor @ np.array([center_world[0], center_world[1], center_world[2], 1.0], dtype=np.float64))[:3]
        translation_sensor = translation_sensor - rot_sensor[:, 2] * approach_offset

        return {
            "score": 0.20,
            "width": fallback_width,
            "depth": float(max(0.02, min(0.08, depth_x))),
            "rotation": rot_sensor.astype(np.float32),
            "translation": translation_sensor.astype(np.float32),
        }

    def compute_candidate_context_features(self, translations, rotations, source_frame, xyz, object_ids, current_object_id):
        num_candidates = len(translations)
        neutral = {
            "forward_alignment": np.zeros(num_candidates, dtype=np.float32),
            "cavity_clearance_norm": np.zeros(num_candidates, dtype=np.float32),
            "neighbor_clearance_norm": np.zeros(num_candidates, dtype=np.float32),
            "context_valid_mask": np.ones(num_candidates, dtype=bool),
            "inside_mask": np.zeros(num_candidates, dtype=bool),
        }
        sensor_to_world = self.get_sensor_to_world_matrix(source_frame)
        if sensor_to_world is None:
            return neutral

        shelf_pose = self.get_shelf_world_pose()
        shelf_tf = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        shelf_tf[:3, 3] = np.array([shelf_pose["x"], shelf_pose["y"], shelf_pose["z"]], dtype=np.float64)
        world_to_shelf = np.linalg.inv(shelf_tf)
        rot_world_to_shelf = world_to_shelf[:3, :3]
        rot_sensor_to_world = sensor_to_world[:3, :3]

        translations_world = (rot_sensor_to_world @ translations.T).T + sensor_to_world[:3, 3]
        translations_world_h = np.hstack([translations_world, np.ones((num_candidates, 1), dtype=np.float64)])
        translations_local = (world_to_shelf @ translations_world_h.T).T[:, :3]

        approach_world = np.einsum("ij,njk->nik", rot_sensor_to_world, rotations)[:, :, 0]
        approach_local = (rot_world_to_shelf @ approach_world.T).T
        # 这里只关心“是否沿货架法向”，不关心符号。
        # 实际执行端已经会强制沿货架法向插入，所以此处不应因正负号把候选全过滤掉。
        forward_alignment = np.clip(np.abs(approach_local[:, 0]), 0.0, 1.0).astype(np.float32)

        preferred_shelf_clearance = float(rospy.get_param("~preferred_shelf_clearance", 0.04))
        preferred_neighbor_clearance = float(rospy.get_param("~preferred_neighbor_clearance", 0.05))
        _min_shelf_forward_alignment = float(rospy.get_param("~min_shelf_forward_alignment", 0.25))

        cavity_clearance = np.zeros(num_candidates, dtype=np.float32)
        inside_mask = np.zeros(num_candidates, dtype=bool)
        regions = self.get_shelf_inner_regions_local()
        for idx, point in enumerate(translations_local):
            for region in regions:
                if (
                    region["xmin"] <= point[0] <= region["xmax"]
                    and region["ymin"] <= point[1] <= region["ymax"]
                    and region["zmin"] <= point[2] <= region["zmax"]
                ):
                    inside_mask[idx] = True
                    cavity_clearance[idx] = float(
                        min(
                            point[0] - region["xmin"],
                            region["xmax"] - point[0],
                            point[1] - region["ymin"],
                            region["ymax"] - point[1],
                            point[2] - region["zmin"],
                            region["zmax"] - point[2],
                        )
                    )
                    break

        cavity_clearance_norm = np.clip(cavity_clearance / max(1e-6, preferred_shelf_clearance), 0.0, 1.0).astype(np.float32)

        other_points = xyz[object_ids != current_object_id]
        if len(other_points) > 1500:
            sample_idx = np.random.choice(len(other_points), 1500, replace=False)
            other_points = other_points[sample_idx]
        if len(other_points) > 0:
            deltas = translations[:, None, :] - other_points[None, :, :]
            neighbor_clearance = np.linalg.norm(deltas, axis=2).min(axis=1)
        else:
            neighbor_clearance = np.full(num_candidates, preferred_neighbor_clearance, dtype=np.float32)
        neighbor_clearance_norm = np.clip(neighbor_clearance / max(1e-6, preferred_neighbor_clearance), 0.0, 1.0).astype(np.float32)

        enforce_shelf_cavity_filter = bool(rospy.get_param("~enforce_shelf_cavity_filter", False))
        # 默认不再把腔体约束当成硬过滤，只作为打分项。
        # 否则在相机/模型有轻微偏差时很容易把所有候选直接杀光。
        context_valid_mask = inside_mask.copy() if enforce_shelf_cavity_filter else np.ones(num_candidates, dtype=bool)
        return {
            "forward_alignment": forward_alignment,
            "cavity_clearance_norm": cavity_clearance_norm,
            "neighbor_clearance_norm": neighbor_clearance_norm,
            "context_valid_mask": context_valid_mask,
            "inside_mask": inside_mask,
        }

    def callback(self, pc_msg):
        try:
            rows = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z", "object_id"), skip_nans=True))
            if len(rows) < 50:
                rospy.logwarn("点云点数太少，无法计算抓取")
                return

            cloud_array = np.array(rows, dtype=np.float64)
            xyz = cloud_array[:, :3].astype(np.float32)
            object_ids = cloud_array[:, 3].astype(np.int32)
            unique_object_ids = sorted(np.unique(object_ids))
            rospy.loginfo(f"🧩 GraspNet 收到带 object_id 点云，目标数: {len(unique_object_ids)}")
            if len(unique_object_ids) == 0:
                rospy.logwarn("没有解析到有效的 object_id")
                return

            num_point = 8192
            grasp_sampling_passes = max(1, int(rospy.get_param("~grasp_sampling_passes", 3)))
            approach_offset = rospy.get_param("~approach_offset", 0.02)
            penalty_weight = float(rospy.get_param("~center_penalty_weight", 1.5))
            front_alignment_weight = float(rospy.get_param("~front_alignment_weight", 0.25))
            width_preference_weight = float(rospy.get_param("~width_preference_weight", 0.35))
            shelf_clearance_weight = float(rospy.get_param("~shelf_clearance_weight", 0.45))
            neighbor_clearance_weight = float(rospy.get_param("~neighbor_clearance_weight", 0.35))
            allow_clearance_fallback = bool(rospy.get_param("~allow_clearance_fallback", False))
            enable_box_fallback_grasp = bool(rospy.get_param("~enable_box_fallback_grasp", True))
            enable_width_filter = bool(rospy.get_param("~enable_width_filter", True))
            gripper_width_max = float(rospy.get_param("~gripper_width_max", 0.078))
            gripper_width_margin = float(rospy.get_param("~gripper_width_margin", 0.004))

            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = rospy.Time.now()
            pose_array_msg.header.frame_id = pc_msg.header.frame_id
            info_data = []

            raw_rot_vis = []
            raw_trans_vis = []
            corr_rot_vis = []
            corr_trans_vis = []
            widths_vis = []
            depths_vis = []
            vis_cloud_parts = []
            published_object_count = 0
            skipped_width_object_count = 0

            for object_id in unique_object_ids:
                object_cloud = xyz[object_ids == object_id]
                rospy.loginfo(f"  ↳ 目标 {object_id}: 点数 {len(object_cloud)}")
                if len(object_cloud) < 50:
                    rospy.logwarn(f"目标 {object_id} 点云点数太少，跳过")
                    continue

                vis_cloud_parts.append(object_cloud)

                sampled_cloud = None
                merged_grasps = []
                successful_passes = 0
                for pass_idx in range(grasp_sampling_passes):
                    if len(object_cloud) >= num_point:
                        idxs = np.random.choice(len(object_cloud), num_point, replace=False)
                    else:
                        idxs = np.random.choice(len(object_cloud), num_point, replace=True)
                    sampled_cloud_pass = object_cloud[idxs]
                    if sampled_cloud is None:
                        sampled_cloud = sampled_cloud_pass

                    cloud_tensor = torch.from_numpy(sampled_cloud_pass[np.newaxis].astype(np.float32)).to(self.device)
                    end_points = {'point_clouds': cloud_tensor, 'cloud_colors': cloud_tensor}
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        end_points = self.net(end_points)
                        grasp_preds = pred_decode(end_points)

                    grasps_array_pass = grasp_preds[0].detach().cpu().numpy()
                    if len(grasps_array_pass) > 0:
                        merged_grasps.append(grasps_array_pass)
                        successful_passes += 1

                    del cloud_tensor
                    del end_points
                    torch.cuda.empty_cache()

                width_limit = max(0.0, gripper_width_max - gripper_width_margin)
                fallback_grasp = None
                if enable_box_fallback_grasp:
                    fallback_grasp = self.estimate_box_front_fallback_grasp(
                        object_cloud=object_cloud,
                        source_frame=pc_msg.header.frame_id,
                        width_limit=width_limit,
                        approach_offset=approach_offset,
                    )

                if len(merged_grasps) == 0:
                    if fallback_grasp is not None:
                        rospy.logwarn(
                            f"目标 {object_id}: GraspNet 在 {grasp_sampling_passes} 次随机采样中均未输出姿态，"
                            "改用几何 box fallback 候选。"
                        )
                        scores = np.array([fallback_grasp["score"]], dtype=np.float32)
                        widths = np.array([fallback_grasp["width"]], dtype=np.float32)
                        depths = np.array([fallback_grasp["depth"]], dtype=np.float32)
                        rotations = fallback_grasp["rotation"][None, :, :]
                        translations = fallback_grasp["translation"][None, :]
                        raw_scores = scores.copy()
                        raw_translations = translations.copy()
                        raw_rotations = rotations.copy()
                    else:
                        rospy.logwarn(
                            f"目标 {object_id}: GraspNet 在 {grasp_sampling_passes} 次随机采样中均未输出姿态"
                        )
                        continue
                else:
                    grasps_array = np.concatenate(merged_grasps, axis=0)
                    rospy.loginfo(
                        f"目标 {object_id}: 多次采样汇总候选 {len(grasps_array)} 个 "
                        f"(successful_passes={successful_passes}/{grasp_sampling_passes})"
                    )

                    scores = grasps_array[:, 0]
                    widths = grasps_array[:, 1]
                    depths = grasps_array[:, 3]
                    rotations = grasps_array[:, 4:13].reshape(-1, 3, 3)
                    translations = grasps_array[:, 13:16]
                    raw_scores = scores.copy()
                    raw_translations = translations.copy()
                    raw_rotations = rotations.copy()

                if np.max(scores) < 0.05:
                    rospy.logwarn(f"目标 {object_id}: 所有抓取分数过低，跳过")
                    continue

                forward_vector = rotations[:, :, 2]
                translations = translations + forward_vector * approach_offset

                object_center = np.mean(object_cloud, axis=0)
                distances = np.linalg.norm(translations - object_center, axis=1)
                max_dist = np.max(distances) + 1e-5
                normalized_distances = distances / max_dist
                context = self.compute_candidate_context_features(
                    translations=translations,
                    rotations=rotations,
                    source_frame=pc_msg.header.frame_id,
                    xyz=xyz,
                    object_ids=object_ids,
                    current_object_id=object_id,
                )
                forward_alignment = context["forward_alignment"]
                cavity_clearance_norm = context["cavity_clearance_norm"]
                neighbor_clearance_norm = context["neighbor_clearance_norm"]
                corrected_scores = (
                    scores
                    - (penalty_weight * normalized_distances)
                    + (front_alignment_weight * forward_alignment)
                    + (shelf_clearance_weight * cavity_clearance_norm)
                    + (neighbor_clearance_weight * neighbor_clearance_norm)
                )

                valid_indices = np.arange(len(corrected_scores), dtype=np.int32)
                if enable_width_filter:
                    valid_width = widths <= width_limit
                    if np.any(valid_width):
                        invalid_count = int(np.count_nonzero(~valid_width))
                        if invalid_count > 0:
                            rospy.loginfo(
                                f"目标 {object_id}: 宽度过滤剔除 {invalid_count} 个候选 "
                                f"(limit={width_limit:.3f} m)"
                            )
                        valid_indices = np.flatnonzero(valid_width).astype(np.int32)
                    else:
                        if fallback_grasp is not None and fallback_grasp["width"] <= width_limit:
                            rospy.logwarn(
                                f"目标 {object_id}: 所有 GraspNet 候选宽度均超过夹爪上限 {width_limit:.3f} m，"
                                "改用几何 box fallback 候选。"
                            )
                            scores = np.array([fallback_grasp["score"]], dtype=np.float32)
                            widths = np.array([fallback_grasp["width"]], dtype=np.float32)
                            depths = np.array([fallback_grasp["depth"]], dtype=np.float32)
                            rotations = fallback_grasp["rotation"][None, :, :]
                            translations = fallback_grasp["translation"][None, :]
                            raw_scores = scores.copy()
                            raw_translations = translations.copy()
                            raw_rotations = rotations.copy()
                            corrected_scores = scores.copy()
                            valid_indices = np.array([0], dtype=np.int32)
                        else:
                            rospy.logwarn(
                                f"目标 {object_id}: 所有候选宽度均超过夹爪上限 {width_limit:.3f} m，"
                                "当前轮直接标记为不可抓并跳过发布。"
                            )
                            skipped_width_object_count += 1
                            continue
                width_valid_indices = valid_indices.copy()

                context_valid_indices = np.flatnonzero(context["context_valid_mask"]).astype(np.int32)
                if len(context_valid_indices) > 0 and len(context_valid_indices) < len(valid_indices):
                    context_invalid_count = int(np.count_nonzero(~context["context_valid_mask"]))
                    if context_invalid_count > 0:
                        rospy.loginfo(
                            f"目标 {object_id}: 腔体约束过滤剔除 {context_invalid_count} 个候选"
                        )
                    valid_indices = np.intersect1d(valid_indices, context_valid_indices, assume_unique=False)
                    if len(valid_indices) == 0:
                        if allow_clearance_fallback:
                            rospy.logwarn(f"目标 {object_id}: 腔体约束过滤后无候选，回退到宽度过滤结果。")
                            valid_indices = width_valid_indices
                        else:
                            rospy.logwarn(
                                f"目标 {object_id}: 腔体约束过滤后无候选，当前轮直接跳过，避免退回到坏姿态。"
                            )
                            continue

                if width_limit > 1e-6:
                    width_ratio = np.clip(widths / width_limit, 0.0, 1.5)
                    corrected_scores = corrected_scores - (width_preference_weight * width_ratio)

                if len(valid_indices) == 0:
                    rospy.logwarn(f"目标 {object_id}: 没有剩余可用候选，跳过")
                    continue

                ranked_valid = valid_indices[np.argsort(-corrected_scores[valid_indices])]
                top_k = min(self.top_k_per_object, len(ranked_valid))
                top_indices = ranked_valid[:top_k]
                raw_top_indices = np.argsort(-raw_scores)[: min(self.top_k_per_object, len(raw_scores))]
                rospy.loginfo(f"目标 {object_id}: 保留姿态 {top_k} 个")

                raw_rot_vis.append(raw_rotations)
                raw_trans_vis.append(raw_translations)
                corr_rot_vis.append(rotations)
                corr_trans_vis.append(translations)
                widths_vis.append(widths)
                depths_vis.append(depths)

                self.save_grasp_comparison_figure(
                    sampled_cloud=sampled_cloud,
                    raw_translations=raw_translations,
                    raw_scores=raw_scores,
                    raw_top_indices=raw_top_indices,
                    corrected_translations=translations,
                    corrected_scores=corrected_scores,
                    corrected_top_indices=top_indices,
                )

                for idx in top_indices:
                    matrix = np.eye(4)
                    matrix[:3, :3] = rotations[idx]
                    matrix[:3, 3] = translations[idx]
                    q = quaternion_from_matrix(matrix)

                    pose = Pose()
                    pose.position.x, pose.position.y, pose.position.z = translations[idx]
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
                    pose_array_msg.poses.append(pose)
                    info_data.extend([float(object_id), widths[idx], corrected_scores[idx], depths[idx]])
                published_object_count += 1

            if len(pose_array_msg.poses) == 0:
                rospy.logwarn("没有生成任何可用的抓取姿态（可能都超出夹爪宽度上限或分数不足）")
                return

            if skipped_width_object_count > 0:
                rospy.logwarn(
                    f"⚠️ 本轮有 {skipped_width_object_count} 个目标因所有候选超出夹爪宽度上限而被跳过。"
                )

            rospy.loginfo("🧪 正在汇总各物体抓取结果并准备发布...")
            vis_cloud = np.concatenate(vis_cloud_parts, axis=0) if len(vis_cloud_parts) > 0 else xyz
            raw_rot_vis = np.concatenate(raw_rot_vis, axis=0)
            raw_trans_vis = np.concatenate(raw_trans_vis, axis=0)
            corr_rot_vis = np.concatenate(corr_rot_vis, axis=0)
            corr_trans_vis = np.concatenate(corr_trans_vis, axis=0)
            widths_vis = np.concatenate(widths_vis, axis=0)
            depths_vis = np.concatenate(depths_vis, axis=0)

            self.pub_pose_array.publish(pose_array_msg)
            self.pub_info.publish(Float32MultiArray(data=info_data))
            self.pub_raw_markers.publish(
                self.build_grasp_marker_array(
                    frame_id=pc_msg.header.frame_id,
                    rotations=raw_rot_vis,
                    translations=raw_trans_vis,
                    widths=widths_vis,
                    depths=depths_vis,
                    indices=np.arange(len(raw_rot_vis)),
                    namespace="raw_grasps",
                    color=(1.0, 0.2, 0.2),
                )
            )
            self.pub_corrected_markers.publish(
                self.build_grasp_marker_array(
                    frame_id=pc_msg.header.frame_id,
                    rotations=corr_rot_vis,
                    translations=corr_trans_vis,
                    widths=widths_vis,
                    depths=depths_vis,
                    indices=np.arange(len(corr_rot_vis)),
                    namespace="corrected_grasps",
                    color=(0.2, 1.0, 0.2),
                )
            )
            rospy.loginfo(
                f"⚡ 已按物体独立发布抓取姿态！物体数: {published_object_count}, 总姿态数: {len(pose_array_msg.poses)}"
            )

            all_indices = np.arange(len(raw_rot_vis))
            try:
                self.visualize_grasp_poses_on_depth(
                    raw_rotations=raw_rot_vis,
                    raw_translations=raw_trans_vis,
                    raw_indices=all_indices,
                    corrected_rotations=corr_rot_vis,
                    corrected_translations=corr_trans_vis,
                    corrected_indices=all_indices,
                )
            except Exception as e:
                rospy.logwarn(f"⚠️ 2D grasp 可视化失败（不影响主流程）: {e}")

            try:
                self.update_o3d_visualization(
                    sampled_cloud=vis_cloud,
                    widths=widths_vis,
                    depths=depths_vis,
                    raw_rotations=raw_rot_vis,
                    raw_translations=raw_trans_vis,
                    raw_indices=all_indices,
                    corrected_rotations=corr_rot_vis,
                    corrected_translations=corr_trans_vis,
                    corrected_indices=all_indices,
                )
            except Exception as e:
                rospy.logwarn(f"⚠️ 3D grasp 可视化失败（不影响主流程）: {e}")
        except Exception as e:
            rospy.logerr(f"❌ grasp_from_sam callback 异常: {e}")
            rospy.logerr(traceback.format_exc())

    def create_gripper_lineset(self, rotation, translation, width=0.06, depth=0.06, color=(0.0, 1.0, 0.0)):
        """画一个简化版平行夹爪线框。局部坐标系: z为前进方向, x为开合方向。"""
        half_w = max(0.005, float(width) * 0.5)
        finger_len = min(0.05, max(0.025, float(depth) + 0.01))
        palm_back = 0.02
        palm_height = 0.02

        local_points = np.array([
            [-half_w, 0.0, -palm_back],   # 0 left back
            [ half_w, 0.0, -palm_back],   # 1 right back
            [-half_w, 0.0, 0.0],          # 2 left finger base
            [ half_w, 0.0, 0.0],          # 3 right finger base
            [-half_w, 0.0, finger_len],   # 4 left tip
            [ half_w, 0.0, finger_len],   # 5 right tip
            [0.0,  palm_height, -palm_back],  # 6 palm top back
            [0.0,  palm_height, 0.0],         # 7 palm top front
        ], dtype=np.float64)

        world_points = (rotation @ local_points.T).T + translation
        lines = [
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [0, 6], [1, 6], [6, 7], [2, 7], [3, 7]
        ]
        colors = [list(color) for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(world_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def local_gripper_points(self, width=0.06, depth=0.06):
        half_w = max(0.005, float(width) * 0.5)
        finger_len = min(0.05, max(0.025, float(depth) + 0.01))
        palm_back = 0.02
        palm_height = 0.02
        return np.array([
            [-half_w, 0.0, -palm_back],
            [ half_w, 0.0, -palm_back],
            [-half_w, 0.0, 0.0],
            [ half_w, 0.0, 0.0],
            [-half_w, 0.0, finger_len],
            [ half_w, 0.0, finger_len],
            [0.0,  palm_height, -palm_back],
            [0.0,  palm_height, 0.0],
        ], dtype=np.float64)

    def build_grasp_marker_array(self, frame_id, rotations, translations, widths, depths, indices, namespace, color):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = namespace
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        line_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 6), (1, 6), (6, 7), (2, 7), (3, 7)]
        for marker_id, idx in enumerate(indices, start=1):
            pts_local = self.local_gripper_points(widths[idx], depths[idx])
            pts_world = (rotations[idx] @ pts_local.T).T + translations[idx]

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.002
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.85

            for a, b in line_pairs:
                pa = Point(*pts_world[a])
                pb = Point(*pts_world[b])
                marker.points.append(pa)
                marker.points.append(pb)

            marker_array.markers.append(marker)
        return marker_array

    def update_o3d_visualization(self, sampled_cloud, widths, depths, raw_rotations, raw_translations, raw_indices,
                                 corrected_rotations, corrected_translations, corrected_indices):
        if not self.visualize_grasp_o3d:
            return

        cloud_vis = sampled_cloud[::max(1, len(sampled_cloud) // 4000)]
        items_raw = {
            "cloud": cloud_vis.astype(np.float64),
            "grippers": [],
        }
        items_corrected = {
            "cloud": cloud_vis.astype(np.float64),
            "grippers": [],
        }

        for idx in raw_indices:
            items_raw["grippers"].append(
                {
                    "rotation": raw_rotations[idx],
                    "translation": raw_translations[idx],
                    "width": float(widths[idx]),
                    "depth": float(depths[idx]),
                    "color": (1.0, 0.3, 0.3),
                }
            )

        for idx in corrected_indices:
            items_corrected["grippers"].append(
                {
                    "rotation": corrected_rotations[idx],
                    "translation": corrected_translations[idx],
                    "width": float(widths[idx]),
                    "depth": float(depths[idx]),
                    "color": (0.2, 1.0, 0.3),
                }
            )

        with self.o3d_lock:
            self.o3d_raw_items = items_raw
            self.o3d_corrected_items = items_corrected

    def render_o3d_items(self, vis, items):
        vis.clear_geometries()
        if items is None:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(items["cloud"])
        pcd.paint_uniform_color([0.75, 0.75, 0.75])
        vis.add_geometry(pcd, reset_bounding_box=True)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        vis.add_geometry(frame, reset_bounding_box=False)

        for grip in items["grippers"]:
            geom = self.create_gripper_lineset(
                rotation=grip["rotation"],
                translation=grip["translation"],
                width=grip["width"],
                depth=grip["depth"],
                color=grip["color"],
            )
            vis.add_geometry(geom, reset_bounding_box=False)

    def o3d_visualization_loop(self):
        raw_vis = o3d.visualization.Visualizer()
        corrected_vis = o3d.visualization.Visualizer()
        raw_vis.create_window(window_name="GraspNet Raw 3D", width=960, height=720)
        corrected_vis.create_window(window_name="GraspNet Corrected 3D", width=960, height=720)
        raw_vis.get_render_option().background_color = np.asarray([0.08, 0.08, 0.08])
        corrected_vis.get_render_option().background_color = np.asarray([0.08, 0.08, 0.08])

        while not rospy.is_shutdown():
            with self.o3d_lock:
                raw_items = self.o3d_raw_items
                corrected_items = self.o3d_corrected_items

            if raw_items is not None:
                self.render_o3d_items(raw_vis, raw_items)
                self.o3d_raw_items = None
            if corrected_items is not None:
                self.render_o3d_items(corrected_vis, corrected_items)
                self.o3d_corrected_items = None

            raw_vis.poll_events()
            raw_vis.update_renderer()
            corrected_vis.poll_events()
            corrected_vis.update_renderer()
            rospy.sleep(0.05)

    def depth_to_bgr(self, depth_img):
        finite = np.isfinite(depth_img) & (depth_img > 0.0)
        vis = np.zeros_like(depth_img, dtype=np.uint8)
        if np.any(finite):
            depth_valid = depth_img[finite]
            d_min = float(np.percentile(depth_valid, 5))
            d_max = float(np.percentile(depth_valid, 95))
            if d_max <= d_min:
                d_max = d_min + 1e-6
            scaled = np.clip((depth_img - d_min) / (d_max - d_min), 0.0, 1.0)
            vis = (scaled * 255.0).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    def make_visualization_canvas(self):
        if self.latest_depth is not None:
            finite = np.isfinite(self.latest_depth) & (self.latest_depth > 0.0)
            if np.any(finite):
                return self.depth_to_bgr(self.latest_depth), "Depth"

        if self.latest_rgb is not None:
            return self.latest_rgb.copy(), "RGB"

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Waiting for depth/RGB image...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return canvas, "None"

    def project_point(self, point_xyz):
        if self.intrinsics is None:
            return None
        x, y, z = point_xyz
        if z <= 1e-6:
            return None
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return u, v

    def draw_pose_overlay(self, canvas, rotations, translations, indices, label_prefix):
        if canvas is None:
            return 0

        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z
        axis_length = 0.03
        drawn_count = 0

        for rank, idx in enumerate(indices[:12]):
            center = translations[idx]
            center_px = self.project_point(center)
            if center_px is None:
                continue

            u, v = center_px
            if not (0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]):
                continue

            cv2.circle(canvas, (u, v), 3, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"{label_prefix}{rank+1}",
                (u + 4, max(v - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            rot = rotations[idx]
            for axis_i in range(3):
                end_point = center + rot[:, axis_i] * axis_length
                end_px = self.project_point(end_point)
                if end_px is None:
                    continue
                cv2.arrowedLine(
                    canvas,
                    (u, v),
                    end_px,
                    axis_colors[axis_i],
                    2,
                    tipLength=0.2,
                )
            drawn_count += 1

        return drawn_count

    def visualize_grasp_poses_on_depth(
        self,
        raw_rotations,
        raw_translations,
        raw_indices,
        corrected_rotations,
        corrected_translations,
        corrected_indices,
    ):
        if not self.visualize_grasp_on_depth:
            return
        raw_canvas, background_name = self.make_visualization_canvas()
        corrected_canvas = raw_canvas.copy()

        raw_drawn = 0
        corrected_drawn = 0
        if self.intrinsics is not None:
            raw_drawn = self.draw_pose_overlay(raw_canvas, raw_rotations, raw_translations, raw_indices, "R")
            corrected_drawn = self.draw_pose_overlay(corrected_canvas, corrected_rotations, corrected_translations, corrected_indices, "C")

        cv2.putText(raw_canvas, f"Before Correction [{background_name}]", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(corrected_canvas, f"After Correction [{background_name}]", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(raw_canvas, f"Projected poses: {raw_drawn}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(corrected_canvas, f"Projected poses: {corrected_drawn}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.intrinsics is None:
            cv2.putText(raw_canvas, "No camera intrinsics yet", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(corrected_canvas, "No camera intrinsics yet", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif raw_drawn == 0 and corrected_drawn == 0:
            cv2.putText(raw_canvas, "No valid projected grasp poses", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(corrected_canvas, "No valid projected grasp poses", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("GraspNet Raw Poses", raw_canvas)
        cv2.imshow("GraspNet Corrected Poses", corrected_canvas)
        cv2.waitKey(1)

    def save_grasp_comparison_figure(
        self,
        sampled_cloud,
        raw_translations,
        raw_scores,
        raw_top_indices,
        corrected_translations,
        corrected_scores,
        corrected_top_indices,
    ):
        if not self.save_thesis_figures:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            rospy.logwarn(f"无法导入 matplotlib，跳过抓取对比图保存: {e}")
            return

        os.makedirs(self.figure_output_dir, exist_ok=True)
        self.figure_index += 1
        prefix = f"{self.figure_index:03d}"

        # 点云下采样一点，避免图太密
        cloud_vis = sampled_cloud[::max(1, len(sampled_cloud) // 2000)]
        raw_vis = raw_translations[raw_top_indices[:20]]
        corrected_vis = corrected_translations[corrected_top_indices[:20]]

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(cloud_vis[:, 0], cloud_vis[:, 1], cloud_vis[:, 2], s=1, c="lightgray", alpha=0.6)
        ax1.scatter(raw_vis[:, 0], raw_vis[:, 1], raw_vis[:, 2], s=20, c="red", alpha=0.9)
        ax1.set_title("Before Physical Correction")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(cloud_vis[:, 0], cloud_vis[:, 1], cloud_vis[:, 2], s=1, c="lightgray", alpha=0.6)
        ax2.scatter(corrected_vis[:, 0], corrected_vis[:, 1], corrected_vis[:, 2], s=20, c="green", alpha=0.9)
        ax2.set_title("After Physical Correction")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        fig.tight_layout()
        fig.savefig(os.path.join(self.figure_output_dir, f"{prefix}_grasp_correction_compare.png"), dpi=200)
        plt.close(fig)

        np.savez(
            os.path.join(self.figure_output_dir, f"{prefix}_grasp_correction_data.npz"),
            sampled_cloud=sampled_cloud,
            raw_translations=raw_translations,
            raw_scores=raw_scores,
            corrected_translations=corrected_translations,
            corrected_scores=corrected_scores,
        )

        
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
