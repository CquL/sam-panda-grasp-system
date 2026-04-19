#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# ==============================================================================
# 1. 修复 Numpy 版本冲突
# ==============================================================================
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import rospy
import numpy as np
import cv2
import struct
import os
import tf.transformations as tft
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Float32MultiArray, String
from cv_bridge import CvBridge
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty, EmptyResponse
from gazebo_msgs.srv import GetModelState

class SAMPerceptionNode:
    def __init__(self):
        rospy.init_node('sam_perception_node')
        self.debug_o3d = rospy.get_param("~debug_o3d", False)
        self.bridge = CvBridge()
        self.save_thesis_figures = rospy.get_param("~save_thesis_figures", False)
        self.figure_output_dir = os.path.expanduser(
            rospy.get_param("~figure_output_dir", "~/grasp_robot_ws/thesis_figures")
        )
        self.figure_index = 0
        self.object_front_depth_percentile = float(
            rospy.get_param("~object_front_depth_percentile", 15.0)
        )
        self.object_depth_gate_band = float(
            rospy.get_param("~object_depth_gate_band", 0.10)
        )
        self.object_component_min_pixels = int(
            rospy.get_param("~object_component_min_pixels", 80)
        )
        self.publish_background_collision_cloud = bool(
            rospy.get_param("~publish_background_collision_cloud", False)
        )
        self.publish_local_obstacle_cloud = bool(
            rospy.get_param("~publish_local_obstacle_cloud", True)
        )
        self.local_obstacle_max_points = int(
            rospy.get_param("~local_obstacle_max_points", 6000)
        )
        self.shelf_model_name = str(
            rospy.get_param("~shelf_model_name", "narrow_supermarket_shelf_enclosed_0")
        )
        self.shelf_pose_fallback = {
            "x": float(rospy.get_param("~shelf_pose_fallback_x", 0.737098)),
            "y": float(rospy.get_param("~shelf_pose_fallback_y", -0.148598)),
            "z": float(rospy.get_param("~shelf_pose_fallback_z", 0.205537)),
            "yaw": float(rospy.get_param("~shelf_pose_fallback_yaw", 0.0)),
        }
        self.shelf_inner_margin_x = float(
            rospy.get_param("~shelf_inner_margin_x", 0.03)
        )
        self.shelf_inner_margin_y = float(
            rospy.get_param("~shelf_inner_margin_y", 0.03)
        )
        self.shelf_inner_margin_z = float(
            rospy.get_param("~shelf_inner_margin_z", 0.015)
        )
        self.background_min_depth = float(
            rospy.get_param("~background_min_depth", 0.55)
        )
        self.object_clearance_margin_x = float(
            rospy.get_param("~object_clearance_margin_x", 0.06)
        )
        self.object_clearance_margin_y = float(
            rospy.get_param("~object_clearance_margin_y", 0.08)
        )
        self.object_clearance_margin_z = float(
            rospy.get_param("~object_clearance_margin_z", 0.08)
        )
        self.collision_clearance_margin_px = int(
            rospy.get_param("~collision_clearance_margin_px", 35)
        )
        self.collision_clearance_kernel = int(
            rospy.get_param("~collision_clearance_kernel", 75)
        )
        self.collision_clearance_depth_band = float(
            rospy.get_param("~collision_clearance_depth_band", 0.20)
        )
        
        package_path = os.path.dirname(os.path.dirname(__file__))
        checkpoint = os.path.join(package_path, "models", "sam_vit_b_01ec64.pth")
        
        if not os.path.exists(checkpoint):
            rospy.logerr(f"❌ 找不到权重文件: {checkpoint}")
            sys.exit(1)

        rospy.loginfo("⏳ 正在初始化 SAM 模型...")
        try:
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam.to(device="cuda") 
            self.predictor = SamPredictor(sam)
            rospy.loginfo("✅ SAM 模型就绪！")
        except Exception as e:
            rospy.logerr(f"❌ 模型加载失败: {e}")
            sys.exit(1)

        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)
        self.cmd_sub = rospy.Subscriber("/sam/prompt_bbox", Float32MultiArray, self.cmd_callback)
        self.refresh_bg_srv = rospy.Service("/sam_perception/refresh_background", Empty, self.handle_refresh_background)
        
        self.cloud_pub = rospy.Publisher("/sam_perception/object_cloud", PointCloud2, queue_size=1)
        self.bg_cloud_pub = rospy.Publisher("/sam_perception/background_cloud", PointCloud2, queue_size=1)
        self.metadata_pub = rospy.Publisher("/sam_perception/object_metadata", String, queue_size=1)
        self.active_grasp_target = None
        rospy.Subscriber("/sam_perception/active_grasp_target", String, self.active_grasp_target_cb, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.get_model_state = None
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=0.5)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState, persistent=True)
        except Exception:
            rospy.logwarn("⚠️ SAM 节点未连接到 /gazebo/get_model_state，将使用货架 fallback 位姿。")

        self.curr_rgb = None
        self.curr_depth = None
        self.intrinsics = None
        self.sam_result_img = None
        self.sam_mask_img = None
        self.vlm_bbox_img = None

        self.drawing = False      
        self.box_start = None     
        self.target_boxes = []    
        self.should_process = False

        cv2.namedWindow("SAM Selection")
        cv2.setMouseCallback("SAM Selection", self.on_mouse_click)
        cv2.namedWindow("VLM BBoxes")
        cv2.namedWindow("SAM Result")
        cv2.namedWindow("SAM Mask")

    def active_grasp_target_cb(self, msg):
        text = msg.data.strip()
        if not text:
            self.active_grasp_target = None
            return
        try:
            payload = json.loads(text)
        except Exception:
            rospy.logwarn("⚠️ 无法解析 active_grasp_target，已忽略。")
            return
        if not isinstance(payload, dict) or not payload.get("enabled", True):
            self.active_grasp_target = None
            return
        self.active_grasp_target = payload

    def info_callback(self, msg):
        self.intrinsics = np.array(msg.K).reshape(3, 3)

    def rgb_callback(self, msg):
        try:
            self.curr_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def depth_callback(self, msg):
        try:
            self.curr_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            pass

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.box_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x_min, x_max = min(self.box_start[0], x), max(self.box_start[0], x)
            y_min, y_max = min(self.box_start[1], y), max(self.box_start[1], y)
            
            self.target_boxes = [(x_min, y_min, x_max, y_max)]
            rospy.loginfo(f"📦 手动单目标框选完成: {self.target_boxes[0]}")
            self.should_process = True

    def cmd_callback(self, msg):
        """处理来自 LLM 的多目标坐标指令"""
        data = msg.data
        if len(data) % 4 != 0 or len(data) == 0:
            rospy.logerr(f"❌ 收到无效的 BBox 数组，长度 {len(data)} 不是 4 的倍数！")
            return
            
        self.target_boxes = []
        num_targets = len(data) // 4
        rospy.loginfo(f"📥 SAM 收到 LLM 下发的 {num_targets} 个目标框指令！")
        
        for i in range(num_targets):
            idx = i * 4
            x_min, y_min, x_max, y_max = [int(val) for val in data[idx:idx+4]]
            self.target_boxes.append((x_min, y_min, x_max, y_max))
            rospy.loginfo(f"  -> 目标 {i+1}: [{x_min}, {y_min}] -> [{x_max}, {y_max}]")

        if self.curr_rgb is not None:
            bbox_vis = self.curr_rgb.copy()
            for i, (x_min, y_min, x_max, y_max) in enumerate(self.target_boxes):
                cv2.rectangle(bbox_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(
                    bbox_vis,
                    f"Target-{i+1}",
                    (x_min, max(y_min - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            self.vlm_bbox_img = bbox_vis
            
        self.drawing = False  
        self.should_process = True

    # =========================================================
    # 💥 下面的所有函数都已经重新修正了正确的类缩进 (4个空格)
    # =========================================================
    def process_segmentation(self):
        import torch

        if self.curr_rgb is None or self.curr_depth is None or not self.target_boxes:
            rospy.logwarn("⚠️ 数据不全或未获取到目标框，跳过处理")
            return

        combined_mask = np.zeros(self.curr_rgb.shape[:2], dtype=bool)
        object_masks = []
        object_metadata = []
        vis_img = self.curr_rgb.copy()
        image_set = False

        try:
            self.predictor.set_image(self.curr_rgb)
            image_set = True

            rospy.loginfo(f"⚡ 开始批量提取 {len(self.target_boxes)} 个目标的掩膜...")

            for idx, box in enumerate(self.target_boxes):
                x_min, y_min, x_max, y_max = box

                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    rospy.logwarn(f"⚠️ 目标框 {idx+1} 过小，跳过: {box}")
                    continue

                w = x_max - x_min
                h = y_max - y_min
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)

                inner_x_min = x_min + pad_x
                inner_x_max = x_max - pad_x
                inner_y_min = y_min + pad_y
                inner_y_max = y_max - pad_y

                pts_array = np.array([
                    [inner_x_min, inner_y_min],
                    [inner_x_max, inner_y_max],
                    [inner_x_min, inner_y_max],
                    [inner_x_max, inner_y_min],
                    [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]
                ])
                lbls_array = np.array([1, 1, 1, 1, 1])

                masks, scores, _ = self.predictor.predict(
                    point_coords=pts_array,
                    point_labels=lbls_array,
                    multimask_output=False,
                )

                total_pixels = self.curr_rgb.shape[0] * self.curr_rgb.shape[1]
                areas = [np.sum(m) for m in masks]
                valid_indices = []

                for i, area in enumerate(areas):
                    if area / total_pixels < 0.40:
                        valid_indices.append(i)

                if len(valid_indices) > 0:
                    best_idx = valid_indices[np.argmax([areas[i] for i in valid_indices])]
                else:
                    best_idx = int(np.argmax(scores))

                mask = masks[best_idx]
                mask = mask & (~combined_mask)
                if not np.any(mask):
                    rospy.logwarn(f"⚠️ 目标框 {idx+1} 与已有目标掩膜重叠过多，跳过独立建模")
                    continue

                object_masks.append(mask)
                combined_mask |= mask
                mask_y_indices, mask_x_indices = np.where(mask)
                object_metadata.append(
                    {
                        "object_id": len(object_masks) - 1,
                        "source_box_index": idx,
                        "source_bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "mask_bbox": [
                            int(np.min(mask_x_indices)),
                            int(np.min(mask_y_indices)),
                            int(np.max(mask_x_indices)),
                            int(np.max(mask_y_indices)),
                        ],
                    }
                )

                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(vis_img, (inner_x_min, inner_y_min), (inner_x_max, inner_y_max), (255, 0, 0), 1)

            if not np.any(combined_mask):
                rospy.logwarn("❌ 所有目标框均未能生成有效掩膜！")
                return

            rospy.loginfo("✅ 多目标掩膜融合完成！")
            torch.cuda.empty_cache()

            vis_img[combined_mask] = vis_img[combined_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
            self.sam_result_img = vis_img.astype(np.uint8)
            self.sam_mask_img = (combined_mask.astype(np.uint8) * 255)
            self.save_current_figures()

            if self.intrinsics is None:
                rospy.logwarn("⚠️ 尚未收到相机内参，跳过点云生成")
                return

            h_d, w_d = self.curr_depth.shape
            fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
            cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
            u, v = np.meshgrid(np.arange(w_d), np.arange(h_d))

            all_points = []
            all_colors_bgr = []
            all_object_ids = []

            for object_idx, object_mask in enumerate(object_masks):
                valid = object_mask & np.isfinite(self.curr_depth) & (self.curr_depth > 0.1) & (self.curr_depth < 2.0)
                z = self.curr_depth[valid]
                if len(z) == 0:
                    rospy.logwarn(f"⚠️ 目标 {object_idx+1} 无有效深度点，跳过")
                    continue

                front_depth = float(np.percentile(z, self.object_front_depth_percentile))
                depth_gate = self.curr_depth <= (front_depth + self.object_depth_gate_band)
                refined_valid = valid & depth_gate
                refined_z = self.curr_depth[refined_valid]
                if len(refined_z) >= max(80, int(0.2 * len(z))):
                    valid = refined_valid
                    z = refined_z

                source_bbox = None
                if object_idx < len(object_metadata):
                    source_bbox = object_metadata[object_idx].get("source_bbox")
                component_mask = self.pick_target_component(valid, source_bbox)
                component_z = self.curr_depth[component_mask]
                if len(component_z) >= self.object_component_min_pixels:
                    valid = component_mask
                    z = component_z

                x_3d = (u[valid] - cx) * z / fx
                y_3d = (v[valid] - cy) * z / fy
                points = np.stack((x_3d, y_3d, z), axis=-1)
                colors_bgr = self.curr_rgb[valid]

                if len(points) == 0:
                    continue

                all_points.append(points)
                all_colors_bgr.append(colors_bgr)
                all_object_ids.append(np.full(len(points), object_idx, dtype=np.uint32))

            if len(all_points) == 0:
                rospy.logwarn("❌ 所有目标均未生成有效点云")
                return

            points = np.concatenate(all_points, axis=0)
            colors_bgr = np.concatenate(all_colors_bgr, axis=0)
            object_ids = np.concatenate(all_object_ids, axis=0)
            colors_rgb = colors_bgr[:, ::-1] / 255.0

            debug_o3d = rospy.get_param("~debug_o3d", False)
            if debug_o3d:
                rospy.loginfo("👀 调试模式：打开 Open3D 检查【多目标点云】...")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries(
                    [pcd, frame],
                    window_name="Multi-Target Point Cloud Check"
                )

            if self.publish_background_collision_cloud or self.publish_local_obstacle_cloud:
                dilated_mask = self.build_collision_clearance_mask(object_masks, object_metadata)

                obstacle_valid = (
                    (~dilated_mask)
                    & np.isfinite(self.curr_depth)
                    & (self.curr_depth > 0.1)
                    & (self.curr_depth < 2.0)
                )
                obstacle_z = self.curr_depth[obstacle_valid]

                if len(obstacle_z) > 0:
                    obstacle_x_3d = (u[obstacle_valid] - cx) * obstacle_z / fx
                    obstacle_y_3d = (v[obstacle_valid] - cy) * obstacle_z / fy
                    bg_points = np.stack((obstacle_x_3d, obstacle_y_3d, obstacle_z), axis=-1).astype(np.float32)
                    bg_points = self.remove_points_near_objects(bg_points, all_points)

                    if self.publish_background_collision_cloud:
                        if len(bg_points) > 20000:
                            idxs = np.random.choice(len(bg_points), 20000, replace=False)
                            bg_points = bg_points[idxs]
                    elif self.publish_local_obstacle_cloud:
                        bg_points = self.filter_points_to_shelf_cavities(bg_points)
                        bg_points = self.remove_points_in_active_grasp_corridor(bg_points)
                        if len(bg_points) > self.local_obstacle_max_points:
                            idxs = np.random.choice(len(bg_points), self.local_obstacle_max_points, replace=False)
                            bg_points = bg_points[idxs]
                else:
                    bg_points = np.empty((0, 3), dtype=np.float32)
            else:
                bg_points = np.empty((0, 3), dtype=np.float32)

            if self.publish_background_collision_cloud:
                rospy.loginfo("🧱 当前发布完整背景点云碰撞。")
            elif self.publish_local_obstacle_cloud:
                rospy.loginfo("🧱 当前发布货架腔体内的局部动态障碍点云。")
            else:
                rospy.loginfo("🧱 当前使用固定货架几何，已跳过背景点云碰撞发布。")

            rospy.loginfo(f"✨ 点云处理成功！总目标点数: {len(points)}, 环境点数: {len(bg_points)}")

            self.publish_object_metadata(object_metadata)
            self.publish_point_cloud(points, colors_bgr, object_ids)
            self.publish_background_point_cloud(bg_points)

        except Exception as e:
            rospy.logerr(f"❌ process_segmentation 异常: {e}")

        finally:
            self.target_boxes = []
            if image_set:
                try:
                    self.predictor.reset_image()
                except Exception as e:
                    rospy.logwarn(f"⚠️ predictor.reset_image() 失败: {e}")
            torch.cuda.empty_cache()

    def save_current_figures(self):
        if not self.save_thesis_figures:
            return

        os.makedirs(self.figure_output_dir, exist_ok=True)
        self.figure_index += 1
        prefix = f"{self.figure_index:03d}"

        if self.curr_rgb is not None:
            cv2.imwrite(os.path.join(self.figure_output_dir, f"{prefix}_rgb_original.png"), self.curr_rgb)
        if self.vlm_bbox_img is not None:
            cv2.imwrite(os.path.join(self.figure_output_dir, f"{prefix}_vlm_bbox_overlay.png"), self.vlm_bbox_img)
        if self.sam_result_img is not None:
            cv2.imwrite(os.path.join(self.figure_output_dir, f"{prefix}_sam_overlay.png"), self.sam_result_img)
        if self.sam_mask_img is not None:
            cv2.imwrite(os.path.join(self.figure_output_dir, f"{prefix}_sam_mask_binary.png"), self.sam_mask_img)

    def build_full_scene_background(self):
        if self.curr_depth is None or self.intrinsics is None:
            return np.empty((0, 3), dtype=np.float32)

        h_d, w_d = self.curr_depth.shape
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        u, v = np.meshgrid(np.arange(w_d), np.arange(h_d))

        valid = (
            np.isfinite(self.curr_depth)
            & (self.curr_depth > self.background_min_depth)
            & (self.curr_depth < 2.0)
        )
        z = self.curr_depth[valid]
        if len(z) == 0:
            return np.empty((0, 3), dtype=np.float32)

        x_3d = (u[valid] - cx) * z / fx
        y_3d = (v[valid] - cy) * z / fy
        bg_points = np.stack((x_3d, y_3d, z), axis=-1).astype(np.float32)

        if len(bg_points) > 25000:
            idxs = np.random.choice(len(bg_points), 25000, replace=False)
            bg_points = bg_points[idxs]
        return bg_points

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
                rospy.logwarn(f"⚠️ SAM 查询货架位姿失败，改用 fallback: {e}")
        return dict(self.shelf_pose_fallback)

    def get_camera_to_world_matrix(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "world", "depth_camera_link", rospy.Time(0), rospy.Duration(0.5)
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
            rospy.logwarn(f"⚠️ SAM 获取 depth_camera_link -> world TF 失败: {e}")
            return None

    def get_shelf_inner_regions_local(self):
        mx = self.shelf_inner_margin_x
        my = self.shelf_inner_margin_y
        mz = self.shelf_inner_margin_z
        return [
            {
                "xmin": -0.15 + mx,
                "xmax": 0.15 - mx,
                "ymin": -0.30 + my,
                "ymax": 0.30 - my,
                "zmin": 0.22 + mz,
                "zmax": 0.40 - mz,
            },
            {
                "xmin": -0.15 + mx,
                "xmax": 0.15 - mx,
                "ymin": -0.30 + my,
                "ymax": 0.30 - my,
                "zmin": 0.42 + mz,
                "zmax": 0.60 - mz,
            },
            {
                "xmin": -0.15 + mx,
                "xmax": 0.15 - mx,
                "ymin": -0.30 + my,
                "ymax": 0.30 - my,
                "zmin": 0.62 + mz,
                "zmax": 0.79 - mz,
            },
        ]

    def filter_points_to_shelf_cavities(self, points_cam):
        if len(points_cam) == 0:
            return points_cam

        cam_to_world = self.get_camera_to_world_matrix()
        if cam_to_world is None:
            return np.empty((0, 3), dtype=np.float32)

        shelf_pose = self.get_shelf_world_pose()
        shelf_tf = tft.euler_matrix(0.0, 0.0, shelf_pose["yaw"])
        shelf_tf[:3, 3] = np.array([shelf_pose["x"], shelf_pose["y"], shelf_pose["z"]], dtype=np.float64)
        world_to_shelf = np.linalg.inv(shelf_tf)

        points_h = np.hstack([points_cam.astype(np.float64), np.ones((len(points_cam), 1), dtype=np.float64)])
        points_world = (cam_to_world @ points_h.T).T
        points_local = (world_to_shelf @ points_world.T).T[:, :3]

        keep_mask = np.zeros(len(points_local), dtype=bool)
        for region in self.get_shelf_inner_regions_local():
            inside = (
                (points_local[:, 0] >= region["xmin"]) & (points_local[:, 0] <= region["xmax"]) &
                (points_local[:, 1] >= region["ymin"]) & (points_local[:, 1] <= region["ymax"]) &
                (points_local[:, 2] >= region["zmin"]) & (points_local[:, 2] <= region["zmax"])
            )
            keep_mask |= inside

        return points_cam[keep_mask].astype(np.float32)

    def remove_points_in_active_grasp_corridor(self, points_cam):
        if len(points_cam) == 0 or not isinstance(self.active_grasp_target, dict):
            return points_cam

        radius = float(self.active_grasp_target.get("corridor_radius_m", 0.07))
        segments = self.active_grasp_target.get("corridor_segments_world")
        if not isinstance(segments, list) or len(segments) == 0:
            start = self.active_grasp_target.get("corridor_start_world")
            end = self.active_grasp_target.get("corridor_end_world")
            if (
                isinstance(start, (list, tuple)) and len(start) == 3
                and isinstance(end, (list, tuple)) and len(end) == 3
            ):
                segments = [{"start": start, "end": end}]
            else:
                return points_cam

        cam_to_world = self.get_camera_to_world_matrix()
        if cam_to_world is None:
            return points_cam

        points_h = np.hstack([points_cam.astype(np.float64), np.ones((len(points_cam), 1), dtype=np.float64)])
        points_world = (cam_to_world @ points_h.T).T[:, :3]

        keep_mask = np.ones(len(points_world), dtype=bool)
        for segment in segments:
            start = segment.get("start")
            end = segment.get("end")
            if not (
                isinstance(start, (list, tuple)) and len(start) == 3
                and isinstance(end, (list, tuple)) and len(end) == 3
            ):
                continue
            start = np.array(start, dtype=np.float64)
            end = np.array(end, dtype=np.float64)
            seg = end - start
            seg_norm_sq = float(np.dot(seg, seg))
            if seg_norm_sq < 1e-9:
                continue

            rel = points_world - start
            t = np.clip((rel @ seg) / seg_norm_sq, 0.0, 1.0)
            closest = start + np.outer(t, seg)
            dist = np.linalg.norm(points_world - closest, axis=1)
            keep_mask &= (dist > radius)

        return points_cam[keep_mask].astype(np.float32)

    def handle_refresh_background(self, _req):
        if not self.publish_background_collision_cloud and not self.publish_local_obstacle_cloud:
            self.publish_background_point_cloud(np.empty((0, 3), dtype=np.float32))
            rospy.loginfo("🔄 当前禁用背景点云碰撞，refresh_background 仅清空 octomap。")
            return EmptyResponse()
        bg_points = self.build_full_scene_background()
        if len(bg_points) == 0:
            rospy.logwarn("⚠️ 当前无法重建完整背景点云，跳过 refresh_background。")
            return EmptyResponse()
        if self.publish_local_obstacle_cloud and not self.publish_background_collision_cloud:
            bg_points = self.filter_points_to_shelf_cavities(bg_points)
            bg_points = self.remove_points_in_active_grasp_corridor(bg_points)
            if len(bg_points) > self.local_obstacle_max_points:
                idxs = np.random.choice(len(bg_points), self.local_obstacle_max_points, replace=False)
                bg_points = bg_points[idxs]
        self.publish_background_point_cloud(bg_points)
        rospy.loginfo("🔄 已按当前场景重新发布局部动态障碍点云，用于恢复货架碰撞空间。")
        return EmptyResponse()

    def publish_point_cloud(self, points, colors_bgr, object_ids):
        if len(points) == 0: return
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_camera_link" 
        
        colors_rgb = colors_bgr[:, ::-1]
        packed_colors = np.zeros(len(points), dtype=np.float32)
        for i in range(len(points)):
            r, g, b = colors_rgb[i]
            rgb = struct.unpack('f', struct.pack('i', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
            packed_colors[i] = rgb

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
            PointField('object_id', 16, PointField.UINT32, 1),
        ]
        rows = [
            (float(points[i, 0]), float(points[i, 1]), float(points[i, 2]), float(packed_colors[i]), int(object_ids[i]))
            for i in range(len(points))
        ]
        pc2_msg = pc2.create_cloud(header, fields, rows)
        self.cloud_pub.publish(pc2_msg)
        rospy.loginfo(f"☁️ 已发布多目标带 object_id 点云！目标数: {len(np.unique(object_ids))}")

    def publish_background_point_cloud(self, bg_points):
        try:
            rospy.wait_for_service('/clear_octomap', timeout=0.5)
            clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
            clear_octomap()
            rospy.loginfo("🧹 已刷新环境碰撞地图！")
        except Exception:
            pass

        if len(bg_points) == 0:
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_camera_link" 
        cloud_msg = pc2.create_cloud_xyz32(header, bg_points)
        self.bg_cloud_pub.publish(cloud_msg)

    def publish_object_metadata(self, object_metadata):
        payload = {
            "count": len(object_metadata),
            "objects": object_metadata,
        }
        self.metadata_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.loginfo("🗂️ 已发布 object metadata，目标数: %d", len(object_metadata))

    def remove_points_near_objects(self, bg_points, object_point_sets):
        if len(bg_points) == 0 or len(object_point_sets) == 0:
            return bg_points

        keep_mask = np.ones(len(bg_points), dtype=bool)
        for points in object_point_sets:
            if len(points) == 0:
                continue
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            mins[0] -= self.object_clearance_margin_x
            maxs[0] += self.object_clearance_margin_x
            mins[1] -= self.object_clearance_margin_y
            maxs[1] += self.object_clearance_margin_y
            mins[2] -= self.object_clearance_margin_z
            maxs[2] += self.object_clearance_margin_z

            inside = (
                (bg_points[:, 0] >= mins[0]) & (bg_points[:, 0] <= maxs[0]) &
                (bg_points[:, 1] >= mins[1]) & (bg_points[:, 1] <= maxs[1]) &
                (bg_points[:, 2] >= mins[2]) & (bg_points[:, 2] <= maxs[2])
            )
            keep_mask &= ~inside

        return bg_points[keep_mask]

    def pick_target_component(self, valid_mask, source_bbox=None):
        if not np.any(valid_mask):
            return valid_mask

        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(
            valid_mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 2:
            return valid_mask

        target_label = None
        if source_bbox is not None:
            xmin, ymin, xmax, ymax = [int(v) for v in source_bbox]
            cx = int((xmin + xmax) / 2.0)
            cy = int((ymin + ymax) / 2.0)
            if 0 <= cx < label_img.shape[1] and 0 <= cy < label_img.shape[0]:
                center_label = int(label_img[cy, cx])
                if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] >= self.object_component_min_pixels:
                    target_label = center_label

        if target_label is None:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) == 0:
                return valid_mask
            target_label = int(np.argmax(areas)) + 1

        refined_mask = label_img == target_label
        return refined_mask

    def build_collision_clearance_mask(self, object_masks, object_metadata):
        if len(object_masks) == 0:
            return np.zeros_like(self.curr_depth, dtype=bool)

        kernel_size = max(3, int(self.collision_clearance_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        clearance_mask = np.zeros_like(object_masks[0], dtype=bool)

        for object_idx, object_mask in enumerate(object_masks):
            local_mask = object_mask.copy()
            front_depth = None
            object_depth_values = self.curr_depth[object_mask & np.isfinite(self.curr_depth) & (self.curr_depth > 0.1)]
            if len(object_depth_values) > 0:
                front_depth = float(np.percentile(object_depth_values, self.object_front_depth_percentile))
            if object_idx < len(object_metadata):
                source_bbox = object_metadata[object_idx].get("source_bbox")
                if isinstance(source_bbox, (list, tuple)) and len(source_bbox) == 4:
                    xmin, ymin, xmax, ymax = [int(v) for v in source_bbox]
                    margin = self.collision_clearance_margin_px
                    xmin = max(0, xmin - margin)
                    ymin = max(0, ymin - margin)
                    xmax = min(local_mask.shape[1] - 1, xmax + margin)
                    ymax = min(local_mask.shape[0] - 1, ymax + margin)
                    local_mask[ymin:ymax + 1, xmin:xmax + 1] = True
                    if front_depth is not None:
                        depth_near_target = (
                            np.isfinite(self.curr_depth)
                            & (self.curr_depth > 0.1)
                            & (self.curr_depth <= (front_depth + self.collision_clearance_depth_band))
                        )
                        local_depth_clearance = np.zeros_like(local_mask, dtype=bool)
                        local_depth_clearance[ymin:ymax + 1, xmin:xmax + 1] = True
                        local_mask |= (local_depth_clearance & depth_near_target)

            dilated = cv2.dilate(local_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
            clearance_mask |= dilated

        return clearance_mask

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.curr_rgb is not None:
                vis_img = self.curr_rgb.copy()
                if self.drawing and self.box_start is not None:
                    pass 
                cv2.imshow("SAM Selection", vis_img)

            if self.vlm_bbox_img is not None:
                cv2.imshow("VLM BBoxes", self.vlm_bbox_img)
            
            if self.sam_result_img is not None:
                cv2.imshow("SAM Result", self.sam_result_img)

            if self.sam_mask_img is not None:
                cv2.imshow("SAM Mask", self.sam_mask_img)

            if self.should_process:
                self.process_segmentation()
                self.should_process = False

            key = cv2.waitKey(1)
            if key == 27: break
            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = SAMPerceptionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
