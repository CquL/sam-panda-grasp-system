#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# ==============================================================================
# 1. 修复 Numpy 版本冲突
# ==============================================================================
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import rospy
import numpy as np
import cv2
import struct
import os
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Float32MultiArray
from cv_bridge import CvBridge
from segment_anything import sam_model_registry, SamPredictor
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty, EmptyResponse

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

            mask_uint8 = (combined_mask.astype(np.uint8) * 255)
            kernel = np.ones((60, 60), np.uint8)
            dilated_mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=2)
            dilated_mask = dilated_mask_uint8 > 0

            bg_valid = (~dilated_mask) & np.isfinite(self.curr_depth) & (self.curr_depth > 0.1) & (self.curr_depth < 2.0)
            bg_z = self.curr_depth[bg_valid]

            if len(bg_z) > 0:
                bg_x_3d = (u[bg_valid] - cx) * bg_z / fx
                bg_y_3d = (v[bg_valid] - cy) * bg_z / fy
                bg_points = np.stack((bg_x_3d, bg_y_3d, bg_z), axis=-1)

                if len(bg_points) > 20000:
                    idxs = np.random.choice(len(bg_points), 20000, replace=False)
                    bg_points = bg_points[idxs]
            else:
                bg_points = np.empty((0, 3), dtype=np.float32)

            rospy.loginfo(f"✨ 点云处理成功！总目标点数: {len(points)}, 环境点数: {len(bg_points)}")

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

        valid = np.isfinite(self.curr_depth) & (self.curr_depth > 0.1) & (self.curr_depth < 2.0)
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

    def handle_refresh_background(self, _req):
        bg_points = self.build_full_scene_background()
        if len(bg_points) == 0:
            rospy.logwarn("⚠️ 当前无法重建完整背景点云，跳过 refresh_background。")
            return EmptyResponse()
        self.publish_background_point_cloud(bg_points)
        rospy.loginfo("🔄 已按当前场景重新发布背景点云，用于恢复货架碰撞空间。")
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
        if len(bg_points) == 0: return
        try:
            rospy.wait_for_service('/clear_octomap', timeout=0.5)
            clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
            clear_octomap()
            rospy.loginfo("🧹 已刷新环境碰撞地图！")
        except Exception:
            pass
            
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_camera_link" 
        cloud_msg = pc2.create_cloud_xyz32(header, bg_points)
        self.bg_cloud_pub.publish(cloud_msg)

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
