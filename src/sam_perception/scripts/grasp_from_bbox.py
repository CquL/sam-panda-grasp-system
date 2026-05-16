#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_from_bbox.py — VLM + GraspNet 消融实验节点（无 SAM）

与完整管线 (VLM → SAM → GraspNet) 的对比：
  - 不运行 SAM 分割，直接用 VLM 输出的 bbox 矩形区域作为「粗糙掩膜」
  - bbox 内的所有深度点（包括背景/邻近物体）都会进入点云
  - 点云噪声更大，用于验证 SAM 精细分割对抓取质量的贡献

上游：订阅 /sam/prompt_bbox（VLM bbox）、RGB-D 图像
下游：发布 /sam_perception/object_cloud（替代 SAM 节点输出）
"""

import os
import sys
import json
import struct
import time

ld_preload = os.environ.get("ANYGRASP_LD_PRELOAD", "").strip()
if ld_preload:
    os.environ["LD_PRELOAD"] = ld_preload
sys.path = [p for p in sys.path if "/usr/lib/python3/dist-packages" not in p]

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Float32MultiArray, String
from sam_perception.ros_image_compat import image_msg_to_numpy


class GraspFromBBoxNode:
    """用 VLM bbox 直接裁剪深度图生成点云，跳过 SAM 精细分割。"""

    def __init__(self):
        rospy.init_node("grasp_from_bbox")

        # --- 输入 topic ---
        self.bbox_topic = rospy.get_param("~bbox_topic", "/sam/prompt_bbox")
        self.global_target_topic = rospy.get_param("~global_target_topic", "/vlm/global_target")
        self.global_target_image_topic = rospy.get_param(
            "~global_target_image_topic", "/vlm/global_target_image"
        )
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        # --- 输出 topic（与 SAM 节点相同，保证下游 GraspNet 无需改动）---
        self.cloud_topic = rospy.get_param("~cloud_topic", "/sam_perception/object_cloud")
        self.metadata_topic = rospy.get_param("~metadata_topic", "/sam_perception/object_metadata")
        self.bg_cloud_topic = rospy.get_param(
            "~bg_cloud_topic", "/sam_perception/background_cloud"
        )

        # --- 点云参数 ---
        self.min_depth = float(rospy.get_param("~min_depth", 0.10))
        self.max_depth = float(rospy.get_param("~max_depth", 2.00))
        self.depth_front_percentile = float(rospy.get_param("~depth_front_percentile", 15.0))
        self.depth_gate_band = float(rospy.get_param("~depth_gate_band", 0.10))
        self.bbox_shrink_ratio = float(rospy.get_param("~bbox_shrink_ratio", 0.05))
        self.publish_background = bool(rospy.get_param("~publish_background", False))
        self.bg_max_points = int(rospy.get_param("~bg_max_points", 5000))
        self.save_debug = bool(rospy.get_param("~save_debug", True))
        self.debug_dir = os.path.expanduser(
            rospy.get_param("~debug_dir", "~/grasp_robot_ws/pointcloud_debug")
        )

        # --- 状态 ---
        self.latest_rgb = None
        self.latest_depth = None
        self.intrinsics = None
        self.latest_target = None
        self.latest_target_image = None
        self.pending_bbox = None
        self._bbox_seq = 0  # 用于去重

        # --- 订阅 ---
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber(self.bbox_topic, Float32MultiArray, self.bbox_cb, queue_size=1)
        rospy.Subscriber(self.global_target_topic, String, self.target_cb, queue_size=1)
        rospy.Subscriber(
            self.global_target_image_topic, Image, self.target_image_cb, queue_size=1
        )

        # --- 发布 ---
        self.cloud_pub = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1)
        self.metadata_pub = rospy.Publisher(self.metadata_topic, String, queue_size=1)
        self.bg_cloud_pub = rospy.Publisher(self.bg_cloud_topic, PointCloud2, queue_size=1)

        rospy.loginfo(
            "📦 grasp_from_bbox ready (NO SAM). bbox_topic=%s → cloud_topic=%s",
            self.bbox_topic,
            self.cloud_topic,
        )

    # ── 回调 ──────────────────────────────────────────────────────────

    def rgb_cb(self, msg):
        try:
            self.latest_rgb = image_msg_to_numpy(msg, "bgr8")
        except Exception:
            self.latest_rgb = None

    def depth_cb(self, msg):
        try:
            self.latest_depth = image_msg_to_numpy(msg, "32FC1")
        except Exception:
            self.latest_depth = None

    def info_cb(self, msg):
        self.intrinsics = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def target_cb(self, msg):
        try:
            self.latest_target = json.loads(msg.data)
        except Exception:
            self.latest_target = None

    def target_image_cb(self, msg):
        try:
            self.latest_target_image = image_msg_to_numpy(msg, "bgr8")
        except Exception:
            self.latest_target_image = None

    def bbox_cb(self, msg):
        data = list(msg.data)
        if len(data) == 0 or len(data) % 4 != 0:
            return
        num_boxes = len(data) // 4
        boxes = []
        for i in range(num_boxes):
            xmin, ymin, xmax, ymax = [int(v) for v in data[i * 4 : i * 4 + 4]]
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

        if not boxes:
            return

        # 去重：相同 bbox 序列不重复处理
        bbox_hash = hash(tuple(tuple(b) for b in boxes))
        if bbox_hash == self._bbox_seq:
            return
        self._bbox_seq = bbox_hash

        rospy.loginfo(f"📥 grasp_from_bbox 收到 {len(boxes)} 个 VLM bbox，开始生成点云（无 SAM）")
        self.process_bboxes(boxes)

    # ── 核心逻辑 ──────────────────────────────────────────────────────

    def process_bboxes(self, boxes):
        if self.latest_depth is None or self.intrinsics is None:
            rospy.logwarn("grasp_from_bbox: depth 或 intrinsics 未就绪，跳过")
            return

        ref_image = (
            self.latest_target_image
            if self.latest_target_image is not None
            else self.latest_rgb
        )

        h_d, w_d = self.latest_depth.shape
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        u_grid, v_grid = np.meshgrid(np.arange(w_d), np.arange(h_d))

        # 全局有效深度
        global_valid = (
            np.isfinite(self.latest_depth)
            & (self.latest_depth > self.min_depth)
            & (self.latest_depth < self.max_depth)
        )

        all_points = []
        all_object_ids = []
        object_metadata = []
        combined_mask = np.zeros((h_d, w_d), dtype=bool)

        for obj_idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            # 稍微收缩 bbox 减少边缘背景
            bw = xmax - xmin
            bh = ymax - ymin
            shrink_x = int(bw * self.bbox_shrink_ratio)
            shrink_y = int(bh * self.bbox_shrink_ratio)
            sxmin = max(0, xmin + shrink_x)
            symin = max(0, ymin + shrink_y)
            sxmax = min(w_d - 1, xmax - shrink_x)
            symax = min(h_d - 1, ymax - shrink_y)

            if sxmax <= sxmin or symax <= symin:
                rospy.logwarn(f"grasp_from_bbox: bbox {obj_idx} 收缩后无效，跳过")
                continue

            # bbox 矩形区域作为粗糙掩膜
            bbox_mask = np.zeros((h_d, w_d), dtype=bool)
            bbox_mask[symin : symax + 1, sxmin : sxmax + 1] = True

            # 与全局有效深度取交集，排除已分配给前面目标的区域
            valid = bbox_mask & global_valid & (~combined_mask)
            z = self.latest_depth[valid]

            if len(z) < 50:
                rospy.logwarn(f"grasp_from_bbox: bbox {obj_idx} 有效深度点不足 ({len(z)})，跳过")
                continue

            # depth gate: 过滤远离前景的点
            front_depth = float(np.percentile(z, self.depth_front_percentile))
            depth_gate = self.latest_depth <= (front_depth + self.depth_gate_band)
            refined_valid = valid & depth_gate
            if np.count_nonzero(refined_valid) >= max(50, int(0.2 * len(z))):
                valid = refined_valid
                z = self.latest_depth[valid]

            # 转 3D
            x_3d = (u_grid[valid] - cx) * z / fx
            y_3d = (v_grid[valid] - cy) * z / fy
            points = np.stack((x_3d, y_3d, z), axis=-1).astype(np.float32)

            if len(points) < 50:
                continue

            all_points.append(points)
            all_object_ids.append(np.full(len(points), obj_idx, dtype=np.uint32))
            combined_mask |= valid

            mask_y, mask_x = np.where(valid)
            object_metadata.append(
                {
                    "object_id": obj_idx,
                    "source_box_index": obj_idx,
                    "source_bbox": [xmin, ymin, xmax, ymax],
                    "mask_bbox": [
                        int(np.min(mask_x)),
                        int(np.min(mask_y)),
                        int(np.max(mask_x)),
                        int(np.max(mask_y)),
                    ],
                }
            )

            rospy.loginfo(
                f"  ↳ bbox {obj_idx}: [{xmin},{ymin},{xmax},{ymax}] → {len(points)} 点"
            )

        if len(all_points) == 0:
            rospy.logwarn("grasp_from_bbox: 所有 bbox 均未生成有效点云")
            # 仍发布空 metadata 以告知下游无目标
            self.publish_object_metadata([])
            return

        points = np.concatenate(all_points, axis=0)
        object_ids = np.concatenate(all_object_ids, axis=0)

        # 背景点云
        bg_points = np.empty((0, 3), dtype=np.float32)
        if self.publish_background:
            bg_valid = global_valid & (~combined_mask)
            bg_z = self.latest_depth[bg_valid]
            if len(bg_z) > 0:
                bg_x = (u_grid[bg_valid] - cx) * bg_z / fx
                bg_y = (v_grid[bg_valid] - cy) * bg_z / fy
                bg_points = np.stack((bg_x, bg_y, bg_z), axis=-1).astype(np.float32)
                if len(bg_points) > self.bg_max_points:
                    idxs = np.random.choice(len(bg_points), self.bg_max_points, replace=False)
                    bg_points = bg_points[idxs]

        # 发布
        self.publish_object_metadata(object_metadata)
        self.publish_point_cloud(points, object_ids)
        self.publish_background_cloud(bg_points)

        # 保存调试图像（bbox overlay）
        if self.save_debug and ref_image is not None:
            self._save_debug_image(ref_image, boxes, points)

        rospy.loginfo(
            f"✅ grasp_from_bbox (no SAM) 已发布: {len(all_points)} 目标, "
            f"{len(points)} 总点, {len(bg_points)} 背景点"
        )

    # ── 发布函数 ──────────────────────────────────────────────────────

    def publish_point_cloud(self, points, object_ids):
        if len(points) == 0:
            return
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_camera_link"

        # 白色点（无 RGB 信息时使用）
        packed_color = struct.unpack(
            "f", struct.pack("i", (255 << 16) | (255 << 8) | 255)
        )[0]

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
            PointField("object_id", 16, PointField.UINT32, 1),
        ]
        rows = [
            (
                float(points[i, 0]),
                float(points[i, 1]),
                float(points[i, 2]),
                float(packed_color),
                int(object_ids[i]),
            )
            for i in range(len(points))
        ]
        pc2_msg = pc2.create_cloud(header, fields, rows)
        self.cloud_pub.publish(pc2_msg)

    def publish_object_metadata(self, object_metadata):
        payload = {"count": len(object_metadata), "objects": object_metadata}
        self.metadata_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_background_cloud(self, bg_points):
        if len(bg_points) == 0:
            return
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "depth_camera_link"
        cloud_msg = pc2.create_cloud_xyz32(header, bg_points)
        self.bg_cloud_pub.publish(cloud_msg)

    def _save_debug_image(self, ref_image, boxes, points):
        os.makedirs(self.debug_dir, exist_ok=True)
        vis = ref_image.copy()
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            color = (0, 255, 255) if i == 0 else (0, 200, 0)
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                vis,
                f"bbox-{i} ({len(points)}pts)",
                (xmin, max(ymin - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(
            os.path.join(self.debug_dir, f"bbox_nosam_{timestamp}.png"), vis
        )


if __name__ == "__main__":
    try:
        node = GraspFromBBoxNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
