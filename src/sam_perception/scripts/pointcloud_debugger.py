#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import struct
import threading
import time
from functools import partial

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from ros_image_compat import numpy_to_image_msg

try:
    import open3d as o3d
except ImportError:
    o3d = None


def decode_rgb_field(value):
    if isinstance(value, (float, np.floating)):
        packed = struct.pack("f", float(value))
        rgb_uint = struct.unpack("I", packed)[0]
    else:
        rgb_uint = int(value)
    return np.array(
        [
            (rgb_uint >> 16) & 0xFF,
            (rgb_uint >> 8) & 0xFF,
            rgb_uint & 0xFF,
        ],
        dtype=np.uint8,
    )


def id_to_rgb(object_ids):
    if len(object_ids) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    palette = np.array(
        [
            [255, 99, 71],
            [135, 206, 250],
            [255, 215, 0],
            [144, 238, 144],
            [255, 182, 193],
            [221, 160, 221],
            [64, 224, 208],
            [255, 160, 122],
        ],
        dtype=np.uint8,
    )
    idx = np.abs(object_ids.astype(np.int64)) % len(palette)
    return palette[idx]


class PointCloudDebugger:
    def __init__(self):
        rospy.init_node("pointcloud_debugger")

        legacy_show_windows = bool(rospy.get_param("~show_windows", False))
        self.show_projection_windows = bool(
            rospy.get_param("~show_projection_windows", legacy_show_windows)
        )
        self.show_o3d_windows = bool(rospy.get_param("~show_o3d_windows", True))
        self.publish_projection_images = bool(
            rospy.get_param("~publish_projection_images", True)
        )
        self.save_snapshots = bool(rospy.get_param("~save_snapshots", True))
        self.save_interval_sec = float(rospy.get_param("~save_interval_sec", 4.0))
        self.max_points = max(500, int(rospy.get_param("~max_points", 30000)))
        self.image_size = int(rospy.get_param("~image_size", 720))
        self.o3d_point_size = float(rospy.get_param("~o3d_point_size", 6.0))
        self.o3d_voxel_size = float(rospy.get_param("~o3d_voxel_size", 0.0))
        self.o3d_white_background = bool(rospy.get_param("~o3d_white_background", True))
        self.show_raw_o3d = bool(rospy.get_param("~show_raw_o3d", False))
        self.show_object_overview_o3d = bool(
            rospy.get_param("~show_object_overview_o3d", True)
        )
        self.show_background_o3d = bool(rospy.get_param("~show_background_o3d", False))
        self.show_bounding_box = bool(rospy.get_param("~show_bounding_box", False))
        self.show_coordinate_frame = bool(rospy.get_param("~show_coordinate_frame", False))
        self.show_raw_grasps = bool(rospy.get_param("~show_raw_grasps", False))
        self.show_corrected_grasps = bool(rospy.get_param("~show_corrected_grasps", False))
        self.show_output_grasps = bool(rospy.get_param("~show_output_grasps", True))
        self.display_grasp_top_n = max(1, int(rospy.get_param("~display_grasp_top_n", 6)))
        self.overview_grasp_top_n = max(1, int(rospy.get_param("~overview_grasp_top_n", 4)))
        self.focus_object_id = int(rospy.get_param("~focus_object_id", -1))
        self.focus_selection_mode = str(rospy.get_param("~focus_selection_mode", "largest")).strip().lower()
        self.show_focus_object = bool(rospy.get_param("~show_focus_object", True))
        self.overview_use_rgb_texture = bool(
            rospy.get_param("~overview_use_rgb_texture", True)
        )
        self.show_focus_pure_object = bool(
            rospy.get_param("~show_focus_pure_object", self.show_focus_object)
        )
        self.show_focus_overlay_object = bool(
            rospy.get_param("~show_focus_overlay_object", self.show_focus_object)
        )
        self.show_focus_pure_o3d = bool(
            rospy.get_param("~show_focus_pure_o3d", self.show_focus_pure_object)
        )
        self.show_focus_overlay_o3d = bool(
            rospy.get_param("~show_focus_overlay_o3d", self.show_focus_overlay_object)
        )
        self.show_all_object_focus_windows = bool(
            rospy.get_param("~show_all_object_focus_windows", False)
        )
        self.show_all_object_focus_pure_o3d = bool(
            rospy.get_param(
                "~show_all_object_focus_pure_o3d",
                self.show_all_object_focus_windows and self.show_focus_pure_object,
            )
        )
        self.show_all_object_focus_overlay_o3d = bool(
            rospy.get_param(
                "~show_all_object_focus_overlay_o3d",
                self.show_all_object_focus_windows and self.show_focus_overlay_object,
            )
        )
        self.all_object_detail_max_windows = max(
            1, int(rospy.get_param("~all_object_detail_max_windows", 6))
        )
        self.focus_use_rgb_texture = bool(rospy.get_param("~focus_use_rgb_texture", True))
        self.object_solid_color = bool(rospy.get_param("~object_solid_color", True))
        self.object_remove_outliers = bool(rospy.get_param("~object_remove_outliers", True))
        self.object_outlier_nb_neighbors = max(5, int(rospy.get_param("~object_outlier_nb_neighbors", 20)))
        self.object_outlier_std_ratio = float(rospy.get_param("~object_outlier_std_ratio", 1.2))
        self.grasp_mesh_radius = float(rospy.get_param("~grasp_mesh_radius", 0.0031))
        self.grasp_center_radius = float(rospy.get_param("~grasp_center_radius", 0.0062))
        self.output_dir = os.path.expanduser(
            rospy.get_param("~output_dir", "~/grasp_robot_ws/pointcloud_debug")
        )
        self.focus_pure_projection_topic = rospy.get_param(
            "~focus_pure_projection_topic", "/pointcloud_debug/object_focus_pure_projection"
        )
        self.focus_overlay_projection_topic = rospy.get_param(
            "~focus_overlay_projection_topic", "/pointcloud_debug/object_focus_overlay_projection"
        )
        self.focus_pure_window_name = rospy.get_param(
            "~focus_pure_window_name", "PointCloud Object Focus Pure"
        )
        self.focus_overlay_window_name = rospy.get_param(
            "~focus_overlay_window_name", "PointCloud Object Focus Overlay"
        )
        self.focus_pure_label = rospy.get_param(
            "~focus_pure_label", "SAM Focus Object Cloud"
        )
        self.focus_overlay_label = rospy.get_param(
            "~focus_overlay_label", "SAM Focus Object + Grasp"
        )

        if (self.show_projection_windows or self.show_o3d_windows) and not os.environ.get("DISPLAY"):
            rospy.logwarn("⚠️ 未检测到 DISPLAY，关闭本地点云窗口显示，仅保留消息发布与文件保存。")
            self.show_projection_windows = False
            self.show_o3d_windows = False
        if self.show_o3d_windows and o3d is None:
            rospy.logwarn("⚠️ 当前环境未安装 open3d，回退到投影窗口模式。")
            self.show_o3d_windows = False

        os.makedirs(self.output_dir, exist_ok=True)
        self.stats_pub = rospy.Publisher(
            "/pointcloud_debug/stats", String, queue_size=10
        )
        self.received_counts = {}
        self.last_message_at = {}
        self.current_focus_object_id = None
        self.paper_figure_enabled = True
        self.o3d_window_keys = set()
        self.dynamic_window_specs = {}
        self.dynamic_o3d_items = {}
        self.dynamic_o3d_dirty = {}
        if self.show_raw_o3d:
            self.o3d_window_keys.add("raw")
        if self.show_object_overview_o3d:
            self.o3d_window_keys.add("object")
        if self.show_background_o3d:
            self.o3d_window_keys.add("background")
        if self.show_focus_pure_o3d and self.show_focus_pure_object:
            self.o3d_window_keys.add("object_focus_pure")
        if self.show_focus_overlay_o3d and self.show_focus_overlay_object:
            self.o3d_window_keys.add("object_focus_overlay")

        self.topic_specs = {
            "raw": {
                "topic": rospy.get_param("~raw_cloud_topic", "/camera/depth/points"),
                "label": "Raw Camera Cloud",
                "default_color_rgb": np.array([200, 220, 255], dtype=np.uint8),
                "image_topic": "/pointcloud_debug/raw_projection",
                "window": "PointCloud Raw",
            },
            "object": {
                "topic": rospy.get_param(
                    "~object_cloud_topic", "/sam_perception/object_cloud"
                ),
                "label": "SAM Objects Overview",
                "default_color_rgb": np.array([255, 180, 80], dtype=np.uint8),
                "image_topic": "/pointcloud_debug/object_projection",
                "window": "PointCloud Objects Overview",
            },
            "background": {
                "topic": rospy.get_param(
                    "~background_cloud_topic", "/sam_perception/background_cloud"
                ),
                "label": "Background Cloud",
                "default_color_rgb": np.array([120, 255, 160], dtype=np.uint8),
                "image_topic": "/pointcloud_debug/background_projection",
                "window": "PointCloud Background",
            },
        }

        self.image_pubs = {}
        self.last_saved_at = {key: 0.0 for key in self.topic_specs}
        if self.show_focus_pure_object:
            self.last_saved_at["object_focus_pure"] = 0.0
        if self.show_focus_overlay_object:
            self.last_saved_at["object_focus_overlay"] = 0.0
        self.received_counts = {key: 0 for key in self.topic_specs}
        self.last_message_at = {key: None for key in self.topic_specs}
        self.o3d_lock = threading.Lock()
        self.o3d_items = {key: None for key in self.topic_specs}
        self.o3d_dirty = {key: False for key in self.topic_specs}
        if self.show_focus_pure_object:
            self.o3d_items["object_focus_pure"] = None
            self.o3d_dirty["object_focus_pure"] = False
        if self.show_focus_overlay_object:
            self.o3d_items["object_focus_overlay"] = None
            self.o3d_dirty["object_focus_overlay"] = False
        self.grasp_line_items = {
            "raw": None,
            "corrected": None,
            "output_overview": None,
            "output_focus": None,
            "output_per_object": {},
        }
        self.grasp_dirty = False
        self.o3d_thread = None
        self.latest_grasp_pose_msg = None
        self.latest_grasp_info = None

        if self.show_projection_windows:
            for spec in self.topic_specs.values():
                cv2.namedWindow(spec["window"], cv2.WINDOW_NORMAL)
            if self.show_focus_pure_object:
                cv2.namedWindow(self.focus_pure_window_name, cv2.WINDOW_NORMAL)
            if self.show_focus_overlay_object:
                cv2.namedWindow(self.focus_overlay_window_name, cv2.WINDOW_NORMAL)
        if self.show_o3d_windows and self.o3d_window_keys:
            self.o3d_thread = threading.Thread(
                target=self.o3d_visualization_loop, daemon=True
            )
            self.o3d_thread.start()

        for key, spec in self.topic_specs.items():
            if self.publish_projection_images:
                self.image_pubs[key] = rospy.Publisher(
                    spec["image_topic"], Image,
                    queue_size=1,
                )
        if self.publish_projection_images and self.show_focus_pure_object:
            self.image_pubs["object_focus_pure"] = rospy.Publisher(
                self.focus_pure_projection_topic,
                Image,
                queue_size=1,
            )
        if self.publish_projection_images and self.show_focus_overlay_object:
            self.image_pubs["object_focus_overlay"] = rospy.Publisher(
                self.focus_overlay_projection_topic,
                Image,
                queue_size=1,
            )
        for key, spec in self.topic_specs.items():
            rospy.Subscriber(
                spec["topic"],
                PointCloud2,
                partial(self.cloud_callback, key),
                queue_size=1,
            )
        rospy.Subscriber(
            rospy.get_param("~raw_grasp_marker_topic", "/graspnet/raw_grasp_markers"),
            MarkerArray,
            partial(self.marker_callback, "raw"),
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~corrected_grasp_marker_topic", "/graspnet/corrected_grasp_markers"),
            MarkerArray,
            partial(self.marker_callback, "corrected"),
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~grasp_pose_topic", "/graspnet/grasp_pose_array_raw"),
            PoseArray,
            self.grasp_pose_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~grasp_info_topic", "/graspnet/grasp_info_raw"),
            Float32MultiArray,
            self.grasp_info_callback,
            queue_size=1,
        )
        self.health_timer = rospy.Timer(
            rospy.Duration(5.0), self.report_missing_topics
        )

        rospy.loginfo(
            "☁️ 点云调试器已启动。订阅 topics: raw=%s object=%s background=%s, 输出目录=%s, "
            "projection=%s, o3d=%s, o3d_keys=%s, focus_pure=%s, focus_overlay=%s, "
            "all_object_windows=%s, all_object_limit=%d",
            self.topic_specs["raw"]["topic"],
            self.topic_specs["object"]["topic"],
            self.topic_specs["background"]["topic"],
            self.output_dir,
            self.show_projection_windows,
            self.show_o3d_windows,
            sorted(self.o3d_window_keys),
            self.show_focus_pure_object,
            self.show_focus_overlay_object,
            self.show_all_object_focus_windows,
            self.all_object_detail_max_windows,
        )

    def report_missing_topics(self, _event):
        now = rospy.Time.now().to_sec()
        for key, spec in self.topic_specs.items():
            idle_warn_sec = 10.0 if key == "raw" else 90.0
            last_at = self.last_message_at.get(key)
            if last_at is None:
                rospy.logwarn_throttle(
                    idle_warn_sec,
                    "⚠️ 点云调试器尚未收到 %s 数据: %s",
                    spec["label"],
                    spec["topic"],
                )
                continue
            idle_sec = float(now - last_at)
            if idle_sec > idle_warn_sec:
                rospy.logwarn_throttle(
                    idle_warn_sec,
                    "⚠️ %s 已 %.1f s 未更新: %s",
                    spec["label"],
                    idle_sec,
                    spec["topic"],
                )

    def color_for_object_id(self, object_id):
        palette = np.array(
            [
                [84, 160, 255],
                [98, 201, 112],
                [255, 153, 102],
                [166, 129, 255],
                [255, 120, 150],
                [120, 206, 214],
            ],
            dtype=np.uint8,
        )
        idx = int(abs(int(object_id)) % len(palette))
        return palette[idx]

    def grasp_color_for_object_id(self, object_id):
        palette = np.array(
            [
                [0.04, 0.39, 0.98],
                [0.98, 0.25, 0.14],
                [0.00, 0.72, 0.29],
                [0.98, 0.63, 0.06],
                [0.63, 0.19, 0.98],
                [0.00, 0.75, 0.82],
            ],
            dtype=np.float64,
        )
        idx = int(abs(int(object_id)) % len(palette))
        return palette[idx]

    def recolor_points_by_object_id(self, object_ids):
        if object_ids is None or len(object_ids) == 0:
            return np.empty((0, 3), dtype=np.uint8)
        colors = np.zeros((len(object_ids), 3), dtype=np.uint8)
        for object_id in np.unique(object_ids):
            colors[object_ids == object_id] = self.color_for_object_id(object_id)
        return colors

    def select_focus_object_id(self, object_ids):
        if object_ids is None or len(object_ids) == 0:
            return None
        unique_ids, counts = np.unique(object_ids, return_counts=True)
        if self.focus_object_id >= 0:
            if self.focus_object_id in set(unique_ids.tolist()):
                return int(self.focus_object_id)
            rospy.logwarn_throttle(
                5.0,
                "⚠️ 请求的 focus_object_id=%d 不在当前点云里，可用 ids=%s",
                int(self.focus_object_id),
                unique_ids.tolist(),
            )
        if self.focus_selection_mode == "first":
            return int(np.min(unique_ids))
        return int(unique_ids[np.argmax(counts)])

    def select_detail_object_ids(self, object_ids):
        if object_ids is None or len(object_ids) == 0:
            return []
        unique_ids = sorted(np.unique(object_ids).astype(np.int32).tolist())
        return unique_ids[: self.all_object_detail_max_windows]

    def dynamic_object_key(self, object_id, mode):
        return f"object_multi_{mode}_{int(object_id)}"

    def dynamic_object_label(self, object_id, mode):
        if mode == "pure":
            return f"SAM Object {int(object_id)} Cloud"
        return f"SAM Object {int(object_id)} + Grasp"

    def parse_grasp_groups(self, pose_msg, info_data):
        poses = list(getattr(pose_msg, "poses", []))
        if not poses:
            return None
        info = list(getattr(info_data, "data", []))
        if not info:
            return None
        num_poses = len(poses)
        if num_poses == 0:
            return None
        stride = len(info) // num_poses if num_poses > 0 else 0
        if stride < 4 or stride * num_poses != len(info):
            rospy.logwarn_throttle(
                5.0,
                "⚠️ 无法解析 grasp_info_raw：len(info)=%d num_poses=%d stride=%d",
                len(info),
                num_poses,
                stride,
            )
            return None

        per_object = {}
        for idx, pose in enumerate(poses):
            base = idx * stride
            object_id = int(info[base])
            width = float(info[base + 1])
            score = float(info[base + 2])
            depth = float(info[base + 3])
            per_object.setdefault(object_id, []).append(
                {
                    "pose": pose,
                    "width": width,
                    "score": score,
                    "depth": depth,
                }
            )
        return per_object

    def refine_object_cloud(self, points, colors_rgb):
        if len(points) == 0:
            return points, colors_rgb
        if not self.object_remove_outliers or o3d is None or len(points) < 32:
            return points, colors_rgb
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if len(colors_rgb) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64) / 255.0)
            _, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.object_outlier_nb_neighbors,
                std_ratio=self.object_outlier_std_ratio,
            )
            if not inlier_indices:
                return points, colors_rgb
            inlier_indices = np.asarray(inlier_indices, dtype=np.int32)
            return points[inlier_indices], colors_rgb[inlier_indices]
        except Exception:
            return points, colors_rgb

    def local_gripper_points(self, width=0.06, depth=0.06):
        half_w = max(0.005, float(width) * 0.5)
        finger_len = min(0.05, max(0.025, float(depth) + 0.01))
        palm_back = 0.02
        palm_height = 0.02
        return np.array(
            [
                [-half_w, 0.0, -palm_back],
                [half_w, 0.0, -palm_back],
                [-half_w, 0.0, 0.0],
                [half_w, 0.0, 0.0],
                [-half_w, 0.0, finger_len],
                [half_w, 0.0, finger_len],
                [0.0, palm_height, -palm_back],
                [0.0, palm_height, 0.0],
            ],
            dtype=np.float64,
        )

    def append_grasp_items_to_lines(self, grasp_items, color, all_points, all_lines, all_colors, point_offset):
        line_pairs = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (0, 6),
            (1, 6),
            (6, 7),
            (2, 7),
            (3, 7),
        ]
        for item in grasp_items:
            q = item["pose"].orientation
            transform = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            rotation = transform[:3, :3]
            translation = np.array(
                [
                    item["pose"].position.x,
                    item["pose"].position.y,
                    item["pose"].position.z,
                ],
                dtype=np.float64,
            )
            local_points = self.local_gripper_points(item["width"], item["depth"])
            world_points = (rotation @ local_points.T).T + translation
            all_points.extend(world_points.tolist())
            for a, b in line_pairs:
                all_lines.append([point_offset + a, point_offset + b])
                all_colors.append(color.tolist())
            point_offset += len(local_points)
        return point_offset

    def build_grasp_lines_for_items(self, grasp_items, color, focus_object_id=None, render_style="line"):
        if not grasp_items:
            return None
        all_points = []
        all_lines = []
        all_colors = []
        grasp_centers = []
        point_offset = 0
        line_pairs = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (0, 6),
            (1, 6),
            (6, 7),
            (2, 7),
            (3, 7),
        ]
        color_arr = np.asarray(color, dtype=np.float64)
        for item in grasp_items:
            q = item["pose"].orientation
            transform = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            rotation = transform[:3, :3]
            translation = np.array(
                [
                    item["pose"].position.x,
                    item["pose"].position.y,
                    item["pose"].position.z,
                ],
                dtype=np.float64,
            )
            local_points = self.local_gripper_points(item["width"], item["depth"])
            world_points = (rotation @ local_points.T).T + translation
            grasp_centers.append(translation.tolist())
            all_points.extend(world_points.tolist())
            for a, b in line_pairs:
                all_lines.append([point_offset + a, point_offset + b])
                all_colors.append(color_arr.tolist())
            point_offset += len(local_points)
        if not all_points:
            return None
        payload = {
            "points": np.asarray(all_points, dtype=np.float64),
            "lines": np.asarray(all_lines, dtype=np.int32),
            "colors": np.asarray(all_colors, dtype=np.float64),
            "centers": np.asarray(grasp_centers, dtype=np.float64) if grasp_centers else np.empty((0, 3), dtype=np.float64),
            "render_style": str(render_style),
        }
        if focus_object_id is not None:
            payload["focus_object_id"] = int(focus_object_id)
        return payload

    def build_overview_output_grasp_lines(self, per_object):
        if not per_object:
            return None
        all_points = []
        all_lines = []
        all_colors = []
        point_offset = 0
        for object_id in sorted(per_object.keys()):
            grasp_items = sorted(
                per_object.get(int(object_id), []),
                key=lambda item: -item["score"],
            )[: self.overview_grasp_top_n]
            if not grasp_items:
                continue
            point_offset = self.append_grasp_items_to_lines(
                grasp_items,
                self.grasp_color_for_object_id(object_id),
                all_points,
                all_lines,
                all_colors,
                point_offset,
            )
        if not all_points:
            return None
        return {
            "points": np.asarray(all_points, dtype=np.float64),
            "lines": np.asarray(all_lines, dtype=np.int32),
            "colors": np.asarray(all_colors, dtype=np.float64),
        }

    def refresh_output_grasps(self):
        if self.latest_grasp_pose_msg is None or self.latest_grasp_info is None:
            return
        per_object = self.parse_grasp_groups(
            self.latest_grasp_pose_msg,
            self.latest_grasp_info,
        )
        if not per_object:
            return
        available_ids = np.array(sorted(per_object.keys()), dtype=np.int32)
        if (
            self.current_focus_object_id is not None
            and int(self.current_focus_object_id) in per_object
        ):
            focus_object_id = int(self.current_focus_object_id)
        else:
            focus_object_id = self.select_focus_object_id(available_ids)

        focus_payload = None
        per_object_payloads = {}
        if focus_object_id is not None and int(focus_object_id) in per_object:
            focus_items = sorted(
                per_object.get(int(focus_object_id), []),
                key=lambda item: -item["score"],
            )[: self.display_grasp_top_n]
            focus_payload = self.build_grasp_lines_for_items(
                focus_items,
                self.grasp_color_for_object_id(focus_object_id),
                focus_object_id=focus_object_id,
                render_style="mesh",
            )
        for object_id in sorted(per_object.keys()):
            object_items = sorted(
                per_object.get(int(object_id), []),
                key=lambda item: -item["score"],
            )[: self.display_grasp_top_n]
            if not object_items:
                continue
            payload = self.build_grasp_lines_for_items(
                object_items,
                self.grasp_color_for_object_id(object_id),
                focus_object_id=object_id,
                render_style="mesh",
            )
            if payload:
                per_object_payloads[int(object_id)] = payload
        overview_payload = self.build_overview_output_grasp_lines(per_object)
        with self.o3d_lock:
            self.grasp_line_items["output_overview"] = overview_payload
            self.grasp_line_items["output_focus"] = focus_payload
            self.grasp_line_items["output_per_object"] = per_object_payloads
            self.grasp_dirty = True

    def grasp_pose_callback(self, msg):
        self.latest_grasp_pose_msg = msg
        self.refresh_output_grasps()

    def grasp_info_callback(self, msg):
        self.latest_grasp_info = msg
        self.refresh_output_grasps()

    def pointcloud_to_arrays(self, msg, default_color_rgb):
        available_fields = [field.name for field in msg.fields]
        if not all(name in available_fields for name in ("x", "y", "z")):
            raise ValueError(f"PointCloud2 缺少 xyz 字段: {available_fields}")

        selected_fields = ["x", "y", "z"]
        has_rgb = "rgb" in available_fields
        has_rgba = "rgba" in available_fields
        has_object_id = "object_id" in available_fields
        if has_rgb:
            selected_fields.append("rgb")
        elif has_rgba:
            selected_fields.append("rgba")
        if has_object_id:
            selected_fields.append("object_id")

        rows = list(pc2.read_points(msg, field_names=selected_fields, skip_nans=True))
        if not rows:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )

        if len(rows) > self.max_points:
            step = int(np.ceil(float(len(rows)) / float(self.max_points)))
            rows = rows[::step]

        field_idx = {name: idx for idx, name in enumerate(selected_fields)}
        points = np.array(
            [[row[field_idx["x"]], row[field_idx["y"]], row[field_idx["z"]]] for row in rows],
            dtype=np.float32,
        )

        object_ids = np.zeros((len(rows),), dtype=np.int32)
        if has_object_id:
            object_ids = np.array(
                [int(row[field_idx["object_id"]]) for row in rows], dtype=np.int32
            )

        if has_rgb or has_rgba:
            rgb_field = "rgb" if has_rgb else "rgba"
            colors_rgb = np.array(
                [decode_rgb_field(row[field_idx[rgb_field]]) for row in rows],
                dtype=np.uint8,
            )
        elif has_object_id:
            colors_rgb = id_to_rgb(object_ids)
        else:
            colors_rgb = np.repeat(default_color_rgb[None, :], len(rows), axis=0)

        return points, colors_rgb, object_ids

    def prepare_object_overview_cloud(self, points, colors_rgb, object_ids):
        cloud = points.astype(np.float32)
        ids = object_ids.astype(np.int32) if len(object_ids) else np.empty((0,), dtype=np.int32)
        if len(cloud) == 0:
            return cloud, colors_rgb.astype(np.uint8), ids

        if len(ids) and self.object_solid_color and not self.overview_use_rgb_texture:
            colors = self.recolor_points_by_object_id(ids)
        else:
            colors = colors_rgb.astype(np.uint8)

        if not len(ids):
            cloud, colors = self.refine_object_cloud(cloud, colors)
            return cloud, colors, ids

        if not self.object_remove_outliers:
            return cloud, colors, ids

        merged_points = []
        merged_colors = []
        merged_ids = []
        for object_id in np.unique(ids):
            mask = ids == int(object_id)
            object_points, object_colors = self.refine_object_cloud(
                cloud[mask],
                colors[mask],
            )
            if len(object_points) == 0:
                continue
            merged_points.append(object_points.astype(np.float32))
            merged_colors.append(object_colors.astype(np.uint8))
            merged_ids.append(
                np.full(len(object_points), int(object_id), dtype=np.int32)
            )
        if not merged_points:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        return (
            np.concatenate(merged_points, axis=0),
            np.concatenate(merged_colors, axis=0),
            np.concatenate(merged_ids, axis=0),
        )

    def prepare_focus_object_cloud(self, points, colors_rgb, object_ids, use_rgb_texture=True):
        if object_ids is None or len(object_ids) == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        focus_object_id = self.select_focus_object_id(object_ids)
        if focus_object_id is None:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        mask = object_ids == int(focus_object_id)
        focus_points = points[mask].astype(np.float32)
        focus_colors = colors_rgb[mask].astype(np.uint8)
        if (not use_rgb_texture) and self.object_solid_color and len(focus_points):
            solid_color = self.color_for_object_id(focus_object_id)
            focus_colors = np.repeat(solid_color[None, :], len(focus_points), axis=0)
        focus_points, focus_colors = self.refine_object_cloud(focus_points, focus_colors)
        focus_ids = np.full(len(focus_points), int(focus_object_id), dtype=np.int32)
        if self.current_focus_object_id != int(focus_object_id):
            self.current_focus_object_id = int(focus_object_id)
            rospy.loginfo(
                "🎯 点云调试器当前聚焦 object_id=%d，点数=%d",
                int(focus_object_id),
                int(len(focus_points)),
            )
            self.refresh_output_grasps()
        return focus_points, focus_colors, focus_ids

    def prepare_object_cloud_for_id(self, points, colors_rgb, object_ids, object_id, use_rgb_texture=True):
        if object_ids is None or len(object_ids) == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        mask = object_ids == int(object_id)
        if not np.any(mask):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        focus_points = points[mask].astype(np.float32)
        focus_colors = colors_rgb[mask].astype(np.uint8)
        if (not use_rgb_texture) and self.object_solid_color and len(focus_points):
            solid_color = self.color_for_object_id(object_id)
            focus_colors = np.repeat(solid_color[None, :], len(focus_points), axis=0)
        focus_points, focus_colors = self.refine_object_cloud(focus_points, focus_colors)
        focus_ids = np.full(len(focus_points), int(object_id), dtype=np.int32)
        return focus_points, focus_colors, focus_ids

    def compute_bounds(self, coords):
        if len(coords) == 0:
            return -1.0, 1.0
        if len(coords) >= 30:
            lo, hi = np.percentile(coords, [1.5, 98.5])
        else:
            lo, hi = float(np.min(coords)), float(np.max(coords))
        if abs(hi - lo) < 1e-4:
            center = 0.5 * (hi + lo)
            lo, hi = center - 0.05, center + 0.05
        margin = 0.08 * (hi - lo)
        return float(lo - margin), float(hi + margin)

    def draw_projection(self, canvas, coords_a, coords_b, colors_bgr, title, a_label, b_label):
        pad = 36
        h, w = canvas.shape[:2]
        if len(coords_a) == 0:
            cv2.putText(
                canvas,
                f"{title}: no points",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            return

        a_min, a_max = self.compute_bounds(coords_a)
        b_min, b_max = self.compute_bounds(coords_b)

        x = (coords_a - a_min) / max(a_max - a_min, 1e-6)
        y = (coords_b - b_min) / max(b_max - b_min, 1e-6)
        px = np.clip((pad + x * (w - 2 * pad - 1)).astype(np.int32), 0, w - 1)
        py = np.clip((h - pad - y * (h - 2 * pad - 1)).astype(np.int32), 0, h - 1)

        canvas[py, px] = colors_bgr
        cv2.rectangle(canvas, (pad, pad), (w - pad, h - pad), (95, 95, 95), 1)
        cv2.putText(
            canvas,
            title,
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{a_label}: {a_min:.3f} -> {a_max:.3f}",
            (18, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{b_label}: {b_min:.3f} -> {b_max:.3f}",
            (18, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def render_debug_image(self, points, colors_rgb, title, frame_id, object_ids):
        side = max(320, int(self.image_size))
        canvas = np.full((side, side * 2, 3), 18, dtype=np.uint8)
        colors_bgr = colors_rgb[:, ::-1] if len(colors_rgb) else np.empty((0, 3), dtype=np.uint8)
        self.draw_projection(
            canvas[:, :side],
            points[:, 0],
            points[:, 1],
            colors_bgr,
            f"{title} | XY Top",
            "x",
            "y",
        )
        self.draw_projection(
            canvas[:, side:],
            points[:, 0],
            points[:, 2],
            colors_bgr,
            f"{title} | XZ Side",
            "x",
            "z",
        )

        unique_ids = np.unique(object_ids).tolist() if len(object_ids) else []
        summary = (
            f"points={len(points)}  frame={frame_id or 'n/a'}  "
            f"xyz_min={np.min(points, axis=0).round(3).tolist()}  "
            f"xyz_max={np.max(points, axis=0).round(3).tolist()}"
            if len(points)
            else f"points=0  frame={frame_id or 'n/a'}"
        )
        cv2.putText(
            canvas,
            summary,
            (16, side - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        if unique_ids:
            cv2.putText(
                canvas,
                f"object_ids={unique_ids[:10]}",
                (16, side - 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def update_o3d_items(self, key, points, colors_rgb, object_ids):
        if (not self.show_o3d_windows) or (key not in self.o3d_window_keys):
            return
        cloud = points.astype(np.float64)
        colors = colors_rgb.astype(np.uint8).copy()
        object_ids_np = object_ids.astype(np.int32) if len(object_ids) else np.empty((0,), dtype=np.int32)
        if len(cloud) > self.max_points:
            step = int(np.ceil(float(len(cloud)) / float(self.max_points)))
            cloud = cloud[::step]
            colors = colors[::step] if len(colors) else colors
            object_ids_np = object_ids_np[::step] if len(object_ids_np) else object_ids_np
        with self.o3d_lock:
            self.o3d_items[key] = {
                "points": cloud,
                "colors": (colors.astype(np.float64) / 255.0) if len(colors) else np.empty((0, 3), dtype=np.float64),
                "object_ids": object_ids_np,
            }
            self.o3d_dirty[key] = True

    def update_dynamic_o3d_item(self, key, label, points, colors_rgb, object_ids, window_object_id):
        if not self.show_o3d_windows:
            return
        cloud = points.astype(np.float64)
        colors = colors_rgb.astype(np.uint8).copy()
        object_ids_np = object_ids.astype(np.int32) if len(object_ids) else np.empty((0,), dtype=np.int32)
        if len(cloud) > self.max_points:
            step = int(np.ceil(float(len(cloud)) / float(self.max_points)))
            cloud = cloud[::step]
            colors = colors[::step] if len(colors) else colors
            object_ids_np = object_ids_np[::step] if len(object_ids_np) else object_ids_np
        with self.o3d_lock:
            self.dynamic_window_specs[key] = {"label": label}
            self.dynamic_o3d_items[key] = {
                "points": cloud,
                "colors": (colors.astype(np.float64) / 255.0) if len(colors) else np.empty((0, 3), dtype=np.float64),
                "object_ids": object_ids_np,
                "window_object_id": int(window_object_id),
            }
            self.dynamic_o3d_dirty[key] = True

    def clear_missing_dynamic_o3d_items(self, active_keys):
        if not self.show_o3d_windows:
            return
        active_keys = set(active_keys)
        with self.o3d_lock:
            for key in list(self.dynamic_window_specs.keys()):
                if key in active_keys:
                    continue
                payload = self.dynamic_o3d_items.get(key)
                if payload is None:
                    continue
                self.dynamic_o3d_items[key] = {
                    "points": np.empty((0, 3), dtype=np.float64),
                    "colors": np.empty((0, 3), dtype=np.float64),
                    "object_ids": np.empty((0,), dtype=np.int32),
                    "window_object_id": payload.get("window_object_id"),
                }
                self.dynamic_o3d_dirty[key] = True

    def marker_array_to_lineset(self, msg, default_color):
        all_points = []
        all_lines = []
        all_colors = []
        point_offset = 0
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue
            if marker.type != Marker.LINE_LIST or len(marker.points) < 2:
                continue
            color = [
                float(marker.color.r) if marker.color.a > 0.0 else default_color[0],
                float(marker.color.g) if marker.color.a > 0.0 else default_color[1],
                float(marker.color.b) if marker.color.a > 0.0 else default_color[2],
            ]
            marker_points = []
            for pt in marker.points:
                marker_points.append([float(pt.x), float(pt.y), float(pt.z)])
            if not marker_points:
                continue
            all_points.extend(marker_points)
            line_count = len(marker_points) // 2
            for idx in range(line_count):
                all_lines.append([point_offset + 2 * idx, point_offset + 2 * idx + 1])
                all_colors.append(color)
            point_offset += len(marker_points)

        if not all_points or not all_lines:
            return None
        return {
            "points": np.asarray(all_points, dtype=np.float64),
            "lines": np.asarray(all_lines, dtype=np.int32),
            "colors": np.asarray(all_colors, dtype=np.float64),
        }

    def marker_callback(self, kind, msg):
        default_color = (1.0, 0.35, 0.35) if kind == "raw" else (0.10, 0.75, 0.20)
        payload = self.marker_array_to_lineset(msg, default_color)
        with self.o3d_lock:
            self.grasp_line_items[kind] = payload
            self.grasp_dirty = True

    def rotation_align_z_to_vector(self, vector):
        target = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(target)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        target = target / norm
        source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cross = np.cross(source, target)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if cross_norm < 1e-9:
            if dot > 0.0:
                return np.eye(3, dtype=np.float64)
            return tft.rotation_matrix(np.pi, [1.0, 0.0, 0.0])[:3, :3]
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return (
            np.eye(3, dtype=np.float64)
            + skew
            + skew @ skew * ((1.0 - dot) / max(cross_norm * cross_norm, 1e-12))
        )

    def create_segment_mesh(self, start, end, radius, color):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-9:
            return None
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=max(1e-4, float(radius)),
            height=length,
            resolution=18,
            split=1,
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))
        mesh.rotate(
            self.rotation_align_z_to_vector(segment),
            center=np.zeros(3, dtype=np.float64),
        )
        mesh.translate(0.5 * (start + end))
        return mesh

    def add_line_payload_as_mesh(self, vis, line_payload):
        points = np.asarray(line_payload.get("points", []), dtype=np.float64)
        lines = np.asarray(line_payload.get("lines", []), dtype=np.int32)
        colors = np.asarray(line_payload.get("colors", []), dtype=np.float64)
        if len(points) == 0 or len(lines) == 0:
            return
        for idx, (a, b) in enumerate(lines):
            color = colors[min(idx, len(colors) - 1)] if len(colors) else np.array([0.10, 0.55, 0.18], dtype=np.float64)
            mesh = self.create_segment_mesh(
                points[a],
                points[b],
                self.grasp_mesh_radius,
                color,
            )
            if mesh is not None:
                vis.add_geometry(mesh, reset_bounding_box=False)
        centers = np.asarray(line_payload.get("centers", []), dtype=np.float64)
        if len(centers):
            sphere_color = np.array([0.93, 0.34, 0.18], dtype=np.float64)
            for center in centers:
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=max(1e-4, self.grasp_center_radius),
                    resolution=16,
                )
                sphere.compute_vertex_normals()
                sphere.paint_uniform_color(sphere_color)
                sphere.translate(center.astype(np.float64))
                vis.add_geometry(sphere, reset_bounding_box=False)

    def render_o3d_cloud(self, vis, key, payload):
        vis.clear_geometries()
        if payload is None or len(payload["points"]) == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(payload["points"])
        if len(payload["colors"]) == len(payload["points"]):
            pcd.colors = o3d.utility.Vector3dVector(payload["colors"])
        else:
            pcd.paint_uniform_color([0.82, 0.82, 0.82])

        if self.o3d_voxel_size > 1e-6:
            pcd = pcd.voxel_down_sample(self.o3d_voxel_size)

        vis.add_geometry(pcd, reset_bounding_box=True)
        if self.show_coordinate_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            vis.add_geometry(frame, reset_bounding_box=False)
        if self.show_bounding_box:
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (0.35, 0.35, 0.35) if self.o3d_white_background else (0.95, 0.95, 0.95)
            vis.add_geometry(bbox, reset_bounding_box=False)

        dynamic_overlay = str(key).startswith("object_multi_overlay_")
        if key in ("object", "object_focus_overlay") or dynamic_overlay:
            line_payloads = []
            if key == "object" and self.show_output_grasps:
                line_payloads.append(self.grasp_line_items.get("output_overview"))
            if key == "object_focus_overlay" and self.show_output_grasps:
                line_payloads.append(self.grasp_line_items.get("output_focus"))
            if dynamic_overlay and self.show_output_grasps:
                object_id = payload.get("window_object_id")
                per_object_payloads = self.grasp_line_items.get("output_per_object", {})
                line_payloads.append(per_object_payloads.get(int(object_id)) if object_id is not None else None)
            if key == "object_focus_overlay" and self.show_raw_grasps:
                line_payloads.append(self.grasp_line_items.get("raw"))
            if key == "object_focus_overlay" and self.show_corrected_grasps:
                line_payloads.append(self.grasp_line_items.get("corrected"))
            for line_payload in line_payloads:
                if not line_payload:
                    continue
                focus_object_id = line_payload.get("focus_object_id")
                if (
                    focus_object_id is not None
                    and self.current_focus_object_id is not None
                    and int(focus_object_id) != int(self.current_focus_object_id)
                ):
                    continue
                if (key == "object_focus_overlay" or dynamic_overlay) and line_payload.get("render_style") == "mesh":
                    self.add_line_payload_as_mesh(vis, line_payload)
                else:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_payload["points"])
                    line_set.lines = o3d.utility.Vector2iVector(line_payload["lines"])
                    line_set.colors = o3d.utility.Vector3dVector(line_payload["colors"])
                    vis.add_geometry(line_set, reset_bounding_box=False)

    def o3d_visualization_loop(self):
        window_specs = dict(self.topic_specs)
        if self.show_focus_pure_object:
            window_specs["object_focus_pure"] = {
                "label": self.focus_pure_label,
            }
        if self.show_focus_overlay_object:
            window_specs["object_focus_overlay"] = {
                "label": self.focus_overlay_label,
            }

        visualizers = {}

        def ensure_visualizer(key):
            if key in visualizers:
                return visualizers[key]
            spec = enabled_window_specs[key]
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{spec['label']} 3D",
                width=960,
                height=720,
            )
            render_option = vis.get_render_option()
            render_option.background_color = (
                np.asarray([1.0, 1.0, 1.0])
                if self.o3d_white_background
                else np.asarray([0.08, 0.08, 0.08])
            )
            render_option.point_size = float(self.o3d_point_size)
            try:
                render_option.line_width = 2.0
            except Exception:
                pass
            visualizers[key] = vis
            return vis

        try:
            while not rospy.is_shutdown():
                with self.o3d_lock:
                    dynamic_specs = dict(self.dynamic_window_specs)
                    latest_items = {
                        key: self.o3d_items.get(key) for key in self.o3d_window_keys
                    }
                    latest_items.update(dynamic_specs and {key: self.dynamic_o3d_items.get(key) for key in dynamic_specs} or {})
                    dirty_flags = dict(self.o3d_dirty)
                    dirty_flags.update(self.dynamic_o3d_dirty)
                    grasp_dirty = bool(self.grasp_dirty)
                enabled_window_specs = {
                    key: spec for key, spec in window_specs.items() if key in self.o3d_window_keys
                }
                enabled_window_specs.update(dynamic_specs)

                for key in enabled_window_specs:
                    payload = latest_items.get(key)
                    should_render = bool(dirty_flags.get(key))
                    if key in ("object", "object_focus_overlay") or str(key).startswith("object_multi_overlay_"):
                        if grasp_dirty:
                            should_render = True
                    if payload is not None and should_render:
                        vis = ensure_visualizer(key)
                        self.render_o3d_cloud(vis, key, payload)
                        with self.o3d_lock:
                            if key in self.o3d_dirty:
                                self.o3d_dirty[key] = False
                            if key in self.dynamic_o3d_dirty:
                                self.dynamic_o3d_dirty[key] = False
                            if key == "object_focus_overlay" or str(key).startswith("object_multi_overlay_") or (
                                key == "object" and not self.show_focus_overlay_object
                            ):
                                self.grasp_dirty = False
                    vis = visualizers.get(key)
                    if vis is not None:
                        vis.poll_events()
                        vis.update_renderer()
                rospy.sleep(0.05)
        except rospy.ROSInterruptException:
            pass

    def write_ply(self, path, points, colors_rgb):
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors_rgb):
                f.write(
                    f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

    def save_publication_figure(self, path, points, colors_rgb, line_payload=None, title=""):
        if len(points) == 0 or not self.paper_figure_enabled:
            return
        fig = None
        plt = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import matplotlib.pyplot as plt
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "⚠️ matplotlib 不可用，跳过论文风格点云图导出: %s", exc)
            self.paper_figure_enabled = False
            return

        try:
            fig = plt.figure(figsize=(7.2, 7.2), facecolor="white")
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("white")

            colors = colors_rgb.astype(np.float32) / 255.0 if len(colors_rgb) else None
            point_size = 1.8 if len(points) > 1500 else 4.0
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=point_size,
                c=colors,
                depthshade=False,
                linewidths=0.0,
                alpha=0.98,
            )

            if line_payload and len(line_payload.get("points", [])) and len(line_payload.get("lines", [])):
                line_points = np.asarray(line_payload["points"], dtype=np.float64)
                line_indices = np.asarray(line_payload["lines"], dtype=np.int32)
                line_colors = np.asarray(line_payload["colors"], dtype=np.float64)
                for idx, (a, b) in enumerate(line_indices):
                    line_color = line_colors[min(idx, len(line_colors) - 1)] if len(line_colors) else np.array([0.12, 0.55, 0.18])
                    seg = line_points[[a, b]]
                    ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        seg[:, 2],
                        color=line_color,
                        linewidth=2.5,
                        alpha=0.98,
                    )
                centers = np.asarray(line_payload.get("centers", []), dtype=np.float64)
                if len(centers):
                    ax.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        centers[:, 2],
                        s=62,
                        c=np.asarray([[0.96, 0.34, 0.16]] * len(centers), dtype=np.float64),
                        depthshade=False,
                        linewidths=0.0,
                        alpha=0.98,
                    )

            xyz_min = np.min(points, axis=0)
            xyz_max = np.max(points, axis=0)
            center = 0.5 * (xyz_min + xyz_max)
            radius = max(float(np.max(xyz_max - xyz_min)) * 0.72, 0.05)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=18, azim=-63)
            ax.set_axis_off()
            if title:
                ax.set_title(title, pad=10, fontsize=13, color="#1f2937")

            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
            fig.savefig(path, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "⚠️ 论文风格点云图导出失败，后续将停用 paper figure: %s", exc)
            self.paper_figure_enabled = False
        finally:
            if plt is not None and fig is not None:
                plt.close(fig)

    def maybe_save_snapshot(self, key, points, colors_rgb, object_ids, image, line_payload=None, title=""):
        if not self.save_snapshots or len(points) == 0:
            return
        now = time.time()
        if now - self.last_saved_at[key] < self.save_interval_sec:
            return
        self.last_saved_at[key] = now

        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(self.output_dir, f"{key}_{stamp}")
        np.savez_compressed(
            f"{prefix}.npz",
            points=points.astype(np.float32),
            colors_rgb=colors_rgb.astype(np.uint8),
            object_ids=object_ids.astype(np.int32),
        )
        self.write_ply(f"{prefix}.ply", points, colors_rgb)
        cv2.imwrite(f"{prefix}.png", image)
        saved_paper = False
        if str(key).startswith("object"):
            self.save_publication_figure(
                f"{prefix}_paper.png",
                points,
                colors_rgb,
                line_payload=line_payload,
                title=title,
            )
            saved_paper = self.paper_figure_enabled
        if saved_paper:
            rospy.loginfo("💾 已保存点云快照: %s.[npz|ply|png|paper.png]", prefix)
        else:
            rospy.loginfo("💾 已保存点云快照: %s.[npz|ply|png]", prefix)

    def publish_stats(self, key, label, frame_id, points, object_ids):
        payload = {
            "stream": key,
            "label": label,
            "frame_id": frame_id,
            "point_count": int(len(points)),
            "object_count": int(len(np.unique(object_ids))) if len(object_ids) else 0,
            "xyz_min": np.min(points, axis=0).round(4).tolist() if len(points) else [],
            "xyz_max": np.max(points, axis=0).round(4).tolist() if len(points) else [],
        }
        self.stats_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def cloud_callback(self, key, msg):
        spec = self.topic_specs[key]
        self.received_counts[key] += 1
        self.last_message_at[key] = rospy.Time.now().to_sec()
        try:
            points, colors_rgb, object_ids = self.pointcloud_to_arrays(
                msg, spec["default_color_rgb"]
            )
        except Exception as exc:
            rospy.logerr_throttle(
                5.0,
                "pointcloud_debugger failed to parse %s (%s): %s",
                key,
                spec["topic"],
                exc,
            )
            return

        frame_id = getattr(msg.header, "frame_id", "")

        if key == "object":
            overview_points, overview_colors_rgb, overview_object_ids = self.prepare_object_overview_cloud(
                points, colors_rgb, object_ids
            )
            self.update_o3d_items("object", overview_points, overview_colors_rgb, overview_object_ids)

            focus_points = np.empty((0, 3), dtype=np.float32)
            focus_colors_rgb = np.empty((0, 3), dtype=np.uint8)
            focus_object_ids = np.empty((0,), dtype=np.int32)
            if self.show_focus_pure_object or self.show_focus_overlay_object:
                focus_points, focus_colors_rgb, focus_object_ids = self.prepare_focus_object_cloud(
                    points,
                    colors_rgb,
                    object_ids,
                    use_rgb_texture=self.focus_use_rgb_texture,
                )
                if self.show_focus_pure_object:
                    self.update_o3d_items("object_focus_pure", focus_points, focus_colors_rgb, focus_object_ids)
                if self.show_focus_overlay_object:
                    self.update_o3d_items("object_focus_overlay", focus_points, focus_colors_rgb, focus_object_ids)

            active_dynamic_keys = []
            if self.show_all_object_focus_windows:
                for object_id in self.select_detail_object_ids(overview_object_ids):
                    detail_points, detail_colors_rgb, detail_object_ids = self.prepare_object_cloud_for_id(
                        overview_points,
                        overview_colors_rgb,
                        overview_object_ids,
                        object_id,
                        use_rgb_texture=self.focus_use_rgb_texture,
                    )
                    if self.show_all_object_focus_pure_o3d and self.show_focus_pure_object:
                        pure_key = self.dynamic_object_key(object_id, "pure")
                        active_dynamic_keys.append(pure_key)
                        self.update_dynamic_o3d_item(
                            pure_key,
                            self.dynamic_object_label(object_id, "pure"),
                            detail_points,
                            detail_colors_rgb,
                            detail_object_ids,
                            object_id,
                        )
                    if self.show_all_object_focus_overlay_o3d and self.show_focus_overlay_object:
                        overlay_key = self.dynamic_object_key(object_id, "overlay")
                        active_dynamic_keys.append(overlay_key)
                        self.update_dynamic_o3d_item(
                            overlay_key,
                            self.dynamic_object_label(object_id, "overlay"),
                            detail_points,
                            detail_colors_rgb,
                            detail_object_ids,
                            object_id,
                        )
            self.clear_missing_dynamic_o3d_items(active_dynamic_keys)

            overview_image = self.render_debug_image(
                overview_points,
                overview_colors_rgb,
                "SAM Objects Overview",
                frame_id,
                overview_object_ids,
            )
            self.publish_stats(
                "object",
                "SAM Objects Overview",
                frame_id,
                overview_points,
                overview_object_ids,
            )

            if self.publish_projection_images:
                self.image_pubs["object"].publish(
                    numpy_to_image_msg(
                        overview_image,
                        encoding="bgr8",
                        stamp=msg.header.stamp,
                        frame_id=frame_id,
                    )
                )
            if self.show_projection_windows:
                cv2.imshow(spec["window"], overview_image)
                cv2.waitKey(1)

            self.maybe_save_snapshot(
                "object",
                overview_points,
                overview_colors_rgb,
                overview_object_ids,
                overview_image,
                line_payload=self.grasp_line_items.get("output_overview"),
                title="SAM Objects Overview",
            )

            if self.show_focus_pure_object:
                focus_pure_image = self.render_debug_image(
                    focus_points,
                    focus_colors_rgb,
                    self.focus_pure_label,
                    frame_id,
                    focus_object_ids,
                )
                self.publish_stats(
                    "object_focus_pure",
                    self.focus_pure_label,
                    frame_id,
                    focus_points,
                    focus_object_ids,
                )
                if self.publish_projection_images and "object_focus_pure" in self.image_pubs:
                    self.image_pubs["object_focus_pure"].publish(
                        numpy_to_image_msg(
                            focus_pure_image,
                            encoding="bgr8",
                            stamp=msg.header.stamp,
                            frame_id=frame_id,
                        )
                    )
                if self.show_projection_windows:
                    cv2.imshow(self.focus_pure_window_name, focus_pure_image)
                    cv2.waitKey(1)
                self.maybe_save_snapshot(
                    "object_focus_pure",
                    focus_points,
                    focus_colors_rgb,
                    focus_object_ids,
                    focus_pure_image,
                    line_payload=None,
                    title=self.focus_pure_label,
                )

            if self.show_focus_overlay_object:
                focus_overlay_image = self.render_debug_image(
                    focus_points,
                    focus_colors_rgb,
                    self.focus_overlay_label,
                    frame_id,
                    focus_object_ids,
                )
                self.publish_stats(
                    "object_focus_overlay",
                    self.focus_overlay_label,
                    frame_id,
                    focus_points,
                    focus_object_ids,
                )
                if self.publish_projection_images and "object_focus_overlay" in self.image_pubs:
                    self.image_pubs["object_focus_overlay"].publish(
                        numpy_to_image_msg(
                            focus_overlay_image,
                            encoding="bgr8",
                            stamp=msg.header.stamp,
                            frame_id=frame_id,
                        )
                    )
                if self.show_projection_windows:
                    cv2.imshow(self.focus_overlay_window_name, focus_overlay_image)
                    cv2.waitKey(1)
                self.maybe_save_snapshot(
                    "object_focus_overlay",
                    focus_points,
                    focus_colors_rgb,
                    focus_object_ids,
                    focus_overlay_image,
                    line_payload=self.grasp_line_items.get("output_focus"),
                    title=self.focus_overlay_label,
                )

            rospy.loginfo_throttle(
                3.0,
                "☁️ %s 已收到点云: topic=%s points=%d objects=%d frame=%s",
                "SAM Objects Overview",
                spec["topic"],
                len(overview_points),
                len(np.unique(overview_object_ids)) if len(overview_object_ids) else 0,
                frame_id,
            )
            return

        self.update_o3d_items(key, points, colors_rgb, object_ids)

        image = self.render_debug_image(
            points,
            colors_rgb,
            spec["label"],
            frame_id,
            object_ids,
        )
        self.publish_stats(
            key,
            spec["label"],
            frame_id,
            points,
            object_ids,
        )

        if self.publish_projection_images:
            self.image_pubs[key].publish(
                numpy_to_image_msg(
                    image,
                    encoding="bgr8",
                    stamp=msg.header.stamp,
                    frame_id=frame_id,
                )
            )
        if self.show_projection_windows:
            cv2.imshow(spec["window"], image)
            cv2.waitKey(1)

        self.maybe_save_snapshot(
            key,
            points,
            colors_rgb,
            object_ids,
            image,
            title=spec["label"],
        )
        rospy.loginfo_throttle(
            3.0,
            "☁️ %s 已收到点云: topic=%s points=%d frame=%s",
            spec["label"],
            spec["topic"],
            len(points),
            frame_id,
        )


if __name__ == "__main__":
    PointCloudDebugger()
    rospy.spin()
