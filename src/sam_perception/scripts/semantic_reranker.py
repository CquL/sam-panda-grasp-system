#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import os
import re
import sys
import time

os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libffi.so.7"
sys.path = [p for p in sys.path if "/usr/lib/python3/dist-packages" not in p]

import cv2
import numpy as np
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
from openai import OpenAI
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String


DEFAULT_API_KEY = "sk-a85a76a8ada94ef2886ecd43bf0f9e80"


class SemanticRerankerNode:
    def __init__(self):
        rospy.init_node("semantic_reranker")
        self.bridge = CvBridge()
        self.latest_image = None
        self.target_image = None
        self.intrinsics = None
        self.latest_target = None
        self.latest_object_metadata = {}
        self.pending_pose_msg = None
        self.pending_info_msg = None

        self.pose_topic = rospy.get_param("~pose_topic", "/graspnet/grasp_pose_array_raw")
        self.info_topic = rospy.get_param("~info_topic", "/graspnet/grasp_info_raw")
        self.out_pose_topic = rospy.get_param("~out_pose_topic", "/graspnet/grasp_pose_array_semantic")
        self.out_info_topic = rospy.get_param("~out_info_topic", "/graspnet/grasp_info_semantic")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.target_topic = rospy.get_param("~target_topic", "/vlm/global_target")
        self.target_image_topic = rospy.get_param("~target_image_topic", "/vlm/global_target_image")
        self.object_metadata_topic = rospy.get_param("~object_metadata_topic", "/sam_perception/object_metadata")
        self.result_topic = rospy.get_param("~result_topic", "/vlm/semantic_rerank")
        self.model_name = rospy.get_param("~model", "qwen-vl-max")
        self.base_url = rospy.get_param(
            "~base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key = rospy.get_param("~api_key", os.environ.get("DASHSCOPE_API_KEY", DEFAULT_API_KEY))
        self.max_candidates_per_object = max(1, int(rospy.get_param("~max_candidates_per_object", 3)))
        self.overview_candidates_per_object = max(1, int(rospy.get_param("~overview_candidates_per_object", 15)))
        self.crop_margin_px = max(20, int(rospy.get_param("~crop_margin_px", 40)))
        self.axis_length = float(rospy.get_param("~axis_length", 0.04))
        self.gripper_width_max = float(rospy.get_param("~gripper_width_max", 0.078))
        self.gripper_width_margin = float(rospy.get_param("~gripper_width_margin", 0.002))
        self.strict_width_filter = bool(rospy.get_param("~strict_width_filter", True))
        self.visualize_candidate_crop = bool(rospy.get_param("~visualize_candidate_crop", True))
        self.save_candidate_crop = bool(rospy.get_param("~save_candidate_crop", True))
        self.save_candidate_json = bool(rospy.get_param("~save_candidate_json", True))
        self.candidate_debug_dir = os.path.expanduser(
            rospy.get_param("~candidate_debug_dir", "~/grasp_robot_ws/semantic_rerank_debug")
        )
        self.visualize_global_overview = bool(rospy.get_param("~visualize_global_overview", True))
        self.save_global_overview = bool(rospy.get_param("~save_global_overview", True))
        self.global_overview_dir = os.path.expanduser(
            rospy.get_param("~global_overview_dir", "~/grasp_robot_ws/semantic_rerank_overview")
        )

        proxy_vars = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        for var in proxy_vars:
            os.environ.pop(var, None)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.target_topic, String, self.target_cb, queue_size=1)
        rospy.Subscriber(self.target_image_topic, Image, self.target_image_cb, queue_size=1)
        rospy.Subscriber(self.object_metadata_topic, String, self.object_metadata_cb, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseArray, self.pose_cb, queue_size=1)
        rospy.Subscriber(self.info_topic, Float32MultiArray, self.info_cb, queue_size=1)

        self.pub_pose = rospy.Publisher(self.out_pose_topic, PoseArray, queue_size=1)
        self.pub_info = rospy.Publisher(self.out_info_topic, Float32MultiArray, queue_size=1)
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=1)

        rospy.loginfo(
            "Semantic reranker ready. input=(%s,%s) output=(%s,%s)",
            self.pose_topic,
            self.info_topic,
            self.out_pose_topic,
            self.out_info_topic,
        )

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.latest_image = None

    def camera_info_cb(self, msg):
        try:
            self.intrinsics = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        except Exception:
            self.intrinsics = None

    def target_image_cb(self, msg):
        try:
            self.target_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.target_image = None

    def target_cb(self, msg):
        try:
            self.latest_target = json.loads(msg.data)
        except Exception:
            self.latest_target = None

    def object_metadata_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.latest_object_metadata = {}
            return

        metadata_map = {}
        for item in payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = item.get("object_id")
            try:
                object_id = int(object_id)
            except Exception:
                continue
            metadata_map[object_id] = item
        self.latest_object_metadata = metadata_map

    def pose_cb(self, msg):
        self.pending_pose_msg = msg
        self.try_process()

    def info_cb(self, msg):
        self.pending_info_msg = msg
        self.try_process()

    def try_process(self):
        if self.pending_pose_msg is None or self.pending_info_msg is None:
            return

        pose_msg = self.pending_pose_msg
        info_msg = self.pending_info_msg
        self.pending_pose_msg = None
        self.pending_info_msg = None

        reordered_pose_msg, reordered_info_msg, debug_payload = self.rerank_batch(pose_msg, info_msg)
        self.pub_pose.publish(reordered_pose_msg)
        self.pub_info.publish(reordered_info_msg)
        rospy.loginfo(
            "📤 semantic_reranker 已发布候选: poses=%d, info_len=%d, target=%s",
            len(reordered_pose_msg.poses),
            len(reordered_info_msg.data),
            debug_payload.get("target", "") if isinstance(debug_payload, dict) else "",
        )
        if debug_payload is not None:
            self.pub_result.publish(String(data=json.dumps(debug_payload, ensure_ascii=False)))

    def rerank_batch(self, pose_msg, info_msg):
        num_poses = len(pose_msg.poses)
        data = list(info_msg.data)
        info_stride = int(len(data) / num_poses) if num_poses > 0 and len(data) % num_poses == 0 else 0
        if num_poses == 0 or info_stride < 4:
            return pose_msg, info_msg, None

        object_ids = [int(data[i * info_stride]) for i in range(num_poses)]
        object_groups = {}
        for idx, object_id in enumerate(object_ids):
            object_groups.setdefault(object_id, []).append(idx)

        semantic_scores = [self.extract_existing_semantic(data, info_stride, idx) for idx in range(num_poses)]
        order = list(range(num_poses))
        debug_payload = {
            "ok": False,
            "target": None,
            "command": None,
            "objects": [],
        }

        if not self.target_ready():
            reordered_pose, reordered_info = self.build_outputs(pose_msg, data, info_stride, order, semantic_scores)
            return reordered_pose, reordered_info, debug_payload

        debug_payload["ok"] = True
        debug_payload["target"] = self.latest_target.get("target", "")
        debug_payload["command"] = self.latest_target.get("command", "")
        debug_payload["overview_items"] = []

        new_order = []
        for object_id in sorted(object_groups.keys()):
            indices = object_groups[object_id]
            bbox = self.lookup_bbox_for_object(object_id)
            if bbox is None:
                new_order.extend(indices)
                continue

            rank_result = self.rank_object_candidates(object_id, bbox, indices, pose_msg, data, info_stride)
            if rank_result is None:
                new_order.extend(indices)
                continue

            chosen_order, chosen_scores, result_payload = rank_result
            for idx, score in chosen_scores.items():
                semantic_scores[idx] = score
            new_order.extend(chosen_order)
            debug_payload["objects"].append(result_payload)
            if "overview_item" in result_payload:
                debug_payload["overview_items"].append(result_payload["overview_item"])

        order = new_order

        self.visualize_global_overview_canvas(debug_payload)
        reordered_pose, reordered_info = self.build_outputs(pose_msg, data, info_stride, order, semantic_scores)
        return reordered_pose, reordered_info, debug_payload

    def build_outputs(self, pose_msg, data, info_stride, order, semantic_scores):
        reordered_pose = PoseArray()
        reordered_pose.header = pose_msg.header
        reordered_info = []
        for idx in order:
            reordered_pose.poses.append(pose_msg.poses[idx])
            base = idx * info_stride
            object_id = int(data[base])
            width = float(data[base + 1])
            score = float(data[base + 2])
            depth = float(data[base + 3])
            reordered_info.extend([object_id, width, score, depth, float(semantic_scores[idx])])
        return reordered_pose, Float32MultiArray(data=reordered_info)

    def target_ready(self):
        return (
            self.get_reference_image() is not None
            and self.intrinsics is not None
            and isinstance(self.latest_target, dict)
            and self.latest_target.get("ok", False)
            and isinstance(self.latest_target.get("bboxes", []), list)
        )

    def get_reference_image(self):
        return self.target_image if self.target_image is not None else self.latest_image

    def extract_existing_semantic(self, data, info_stride, idx):
        if info_stride >= 5:
            try:
                return float(data[idx * info_stride + 4])
            except Exception:
                return 0.0
        return 0.0

    def lookup_bbox_for_object(self, object_id):
        metadata = self.latest_object_metadata.get(int(object_id))
        if isinstance(metadata, dict):
            source_bbox = metadata.get("source_bbox")
            if self.is_valid_bbox(source_bbox):
                return [int(v) for v in source_bbox]
            mask_bbox = metadata.get("mask_bbox")
            if self.is_valid_bbox(mask_bbox):
                return [int(v) for v in mask_bbox]

        boxes = self.latest_target.get("bboxes", [])
        if 0 <= object_id < len(boxes):
            return boxes[object_id]
        if len(boxes) == 1:
            return boxes[0]
        return None

    def is_valid_bbox(self, bbox):
        return isinstance(bbox, (list, tuple)) and len(bbox) == 4

    def rank_object_candidates(self, object_id, bbox, indices, pose_msg, data, info_stride):
        raw_scores = {idx: float(data[idx * info_stride + 2]) for idx in indices}
        width_map = {idx: float(data[idx * info_stride + 1]) for idx in indices}
        depth_map = {idx: float(data[idx * info_stride + 3]) for idx in indices}
        width_limit = max(0.0, self.gripper_width_max - self.gripper_width_margin)
        ranked_by_score = sorted(indices, key=lambda idx: raw_scores[idx], reverse=True)
        width_ok_indices = [idx for idx in ranked_by_score if width_map[idx] <= width_limit]
        overview_candidates = self.collect_projected_candidates(
            ranked_by_score[: self.overview_candidates_per_object],
            pose_msg,
            width_map,
            depth_map,
            raw_scores,
            width_limit,
        )

        if self.strict_width_filter and not width_ok_indices:
            rospy.logwarn(
                f"semantic_reranker: 目标 {object_id} 所有候选均超过夹爪上限 {width_limit:.3f} m，当前轮直接剔除。"
            )
            return [], {}, {
                "object_id": object_id,
                "selected_candidates": [],
                "best_candidate": None,
                "preferred_region": "filtered_out",
                "avoid_region": "over_width_limit",
                "grasp_side": "unknown",
                "reason": "all_candidates_over_width_limit",
            }

        if self.strict_width_filter and width_ok_indices:
            candidate_pool = width_ok_indices
        else:
            candidate_pool = ranked_by_score

        top_indices = candidate_pool[: self.max_candidates_per_object]
        if not top_indices:
            return None

        if len(top_indices) == 1 and len(indices) == 1:
            only_idx = top_indices[0]
            return [only_idx], {only_idx: 1.0}, {
                "object_id": object_id,
                "selected_candidates": [only_idx],
                "best_candidate": only_idx,
                "preferred_region": "unknown",
                "avoid_region": "unknown",
                "grasp_side": "unknown",
                "reason": "single_candidate",
                "overview_item": {
                    "object_id": int(object_id),
                    "bbox": [int(v) for v in bbox],
                    "best_candidate": int(only_idx),
                    "best_label": "best",
                    "preferred_region": "unknown",
                    "candidates": overview_candidates,
                },
            }

        crop_image, labels, visible_indices, candidate_meta = self.render_candidate_crop(
            bbox, top_indices, pose_msg, width_map, depth_map, raw_scores, width_limit
        )
        if crop_image is None or not labels or not visible_indices or not candidate_meta:
            if len(top_indices) == 1:
                only_idx = top_indices[0]
                ordered = [only_idx] + [idx for idx in indices if idx != only_idx]
                return ordered, {only_idx: 1.0}, {
                    "object_id": object_id,
                    "selected_candidates": [only_idx],
                    "best_candidate": only_idx,
                    "preferred_region": "unknown",
                    "avoid_region": "unknown",
                    "grasp_side": "unknown",
                    "reason": "single_visible_candidate",
                }
            return None

        result = self.query_vlm_for_candidates(crop_image, labels, candidate_meta)
        if result is None:
            return None

        semantic_vector = result.get("semantic_scores", [])
        chosen_scores = {}
        for local_idx, candidate_idx in enumerate(visible_indices):
            score = 0.0
            if local_idx < len(semantic_vector):
                score = self.safe_float(semantic_vector[local_idx])
            chosen_scores[candidate_idx] = score

        ordered_indices = sorted(
            indices,
            key=lambda idx: (chosen_scores.get(idx, 0.0), raw_scores[idx]),
            reverse=True,
        )

        best_local_index = int(result.get("best_index", 1)) - 1
        best_local_index = min(max(best_local_index, 0), len(visible_indices) - 1)
        best_candidate = visible_indices[best_local_index]
        best_label = None
        overview_candidates = []
        for item in candidate_meta:
            if int(item["candidate_index"]) == int(best_candidate):
                best_label = item["label"]
            overview_candidates.append(
                {
                    "label": item["label"],
                    "candidate_index": int(item["candidate_index"]),
                    "center_px": item.get("center_px"),
                    "arrow_end_px": item.get("arrow_end_px"),
                    "width_ok": bool(item.get("width_ok", True)),
                }
            )

        object_payload = {
            "object_id": object_id,
            "selected_candidates": visible_indices,
            "best_candidate": best_candidate,
            "preferred_region": result.get("preferred_region", "unknown"),
            "avoid_region": result.get("avoid_region", "unknown"),
            "grasp_side": result.get("grasp_side", "unknown"),
            "reason": result.get("reason", ""),
            "overview_item": {
                "object_id": int(object_id),
                "bbox": [int(v) for v in bbox],
                "best_candidate": int(best_candidate),
                "best_label": best_label or "?",
                "preferred_region": result.get("preferred_region", "unknown"),
                "candidates": overview_candidates,
            },
        }
        self.visualize_candidate_decision(crop_image, object_id, candidate_meta, object_payload)
        return ordered_indices, chosen_scores, object_payload

    def collect_projected_candidates(self, indices, pose_msg, width_map, depth_map, raw_scores, width_limit):
        projected = []
        for rank, idx in enumerate(indices):
            pose = pose_msg.poses[idx]
            center = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
            center_px = self.project_point(center)
            if center_px is None:
                continue

            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            rot = tft.quaternion_matrix(quat)[:3, :3]
            arrow_end = center + rot[:, 0] * self.axis_length
            arrow_end_px = self.project_point(arrow_end)

            projected.append(
                {
                    "rank": int(rank + 1),
                    "candidate_index": int(idx),
                    "center_px": (int(center_px[0]), int(center_px[1])),
                    "arrow_end_px": (int(arrow_end_px[0]), int(arrow_end_px[1])) if arrow_end_px is not None else None,
                    "width_m": float(width_map[idx]),
                    "depth_m": float(depth_map[idx]),
                    "raw_score": float(raw_scores[idx]),
                    "width_ok": bool(width_map[idx] <= width_limit),
                }
            )
        return projected

    def render_candidate_crop(self, bbox, indices, pose_msg, width_map, depth_map, raw_scores, width_limit):
        ref_image = self.get_reference_image()
        if ref_image is None:
            return None, [], [], []

        canvas = ref_image.copy()
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        bbox_w = max(1, xmax - xmin)
        bbox_h = max(1, ymax - ymin)
        margin = self.crop_margin_px + int(0.25 * max(bbox_w, bbox_h))

        crop_xmin = max(0, xmin - margin)
        crop_ymin = max(0, ymin - margin)
        crop_xmax = min(canvas.shape[1] - 1, xmax + margin)
        crop_ymax = min(canvas.shape[0] - 1, ymax + margin)

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)

        colors = [(0, 255, 255), (0, 200, 0), (0, 128, 255), (255, 255, 0)]
        labels = []
        visible_indices = []
        candidate_meta = []
        bbox_cx = int((xmin + xmax) / 2.0)
        bbox_cy = int((ymin + ymax) / 2.0)
        for rank, idx in enumerate(indices):
            pose = pose_msg.poses[idx]
            center = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
            center_px = self.project_point(center)
            if center_px is None:
                continue

            label = f"C{len(labels) + 1}"
            labels.append(label)
            visible_indices.append(idx)
            color = colors[rank % len(colors)]
            u, v = center_px
            if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                cv2.circle(canvas, (u, v), 5, color, -1)
                cv2.putText(
                    canvas,
                    label,
                    (u + 5, max(v - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            rot = tft.quaternion_matrix(quat)[:3, :3]
            arrow_end = center + rot[:, 0] * self.axis_length
            arrow_end_px = self.project_point(arrow_end)
            if arrow_end_px is not None and center_px is not None:
                cv2.arrowedLine(canvas, center_px, arrow_end_px, color, 2, tipLength=0.2)

            center_offset_px = float(np.linalg.norm(np.array([u - bbox_cx, v - bbox_cy], dtype=np.float64)))
            candidate_meta.append(
                {
                    "label": label,
                    "candidate_index": int(idx),
                    "width_m": float(width_map[idx]),
                    "depth_m": float(depth_map[idx]),
                    "raw_score": float(raw_scores[idx]),
                    "center_offset_px": center_offset_px,
                    "width_ok": bool(width_map[idx] <= width_limit),
                    "center_px": (int(u), int(v)),
                    "arrow_end_px": (int(arrow_end_px[0]), int(arrow_end_px[1])) if arrow_end_px is not None else None,
                }
            )

        if not labels:
            return None, [], [], []

        crop = canvas[crop_ymin : crop_ymax + 1, crop_xmin : crop_xmax + 1].copy()
        cv2.putText(
            crop,
            "Choose the best semantic grasp",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return crop, labels, visible_indices, candidate_meta

    def query_vlm_for_candidates(self, crop_image, labels, candidate_meta):
        ok, buffer = cv2.imencode(".jpg", crop_image)
        if not ok:
            return None

        candidate_lines = []
        for item in candidate_meta:
            candidate_lines.append(
                f"{item['label']}: width={item['width_m']:.3f}m, depth={item['depth_m']:.3f}m, "
                f"raw_score={item['raw_score']:.3f}, center_offset={item['center_offset_px']:.1f}px, "
                f"width_ok={'yes' if item['width_ok'] else 'no'}"
            )
        candidate_text = "\n".join(candidate_lines)

        prompt = (
            "你是货架商品拣选机器人的抓取评审器。\n"
            f"当前任务：{self.latest_target.get('command', '')}\n"
            f"目标商品：{self.latest_target.get('target', '')}\n"
            f"图中已经标出了同一目标的抓取候选 {', '.join(labels)}。\n"
            "机器人执行约束：\n"
            f"- 最大夹爪开口约 {self.gripper_width_max:.3f} m\n"
            f"- 若候选宽度超过 {max(0.0, self.gripper_width_max - self.gripper_width_margin):.3f} m，则该候选风险较高\n"
            "- 当前任务是从货架中取出商品，优先正对货架、稳定包裹主体区域、避免抓瓶口/薄边/遮挡边\n"
            "候选的结构化信息如下：\n"
            f"{candidate_text}\n"
            "请结合货架取物任务，优先考虑：\n"
            "1. 抓取稳定且夹爪宽度可包住目标；\n"
            "2. 更容易把商品从货架里抽出来；\n"
            "3. 尽量抓瓶身/盒侧面，避免抓瓶口、薄边或遮挡边；\n"
            "4. 候选中心越接近目标主体中心通常越优；\n"
            "5. 如果都一般，也选最稳且最符合夹爪约束的。\n"
            "严格返回 JSON：\n"
            '{"best_index":1,"semantic_scores":[0.9,0.4,0.2],"preferred_region":"bottle_body","avoid_region":"cap_or_occluded_edge","grasp_side":"front","reason":"..." }\n'
            "注意：best_index 是从 1 开始编号；semantic_scores 顺序必须与候选标签顺序一致；只返回 JSON。"
        )

        base64_image = base64.b64encode(buffer).decode("utf-8")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.0,
            )
        except Exception as exc:
            rospy.logerr("Semantic reranker VLM request failed: %s", exc)
            return None

        result_text = response.choices[0].message.content
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

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

    def safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0

    def draw_text_box(self, image, text, origin, fg_color, bg_color, font_scale=0.55, thickness=2):
        x, y = int(origin[0]), int(origin[1])
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x0 = max(0, x - 2)
        y0 = max(0, y - text_h - baseline - 6)
        x1 = min(image.shape[1] - 1, x + text_w + 6)
        y1 = min(image.shape[0] - 1, y + 4)
        cv2.rectangle(image, (x0, y0), (x1, y1), bg_color, -1)
        cv2.putText(
            image,
            text,
            (x + 2, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            fg_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def visualize_candidate_decision(self, crop_image, object_id, candidate_meta, result_payload):
        if crop_image is None:
            return

        vis = crop_image.copy()
        best_candidate = result_payload.get("best_candidate")
        best_label = None
        for item in candidate_meta:
            if int(item["candidate_index"]) == int(best_candidate):
                best_label = item["label"]
                break

        footer_y = vis.shape[0] - 12
        summary = f"Best: {best_label or '?'} | region={result_payload.get('preferred_region', 'unknown')}"
        self.draw_text_box(vis, summary, (12, footer_y), (0, 255, 255), (0, 0, 0), font_scale=0.55, thickness=2)

        if self.visualize_candidate_crop:
            cv2.imshow("Semantic Candidate Rerank", vis)
            cv2.waitKey(1)

        if self.save_candidate_crop:
            os.makedirs(self.candidate_debug_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            stem = f"semantic_object_{object_id}_{timestamp}"
            filename = f"{stem}.png"
            cv2.imwrite(os.path.join(self.candidate_debug_dir, filename), vis)
            if self.save_candidate_json:
                payload = {
                    "object_id": int(object_id),
                    "candidate_meta": candidate_meta,
                    "result_payload": result_payload,
                }
                with open(
                    os.path.join(self.candidate_debug_dir, f"{stem}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

    def visualize_global_overview_canvas(self, debug_payload):
        ref_image = self.get_reference_image()
        if ref_image is None:
            return

        overview_items = debug_payload.get("overview_items", [])
        if not overview_items:
            return

        vis = ref_image.copy()
        overlay = vis.copy()
        palette = [
            (0, 255, 255),
            (0, 200, 0),
            (0, 128, 255),
            (255, 255, 0),
            (255, 128, 0),
            (255, 0, 255),
        ]

        for rank, item in enumerate(sorted(overview_items, key=lambda x: x.get("object_id", 0))):
            color = palette[rank % len(palette)]
            dark_outline = (0, 0, 0)
            bbox = item.get("bbox", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), dark_outline, 4)
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)

            best_candidate = int(item.get("best_candidate", -1))
            for candidate in item.get("candidates", []):
                center_px = candidate.get("center_px")
                arrow_end_px = candidate.get("arrow_end_px")
                if center_px is None:
                    continue

                candidate_index = int(candidate.get("candidate_index", -1))
                is_best = candidate_index == best_candidate
                thickness = 4 if is_best else 1
                radius = 7 if is_best else 2
                draw_color = color if candidate.get("width_ok", True) else (170, 170, 170)
                target_img = vis if is_best else overlay

                cv2.circle(target_img, tuple(center_px), radius + (2 if is_best else 1), dark_outline, -1)
                cv2.circle(target_img, tuple(center_px), radius, draw_color, -1)
                if arrow_end_px is not None:
                    cv2.arrowedLine(
                        target_img,
                        tuple(center_px),
                        tuple(arrow_end_px),
                        dark_outline,
                        thickness + (2 if is_best else 1),
                        tipLength=0.2,
                    )
                    cv2.arrowedLine(
                        target_img,
                        tuple(center_px),
                        tuple(arrow_end_px),
                        draw_color,
                        thickness,
                        tipLength=0.2,
                    )
                if is_best:
                    self.draw_text_box(
                        vis,
                        item.get("best_label", "best"),
                        (center_px[0] + 6, max(center_px[1] - 4, 18)),
                        draw_color,
                        dark_outline,
                        font_scale=0.45,
                        thickness=1,
                    )

            label = f"O{item.get('object_id', '?')}"
            self.draw_text_box(
                vis,
                label,
                (xmin, max(ymin - 2, 18)),
                color,
                dark_outline,
                font_scale=0.55,
                thickness=2,
            )

        vis = cv2.addWeighted(overlay, 0.65, vis, 0.55, 0.0)

        if self.visualize_global_overview:
            cv2.imshow("Semantic Rerank Overview", vis)
            cv2.waitKey(1)

        if self.save_global_overview:
            os.makedirs(self.global_overview_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_overview_{timestamp}.png"
            cv2.imwrite(os.path.join(self.global_overview_dir, filename), vis)


if __name__ == "__main__":
    node = SemanticRerankerNode()
    rospy.spin()
