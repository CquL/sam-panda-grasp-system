#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import base64
import json
import os
import re
import sys

ld_preload = os.environ.get("ANYGRASP_LD_PRELOAD", "").strip()
if ld_preload:
    os.environ["LD_PRELOAD"] = ld_preload
sys.path = [p for p in sys.path if "/usr/lib/python3/dist-packages" not in p]

import cv2
import rospy
from cv_bridge import CvBridge
from openai import OpenAI
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String


DEFAULT_API_KEY = "sk-a85a76a8ada94ef2886ecd43bf0f9e80"


class GlobalVLMTargetNode:
    def __init__(self):
        rospy.init_node("global_vlm_target_node")
        self.bridge = CvBridge()
        self.latest_image = None

        self.command_topic = rospy.get_param("~command_topic", "/vlm/task_command")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/sam/prompt_bbox")
        self.result_topic = rospy.get_param("~result_topic", "/vlm/global_target")
        self.snapshot_topic = rospy.get_param("~snapshot_topic", "/vlm/global_target_image")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.model_name = rospy.get_param("~model", "qwen-vl-max")
        self.base_url = rospy.get_param(
            "~base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key = rospy.get_param("~api_key", os.environ.get("DASHSCOPE_API_KEY", DEFAULT_API_KEY))
        self.max_boxes = max(1, int(rospy.get_param("~max_boxes", 4)))

        proxy_vars = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        for var in proxy_vars:
            os.environ.pop(var, None)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.command_topic, String, self.command_cb, queue_size=1)

        self.pub_bbox = rospy.Publisher(self.bbox_topic, Float32MultiArray, queue_size=1)
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=1)
        self.pub_snapshot = rospy.Publisher(self.snapshot_topic, Image, queue_size=1, latch=True)

        rospy.loginfo(
            "Global VLM target node ready. image_topic=%s command_topic=%s",
            self.image_topic,
            self.command_topic,
        )

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.latest_image = None

    def command_cb(self, msg):
        command = msg.data.strip()
        if not command:
            return

        if not self.is_plausible_task_command(command):
            rospy.logwarn("Global VLM ignored non-task-like command: %s", command)
            return

        if self.latest_image is None:
            rospy.logwarn("Global VLM has no image yet, cannot process command: %s", command)
            self.publish_failure(command, "no_image")
            return

        rospy.loginfo("Global VLM processing command: %s", command)
        result = self.query_vlm(command, self.latest_image)
        if result is None:
            self.publish_failure(command, "vlm_parse_failed")
            self.pub_bbox.publish(Float32MultiArray(data=[]))
            return

        entries = self.normalize_entries(result, self.latest_image.shape[1], self.latest_image.shape[0], command)
        boxes = [entry["bbox"] for entry in entries]
        target_names = [entry["target"] for entry in entries]
        target = "、".join(target_names) if target_names else (str(result.get("target", command)).strip() or command)
        confidence = self.safe_float(result.get("confidence", 0.0))

        payload = {
            "ok": len(boxes) > 0,
            "command": command,
            "target": target,
            "confidence": confidence,
            "bboxes": boxes,
            "entries": entries,
        }

        flat_coords = [float(v) for box in boxes for v in box]
        self.pub_bbox.publish(Float32MultiArray(data=flat_coords))
        self.pub_result.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        try:
            snapshot_msg = self.bridge.cv2_to_imgmsg(self.latest_image, "bgr8")
            snapshot_msg.header.stamp = rospy.Time.now()
            self.pub_snapshot.publish(snapshot_msg)
        except Exception as exc:
            rospy.logwarn("Global VLM snapshot publish failed: %s", exc)
        rospy.loginfo("Global VLM published %d bbox(es) for target=%s", len(boxes), target)

    def is_plausible_task_command(self, command):
        text = str(command).strip()
        if not text:
            return False
        lowered = text.lower()
        action_keywords = ["抓", "拿", "取", "pick", "grasp", "拿出", "取出"]
        code_markers = [
            "np.array",
            "global_step",
            "args.",
            "learning_starts",
            "for _ in",
            "if ",
            " = ",
            "==",
            "import ",
            "def ",
            "return ",
            "#",
        ]
        if any(marker in lowered for marker in code_markers) and not any(k in text for k in action_keywords):
            return False
        if len(text) > 120 and not any(k in text for k in action_keywords):
            return False
        return True

    def query_vlm(self, command, image_bgr):
        prompt = (
            "你是货架商品拣选机器人的视觉助手。\n"
            f"用户任务：{command}\n"
            "请先理解任务中提到的每一种商品以及数量，再在当前货架图像中找到对应实例。\n"
            "输出严格 JSON，优先使用这个格式：\n"
            '{"targets":[{"name":"商品名","bbox":[xmin,ymin,xmax,ymax]}],"confidence":0.0}\n'
            "要求：\n"
            "1. targets 中每一项只对应一个具体实例；\n"
            "2. 如果用户要一个可乐和一个啤酒，就应输出两项，而不是把它们合并；\n"
            "3. bbox 按图像绝对像素坐标输出；\n"
            "4. 如果找不到，对应商品就不要编造 bbox；\n"
            "5. confidence 取 0 到 1；\n"
            "6. 只返回 JSON，不要解释。"
        )

        ok, buffer = cv2.imencode(".jpg", image_bgr)
        if not ok:
            return None

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
            rospy.logerr("Global VLM request failed: %s", exc)
            return None

        result_text = response.choices[0].message.content
        parsed = self.parse_json_result(result_text)
        if parsed is not None:
            return parsed

        # Fallback: if only bboxes are recoverable, at least preserve the task string.
        bbox_match = re.search(r"\[\s*\[.*?\]\s*\]", result_text, re.DOTALL)
        if bbox_match:
            try:
                boxes = ast.literal_eval(bbox_match.group(0))
                return {"target": command, "bboxes": boxes, "confidence": 0.0}
            except Exception:
                return None
        return None

    def parse_json_result(self, result_text):
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def sanitize_boxes(self, boxes, image_w, image_h):
        sanitized = []
        if not isinstance(boxes, list):
            return sanitized

        for box in boxes[: self.max_boxes]:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                xmin, ymin, xmax, ymax = [int(float(v)) for v in box]
            except Exception:
                continue

            xmin = max(0, min(image_w - 1, xmin))
            ymin = max(0, min(image_h - 1, ymin))
            xmax = max(0, min(image_w - 1, xmax))
            ymax = max(0, min(image_h - 1, ymax))
            if xmax <= xmin or ymax <= ymin:
                continue
            sanitized.append([xmin, ymin, xmax, ymax])
        return sanitized

    def normalize_entries(self, result, image_w, image_h, default_target):
        entries = []

        if isinstance(result.get("targets"), list):
            for item in result["targets"]:
                if not isinstance(item, dict):
                    continue
                target_name = str(item.get("name", default_target)).strip() or default_target
                bbox = item.get("bbox")
                sanitized = self.sanitize_boxes([bbox], image_w, image_h)
                if not sanitized:
                    continue
                entries.append({"target": target_name, "bbox": sanitized[0]})

        if entries:
            return entries

        boxes = self.sanitize_boxes(result.get("bboxes", []), image_w, image_h)
        target_name = str(result.get("target", default_target)).strip() or default_target
        for box in boxes:
            entries.append({"target": target_name, "bbox": box})
        return entries

    def safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0

    def publish_failure(self, command, reason):
        payload = {
            "ok": False,
            "command": command,
            "target": command,
            "confidence": 0.0,
            "bboxes": [],
            "reason": reason,
        }
        self.pub_result.publish(String(data=json.dumps(payload, ensure_ascii=False)))


if __name__ == "__main__":
    node = GlobalVLMTargetNode()
    rospy.spin()
