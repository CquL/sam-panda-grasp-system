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
from openai import OpenAI
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from sam_perception.ros_image_compat import image_msg_to_numpy, numpy_to_image_msg


DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


class GlobalVLMTargetNode:
    def __init__(self):
        rospy.init_node("global_vlm_target_node")
        self.latest_image = None
        self.last_result_text = ""

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
        self.vlm_timeout_sec = float(rospy.get_param("~vlm_timeout_sec", 20.0))
        self.vlm_max_retries = max(0, int(rospy.get_param("~vlm_max_retries", 2)))
        self.allow_partial_targets = bool(rospy.get_param("~allow_partial_targets", False))

        proxy_vars = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        for var in proxy_vars:
            os.environ.pop(var, None)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.vlm_timeout_sec,
            max_retries=0,
        )

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
            self.latest_image = image_msg_to_numpy(msg, "bgr8")
        except Exception as exc:
            self.latest_image = None
            rospy.logerr_throttle(
                5.0,
                "Global VLM image conversion failed for %s (encoding=%s): %s",
                self.image_topic,
                getattr(msg, "encoding", ""),
                exc,
            )

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
        expected_targets = self.extract_expected_targets(command)
        result = None
        entries = []
        missing_targets = []
        last_reason = "vlm_parse_failed"
        for attempt_idx in range(self.vlm_max_retries + 1):
            result = self.query_vlm(
                command,
                self.latest_image,
                expected_targets=expected_targets,
                attempt_idx=attempt_idx,
                missing_targets=missing_targets,
            )
            if result is None:
                last_reason = "vlm_parse_failed"
                continue

            entries = self.normalize_entries(result, self.latest_image.shape[1], self.latest_image.shape[0], command)
            missing_targets = self.find_missing_expected_targets(entries, expected_targets)
            if not missing_targets:
                break

            last_reason = "missing_expected_targets"
            rospy.logwarn(
                "Global VLM returned incomplete target set: expected=%s got=%s missing=%s attempt=%d/%d",
                ",".join([item["name"] for item in expected_targets]),
                ",".join([entry["target"] for entry in entries]) if entries else "<none>",
                ",".join([item["name"] for item in missing_targets]),
                attempt_idx + 1,
                self.vlm_max_retries + 1,
            )

        if result is None:
            preview = (self.last_result_text or "").replace("\n", " ").strip()
            if len(preview) > 240:
                preview = preview[:240] + "..."
            rospy.logwarn(
                "Global VLM parse failed for command=%s, raw_preview=%s",
                command,
                preview if preview else "<empty>",
            )
            self.publish_failure(command, last_reason)
            self.pub_bbox.publish(Float32MultiArray(data=[]))
            return

        if missing_targets and not self.allow_partial_targets:
            rospy.logwarn(
                "Global VLM aborted partial task dispatch: command=%s expected=%s got=%s missing=%s",
                command,
                ",".join([item["name"] for item in expected_targets]),
                ",".join([entry["target"] for entry in entries]) if entries else "<none>",
                ",".join([item["name"] for item in missing_targets]),
            )
            self.publish_failure(
                command,
                "missing_expected_targets:" + ",".join([item["name"] for item in missing_targets]),
            )
            self.pub_bbox.publish(Float32MultiArray(data=[]))
            return

        boxes = [entry["bbox"] for entry in entries]
        if len(boxes) == 0:
            rospy.logwarn("Global VLM produced zero valid bboxes after sanitize. result_keys=%s", list(result.keys()))
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
            "expected_targets": expected_targets,
        }

        flat_coords = [float(v) for box in boxes for v in box]
        self.pub_bbox.publish(Float32MultiArray(data=flat_coords))
        self.pub_result.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        try:
            snapshot_msg = numpy_to_image_msg(self.latest_image, "bgr8", stamp=rospy.Time.now())
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

    def normalize_target_name(self, text):
        text = str(text or "").strip().lower()
        alias_groups = [
            ("白砂糖", ["白砂糖", "砂糖", "糖", "sugar"]),
            ("可乐", ["可乐", "coke", "cola", "coca"]),
            ("啤酒", ["啤酒", "beer"]),
            ("番茄罐头", ["番茄罐头", "番茄", "tomato"]),
        ]
        for canonical, aliases in alias_groups:
            if any(alias in text for alias in aliases):
                return canonical
        cleanup_patterns = [
            r"^抓",
            r"^拿",
            r"^取",
            r"^(一个|一瓶|一罐|一盒|一袋|那罐|那个|这罐|这个)",
            r"(最右边|最左边|右边|左边|中间|上面|下面|上层|下层|那罐|那个|这罐|这个)",
        ]
        cleaned = str(text)
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip() or str(text).strip()

    def extract_expected_targets(self, command):
        text = str(command or "").strip()
        if not text:
            return []
        normalized = text.replace("，", "、").replace(",", "、").replace("和", "、").replace("及", "、")
        parts = [p.strip() for p in re.split(r"[、;；]+", normalized) if p.strip()]
        targets = []
        seen = set()
        for part in parts:
            if not any(token in part.lower() for token in ["可乐", "啤酒", "白砂糖", "砂糖", "糖", "番茄", "coke", "cola", "beer", "sugar", "tomato"]):
                continue
            name = self.normalize_target_name(part)
            if not name or name in seen:
                continue
            seen.add(name)
            targets.append({"name": name, "phrase": part})
        return targets[: self.max_boxes]

    def target_names_match(self, expected_name, actual_name):
        return self.normalize_target_name(expected_name) == self.normalize_target_name(actual_name)

    def find_missing_expected_targets(self, entries, expected_targets):
        if not expected_targets:
            return []
        unused = list(entries or [])
        missing = []
        for expected in expected_targets:
            match_idx = None
            for idx, entry in enumerate(unused):
                if self.target_names_match(expected["name"], entry.get("target", "")):
                    match_idx = idx
                    break
            if match_idx is None:
                missing.append(expected)
            else:
                unused.pop(match_idx)
        return missing

    def query_vlm(self, command, image_bgr, expected_targets=None, attempt_idx=0, missing_targets=None):
        expected_targets = expected_targets or []
        missing_targets = missing_targets or []
        expected_text = "\n".join(
            [
                f"- name={item['name']}, user_phrase={item.get('phrase', item['name'])}"
                for item in expected_targets
            ]
        )
        retry_hint = ""
        if missing_targets:
            retry_hint = (
                "\n上一轮漏掉了这些目标，必须重新检查并补全："
                + "、".join([item["name"] for item in missing_targets])
                + "\n"
            )
        prompt = (
            "你是货架商品拣选机器人的视觉助手。\n"
            f"用户任务：{command}\n"
            + (f"任务中解析出的必需目标如下，输出 name 必须与这里完全一致：\n{expected_text}\n" if expected_targets else "")
            + retry_hint +
            "请先理解任务中提到的每一种商品以及数量，再在当前货架图像中找到对应实例。\n"
            "输出严格 JSON，优先使用这个格式：\n"
            '{"targets":[{"name":"商品名","bbox":[xmin,ymin,xmax,ymax]}],"confidence":0.0}\n'
            "要求：\n"
            "1. targets 中每一项只对应一个具体实例；如果必需目标有 3 个，就尽量输出 3 项；\n"
            "2. 不要把多个商品合并，也不要用白砂糖/糖盒代替可乐、不要用可乐代替啤酒；\n"
            "3. bbox 按图像绝对像素坐标输出；\n"
            "4. 如果用户说最右边/最左边，应在同类商品中按图像 x 方向选择对应实例；\n"
            "5. 如果找不到，对应商品就不要编造 bbox；\n"
            "6. confidence 取 0 到 1；\n"
            "7. 只返回 JSON，不要解释。"
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
                timeout=self.vlm_timeout_sec,
            )
        except Exception as exc:
            rospy.logerr("Global VLM request failed: %s", exc)
            return None

        result_text = response.choices[0].message.content
        self.last_result_text = str(result_text)
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
        if result_text is None:
            return None

        text = str(result_text).strip()
        if not text:
            return None

        # 1) direct parse
        try:
            parsed = json.loads(text)
            normalized = self.normalize_parsed_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            pass

        # 2) parse fenced code blocks first
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in code_blocks:
            block = block.strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
                normalized = self.normalize_parsed_payload(parsed)
                if normalized is not None:
                    return normalized
            except Exception:
                pass

        # 3) scan for any decodable JSON object / array inside mixed text
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
            except Exception:
                continue
            normalized = self.normalize_parsed_payload(parsed)
            if normalized is not None:
                return normalized
        return None

    def normalize_parsed_payload(self, parsed):
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return {"bboxes": [], "confidence": 0.0}
            # [[xmin,ymin,xmax,ymax], ...]
            if all(isinstance(item, (list, tuple)) and len(item) == 4 for item in parsed):
                return {"bboxes": parsed, "confidence": 0.0}
            # [{"name": "...", "bbox": [...]}, ...]
            if all(isinstance(item, dict) for item in parsed):
                return {"targets": parsed, "confidence": 0.0}
        return None

    def sanitize_boxes(self, boxes, image_w, image_h):
        sanitized = []
        if not isinstance(boxes, list):
            return sanitized

        for box in boxes[: self.max_boxes]:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                vals = [float(v) for v in box]
            except Exception:
                continue

            # Handle normalized xyxy in [0, 1].
            if all(-1e-6 <= v <= 1.0 + 1e-6 for v in vals):
                vals = [vals[0] * image_w, vals[1] * image_h, vals[2] * image_w, vals[3] * image_h]

            xmin, ymin, xmax, ymax = vals
            # Fallback: if model returned xywh style.
            if xmax <= xmin or ymax <= ymin:
                x, y, w, h = vals
                if w > 0 and h > 0:
                    xmin, ymin, xmax, ymax = x, y, x + w, y + h

            xmin, ymin, xmax, ymax = [int(round(v)) for v in [xmin, ymin, xmax, ymax]]

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
                if isinstance(bbox, dict):
                    # Accept bbox dict outputs with common key names.
                    keys = [("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")]
                    for kx1, ky1, kx2, ky2 in keys:
                        if all(k in bbox for k in [kx1, ky1, kx2, ky2]):
                            bbox = [bbox[kx1], bbox[ky1], bbox[kx2], bbox[ky2]]
                            break
                sanitized = self.sanitize_boxes([bbox], image_w, image_h)
                if not sanitized:
                    continue
                entries.append({"target": target_name, "bbox": sanitized[0]})

        if entries:
            return entries

        bbox = result.get("bbox")
        if bbox is not None:
            if isinstance(bbox, dict):
                keys = [
                    ("xmin", "ymin", "xmax", "ymax"),
                    ("x1", "y1", "x2", "y2"),
                    ("left", "top", "right", "bottom"),
                ]
                for kx1, ky1, kx2, ky2 in keys:
                    if all(k in bbox for k in [kx1, ky1, kx2, ky2]):
                        bbox = [bbox[kx1], bbox[ky1], bbox[kx2], bbox[ky2]]
                        break
            target_name = str(result.get("name", result.get("target", default_target))).strip() or default_target
            boxes = self.sanitize_boxes([bbox], image_w, image_h)
            for box in boxes:
                entries.append({"target": target_name, "bbox": box})

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
