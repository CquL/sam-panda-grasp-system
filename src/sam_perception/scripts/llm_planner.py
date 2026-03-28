#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# ==============================================================================
# 【核心修复 1】防止 ROS 系统路径干扰 Conda 环境
# ==============================================================================
libffi_preload = os.environ.get("LIBFFI_PRELOAD")
if libffi_preload:
    os.environ['LD_PRELOAD'] = libffi_preload
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from openai import OpenAI
import re
import base64
import ast  # 💥 新增：用于安全解析 Python 的二维列表字符串

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

class QwenPlannerNode:
    def __init__(self):
        rospy.init_node('llm_planner_node')
        self.bridge = CvBridge()
        self.latest_image = None
        
        # 屏蔽系统代理
        proxy_vars = ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]

        if not DASHSCOPE_API_KEY:
            raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        self.pub_bbox = rospy.Publisher('/sam/prompt_bbox', Float32MultiArray, queue_size=10)

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass

    def run(self):
        rospy.loginfo("🤖 千问视觉指挥官 (多目标升级版) 已上线，准备接收指令...")
        
        while not rospy.is_shutdown():
            try:
                try:
                    user_cmd = input("\n请输入你的指令 (例如: 把可乐和啤酒都拿出来): ")
                except NameError:
                    user_cmd = raw_input("\n请输入你的指令 (例如: 把可乐和啤酒都拿出来): ")
                    
                if not user_cmd.strip() or self.latest_image is None:
                    continue
                
                img_cv = self.latest_image.copy()
                h, w = img_cv.shape[:2] 
                
                _, buffer = cv2.imencode('.jpg', img_cv)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                # 💥 终极 Prompt：要求返回严格的二维列表 (List of Lists)
                prompt = (
                    f"用户指令是：'{user_cmd}'。\n"
                    "请在图像中找到符合指令的所有物体，并返回它们的边界框坐标。\n"
                    "返回格式必须严格是嵌套的二维列表，例如：[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2]]。\n"
                    "这些数字是图像的绝对像素坐标。其中 xmin 和 xmax 是横向的宽(左到右)，ymin 和 ymax 是纵向的高(上到下)。\n"
                    "如果只找到一个物体，也请使用二维列表格式返回，例如 [[xmin, ymin, xmax, ymax]]。\n"
                    "请只返回这个二维列表，绝对不要返回任何其他多余的解释文字！"
                )
                
                rospy.loginfo(f"正在向千问发送多目标检索请求: {user_cmd} ...")
                
                response = self.client.chat.completions.create(
                    model="qwen-vl-max",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    temperature=0.0 
                )
                
                result_text = response.choices[0].message.content
                rospy.loginfo(f"千问原始回复: {result_text}")
                
                # 💥 核心解析逻辑：提取并解析二维列表
                match = re.search(r'\[\s*\[.*?\]\s*\]', result_text, re.DOTALL)
                if match:
                    try:
                        # 安全地将字符串转化为 Python 的 List 对象
                        boxes = ast.literal_eval(match.group(0))
                        
                        if isinstance(boxes, list) and len(boxes) > 0 and isinstance(boxes[0], list):
                            rospy.loginfo(f"✅ 成功解析出 {len(boxes)} 个目标框！")
                            
                            debug_img = img_cv.copy()
                            flat_coords = [] # 展平数组用于 ROS 发布
                            
                            for i, box in enumerate(boxes):
                                if len(box) == 4:
                                    xmin, ymin, xmax, ymax = [int(float(x)) for x in box]
                                    
                                    xmin, ymin = max(0, xmin), max(0, ymin)
                                    xmax, ymax = min(w, xmax), min(h, ymax)
                                    
                                    flat_coords.extend([xmin, ymin, xmax, ymax])
                                    rospy.loginfo(f"  -> 目标 {i+1}: [xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}]")
                                    
                                    # 在图上画出所有框并标号
                                    cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                                    cv2.putText(debug_img, f"Target-{i+1}", (xmin, max(ymin-10, 0)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                                
                            # 📸 保存全家福调试图
                            debug_path = os.path.expanduser("~/llm_debug_result.jpg")
                            cv2.imwrite(debug_path, debug_img)
                            rospy.loginfo(f"📸 包含 {len(boxes)} 个目标的调试图已保存！请去主目录查看。")

                            # 打包成一维数组发给 SAM (例如: [x1,y1,x2,y2, x3,y3,x4,y4])
                            bbox_msg = Float32MultiArray(data=flat_coords)
                            self.pub_bbox.publish(bbox_msg)
                            rospy.loginfo(f"✅ 已将 {len(boxes)} 个目标的 BBox 数据下发给 SAM 节点！\n")
                            
                        else:
                            rospy.logerr("解析出的不是二维列表格式！")
                    except Exception as e:
                        rospy.logerr(f"列表转换异常: {e}")
                else:
                    rospy.logerr(f"未匹配到合法的二维列表格式。回复内容: {result_text}")
                    
            except Exception as e:
                rospy.logerr(f"发生异常: {e}")

if __name__ == '__main__':
    try:
        node = QwenPlannerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
