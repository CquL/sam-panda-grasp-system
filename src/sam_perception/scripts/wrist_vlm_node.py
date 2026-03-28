#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import rospy
import cv2
import base64
import ast
import re
from openai import OpenAI
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

class WristVLMNode:
    def __init__(self):
        rospy.init_node('wrist_vlm_node')
        self.bridge = CvBridge()
        self.latest_image = None
        
        # 💥 加入以下代码：强制屏蔽系统代理，防止 httpx 崩溃
        proxy_vars = ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]

        # 屏蔽完代理后，再初始化 client
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 订阅手腕相机画面
        rospy.Subscriber('/wrist_camera/color/image_raw', Image, self.image_cb, queue_size=1)
        # 订阅 demo.py 发来的触发信号
        rospy.Subscriber('/wrist_vlm/trigger', String, self.trigger_cb, queue_size=1)
        # 往外发目标框
        self.pub_bbox = rospy.Publisher('/wrist_vlm/bbox', Float32MultiArray, queue_size=1)
        
        rospy.loginfo("👁️ 手腕 VLM 感知节点已就绪，等待触发...")

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def trigger_cb(self, msg):
        if self.latest_image is None:
            self.pub_bbox.publish(Float32MultiArray(data=[])) # 发空数组代表失败
            return
            
        rospy.loginfo("📸 收到触发信号，召唤千问提取目标框...")
        img_cv = self.latest_image.copy()
        _, buffer = cv2.imencode('.jpg', img_cv)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        prompt = "画面正中间有一个即将被抓取的商品。请精准框出它，严格返回二维数组格式如 [[xmin, ymin, xmax, ymax]]，绝对不要返回任何其他解释性文字。"
        
        try:
            res = self.client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}, {"type": "text", "text": prompt}]}],
                temperature=0.0
            )
            result_text = res.choices[0].message.content
            match = re.search(r'\[\s*\[.*?\]\s*\]', result_text, re.DOTALL)
            
            if match:
                boxes = ast.literal_eval(match.group(0))
                bbox_msg = Float32MultiArray(data=boxes[0])
                self.pub_bbox.publish(bbox_msg)
                rospy.loginfo(f"✅ VLM 返回边框: {boxes[0]}")
            else:
                self.pub_bbox.publish(Float32MultiArray(data=[]))
        except Exception as e:
            rospy.logerr(f"VLM 请求失败: {e}")
            self.pub_bbox.publish(Float32MultiArray(data=[]))

if __name__ == '__main__':
    node = WristVLMNode()
    rospy.spin()
