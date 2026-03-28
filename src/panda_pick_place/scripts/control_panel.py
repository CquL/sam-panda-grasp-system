#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
import tkinter as tk

def send_command(cmd):
    """向后台发送控制指令"""
    pub.publish(cmd)
    rospy.loginfo(f"🚀 已向后台发送干预指令: {cmd}")

def on_closing():
    """关闭窗口时的操作"""
    send_command("quit")
    root.destroy()
    rospy.signal_shutdown("用户关闭了控制面板")

if __name__ == '__main__':
    # 初始化控制面板节点
    rospy.init_node('demo_control_panel', anonymous=True)
    
    # 创建一个发布者，专门往 /demo/command 话题发指令
    pub = rospy.Publisher('/demo/command', String, queue_size=10)

    # 创建 Tkinter GUI 窗口
    root = tk.Tk()
    # 💥 修复：去掉了 Emoji，防止 RenderAddGlyphs 崩溃
    root.title("机械臂控制台") 
    root.geometry("300x180")
    root.attributes('-topmost', True)  # 置顶窗口

    # 💥 修复：使用 Linux 最通用的 sans-serif 字体
    tk.Label(root, text="千问视觉抓取 - 状态干预", font=("sans-serif", 12, "bold")).pack(pady=15)

    # 紧急停止 & 回 Home 按钮
    tk.Button(root, text="紧急中止 & 强制回 Home", bg="red", fg="white", font=("sans-serif", 11, "bold"),
              command=lambda: send_command("stop")).pack(pady=5, fill=tk.X, padx=30, ipady=5)

    # 退出节点按钮
    tk.Button(root, text="完全退出", bg="#555555", fg="white", font=("sans-serif", 10),
              command=on_closing).pack(pady=10, fill=tk.X, padx=60)

    # 绑定窗口关闭事件
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    rospy.loginfo("✅ 控制面板已启动，等待你点击...")
    root.mainloop()