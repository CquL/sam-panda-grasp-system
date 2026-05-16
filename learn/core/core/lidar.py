#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 在这里作为包，不能直接导入import camera，Python会去顶级目录找，在启动文件为所在的路径为目录找，即使camera和lidar在一个文件夹下也找不到
from . import camera 
# 或者 
from .camera import open_camera

def scan():
    print("雷达扫描中...")
    camera.open_camera() # 联动摄像头


scan()