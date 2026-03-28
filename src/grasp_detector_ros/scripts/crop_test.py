#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import message_filters
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from detection_msgs.msg import BoundingBoxes
import time

class PointCloudCropper:
    def __init__(self):
        rospy.init_node('point_cloud_cropper', anonymous=True)
        
        # 1. 订阅者 setup
        self.sub_yolo = message_filters.Subscriber('/yolov5/detections', BoundingBoxes)
        self.sub_pc = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        
        # 2. 同步器 (Slop=1.0 保证仿真环境下能同步上)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_yolo, self.sub_pc], queue_size=10, slop=1.0)
        self.ts.registerCallback(self.callback)
        
        # 3. 数据缓冲区 (用于线程间通信)
        self.latest_cloud_data = None
        self.has_new_data = False
        
        rospy.loginfo("正在等待 YOLO 和 点云 的同步数据...")

    def callback(self, yolo_msg, pc_msg):
        """
        这个函数在 ROS 子线程中运行。
        它的任务只是：计算 -> 存数据。
        绝对不要在这里调用 self.vis.update_geometry()！
        """
        # 1. 找置信度最高的目标
        target_box = None
        max_score = 0
        for box in yolo_msg.bounding_boxes:
            if box.probability > max_score:
                max_score = box.probability
                target_box = box

        if target_box is None:
            return

        # 2. 解析坐标
        u_min = int(target_box.xmin)
        v_min = int(target_box.ymin)
        u_max = int(target_box.xmax)
        v_max = int(target_box.ymax)

        # 3. 边界保护
        u_min = max(0, u_min)
        v_min = max(0, v_min)
        u_max = min(pc_msg.width, u_max)
        v_max = min(pc_msg.height, v_max)

        # 4. 提取点云
        gen = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=False)
        try:
            pc_array = np.array(list(gen)).reshape(pc_msg.height, pc_msg.width, 3)
        except ValueError:
            return

        # 5. Numpy 切片抠图 (核心步骤)
        cropped_cloud = pc_array[v_min:v_max, u_min:u_max, :]
        flat_cloud = cropped_cloud.reshape(-1, 3)
        # 移除 NaN 无效点
        flat_cloud = flat_cloud[~np.isnan(flat_cloud).any(axis=1)]

        if len(flat_cloud) < 10:
            return

        # 6. 【关键】只保存数据，不画图
        self.latest_cloud_data = flat_cloud
        self.has_new_data = True

        
        # ================= 新增：计算并打印位置 =================
        # 计算点云的几何中心 (Centroid) -> 即物体的 3D 坐标
        # 打印一下日志证明还在工作
        # rospy.loginfo(f"锁定目标: {target_box.Class} | 点数: {len(flat_cloud)}")
        centroid = np.mean(flat_cloud, axis=0)
        x, y, z = centroid[0], centroid[1], centroid[2]
        
        # 获取 2D 框的中心
        center_u = int((u_min + u_max) / 2)
        center_v = int((v_min + v_max) / 2)

        print("-" * 30)
        rospy.loginfo(f"【识别成功】 目标: {target_box.Class}")
        rospy.loginfo(f"  > 2D 像素位置: (列u={center_u}, 行v={center_v})")
        rospy.loginfo(f"  > 3D 相机坐标: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
        rospy.loginfo(f"  > 距离相机: {z:.2f} 米")
        print("-" * 30)

    def run_visualization(self):
        """
        这个函数在主线程运行。
        它的任务是：死循环检查数据 -> 刷新 Open3D 窗口。
        """
        # 初始化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Cropped Object (Main Thread)", width=800, height=600)
        
        # 设置渲染选项：深灰色背景，点大一点
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 3.0

        pcd = o3d.geometry.PointCloud()
        is_first_frame = True

        # 主循环
        while not rospy.is_shutdown():
            # 检查是否有新数据送过来
            if self.has_new_data and self.latest_cloud_data is not None:
                
                # 更新几何体
                pcd.points = o3d.utility.Vector3dVector(self.latest_cloud_data)
                pcd.paint_uniform_color([1, 0, 0]) # 染成红色

                if is_first_frame:
                    vis.add_geometry(pcd)
                    vis.reset_view_point(True) # 第一次自动对焦
                    is_first_frame = False
                else:
                    vis.update_geometry(pcd)
                
                # 标记数据已处理
                self.has_new_data = False
                
                # 触发渲染
                vis.poll_events()
                vis.update_renderer()
            else:
                # 即使没有新数据，也要刷新窗口响应鼠标操作
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01) # 稍微休息一下，避免占满 CPU

        vis.destroy_window()

if __name__ == '__main__':
    try:
        cropper = PointCloudCropper()
        # 启动主线程的可视化循环 (替代了原来的 rospy.spin)
        cropper.run_visualization()
    except rospy.ROSInterruptException:
        pass