import numpy as np
import math

class GraspGenerator:
    def __init__(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
        """初始化相机内参"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def compute_grasp_pose(self, depth_image, mask):
        """
        根据深度图和掩码计算目标的三维中心点及抓取姿态
        :return: (x, y, z, qx, qy, qz, qw) 或 None
        """
        # 提取掩码区域的深度值
        target_depths = depth_image[mask > 0]
        if len(target_depths) == 0:
            return None

        # 计算平均深度 (单位假设为毫米，转为米)
        avg_depth = np.mean(target_depths) / 1000.0

        # 计算掩码的二维质心 (u, v)
        ys, xs = np.where(mask > 0)
        u = np.mean(xs)
        v = np.mean(ys)

        # 像素坐标到相机坐标系 (3D) 的转换
        z = avg_depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # 这里使用一个简单的垂直向下抓取姿态 (假数据，你可以替换为 GraspNet 结果)
        qx, qy, qz, qw = 0.0, 1.0, 0.0, 0.0

        return (x, y, z, qx, qy, qz, qw)