import numpy as np
import math

class RoboticArmRRTPlanner:
    def __init__(self, start_joints, goal_joints, joint_limits, obstacles_3d):
        """
        初始化规划器
        :param start_joints: 起点 7 个关节角度 (7D)
        :param goal_joints: 终点 7 个关节角度 (7D)
        :param joint_limits: 每个关节的物理限位 [(min, max), ...]
        :param obstacles_3d: 3D 空间中的障碍物数据 (如货架边界)
        """
        self.start = np.array(start_joints)
        self.goal = np.array(goal_joints)
        self.limits = joint_limits
        self.obstacles = obstacles_3d
        
        # RRT 的节点树：存储所有的 7D 状态和它们的父节点索引
        self.vertices = [self.start]
        self.edges = [-1] # 起点没有父节点
        
        # 算法超参数
        self.step_size = 0.1 # 每次在 7D 空间延伸的步长（弧度）
        self.max_iter = 10000 # 最大尝试撒点次数

    # ==========================================
    # 模块 1：7D 随机采样 (Random Sampling)
    # ==========================================
    def sample_random_state(self):
        """在 7 维关节空间内随机撒一个点（类似你图片里找的新点）"""
        # 有 10% 的概率直接朝着终点撒点（贪心启发，加速收敛）
        if np.random.rand() < 0.1:
            return self.goal
            
        random_joints = []
        for limit in self.limits:
            random_joints.append(np.random.uniform(limit[0], limit[1]))
        return np.array(random_joints)

    # ==========================================
    # 模块 2：碰撞检测核心黑盒 (Collision Checking)
    # ==========================================
    def is_state_valid(self, joints_7d):
        """
        【这是整个算法最耗时的灵魂步骤，对应 MoveIt 中的 FCL 库】
        判断一个 7 维关节角，在 3D 物理世界中是否会撞到东西？
        """
        # 步骤 2.1: 正向运动学 (Forward Kinematics)
        # 将 7 个角度代入机器人的 URDF 几何模型，计算出机械臂各个圆柱体连杆在 3D 空间中的 (x, y, z)
        arm_3d_links = self.compute_forward_kinematics(joints_7d)
        
        # 步骤 2.2: 3D 几何干涉检查
        # 遍历机械臂的每一节连杆，看它是否和我们的货架（obstacles_3d）相交
        for link in arm_3d_links:
            if self.check_intersection_3d(link, self.obstacles):
                return False # 发生碰撞，这个 7D 点是无效的！
                
        return True # 完美，没有干涉，是个合法的状态

    # ==========================================
    # 模块 3：核心 RRT 生长逻辑 (Tree Expansion)
    # ==========================================
    def plan(self):
        """执行 RRT 规划主循环"""
        for i in range(self.max_iter):
            # 1. 撒一个随机 7D 点
            q_rand = self.sample_random_state()
            
            # 2. 找到树中距离这个随机点最近的节点 (算 7 维欧氏距离)
            distances = [np.linalg.norm(v - q_rand) for v in self.vertices]
            nearest_idx = np.argmin(distances)
            q_near = self.vertices[nearest_idx]
            
            # 3. 往随机点的方向走一小步 (步长为 step_size)
            direction = q_rand - q_near
            length = np.linalg.norm(direction)
            if length == 0: continue
            direction = direction / length
            q_new = q_near + direction * min(self.step_size, length)
            
            # 4. 关键验证：新生成的这个节点，以及移动的过程会撞车吗？
            if self.is_state_valid(q_new):
                # 如果安全，把它加入树中
                self.vertices.append(q_new)
                self.edges.append(nearest_idx)
                
                # 5. 检查是否已经抵达终点附近
                if np.linalg.norm(q_new - self.goal) < self.step_size:
                    print(f"🎉 规划成功！迭代次数: {i}")
                    return self.extract_path(len(self.vertices) - 1)
                    
        print("❌ 规划失败：超出最大迭代次数")
        return None

    def extract_path(self, end_idx):
        """顺藤摸瓜，从终点回溯到起点，提取轨迹"""
        path = []
        curr_idx = end_idx
        while curr_idx != -1:
            path.append(self.vertices[curr_idx])
            curr_idx = self.edges[curr_idx]
        return path[::-1] # 逆序输出，变成 起点 -> 终点

    # (省略底层数学实现的占位函数，供理解概念)
    def compute_forward_kinematics(self, joints): pass
    def check_intersection_3d(self, link, obstacles): pass