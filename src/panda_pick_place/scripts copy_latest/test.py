class GazeboCubeManager:
    """负责在 Gazebo 中生成 / 删除目标物体（默认 cube）"""

    def __init__(self,
                 model_name="cube",
                 pkg_name="panda_pick_place",
                 sdf_rel_path="models/cube.sdf"):
        # 这里的三个参数支持外部传入（我们在 demo.py 里传）
        self.model_name = model_name
        self.pkg_name = pkg_name
        self.sdf_rel_path = sdf_rel_path

    def _load_sdf(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path(self.pkg_name)
        sdf_path = os.path.join(pkg_path, self.sdf_rel_path)
        rospy.loginfo(f"[GazeboCubeManager] SDF 路径: {sdf_path}")
        with open(sdf_path, "r") as f:
            sdf_xml = f.read()
        return sdf_xml

    def spawn_cube(self, x, y, z, reference_frame="world"):
        """在 Gazebo 中生成当前配置的物体（model_name 对应 cube 或 cylinder）"""
        rospy.loginfo("[GazeboCubeManager] 等待 /gazebo/spawn_sdf_model 服务...")
        ...
        # 这一段原来的代码保持不动，只是把日志稍微改一下
        req = SpawnModelRequest()
        req.model_name = self.model_name
        req.model_xml = sdf_xml
        ...
        try:
            resp = spawn_srv(req)
            if resp.success:
                rospy.loginfo(
                    f"[GazeboCubeManager] ✓ 已在 Gazebo 中生成模型 {self.model_name}"
                )
                return True
            else:
                rospy.logwarn(
                    f"[GazeboCubeManager] 生成 {self.model_name} 失败，可能已存在: "
                    f"{resp.status_message}"
                )
                return False
        ...
