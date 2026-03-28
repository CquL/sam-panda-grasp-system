#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import rospy
import rospkg
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import (
    SpawnModel,
    SpawnModelRequest,
    DeleteModel,
    DeleteModelRequest,
)
from gazebo_ros_link_attacher.srv import Attach, AttachRequest


class GazeboCubeManager:
    """负责在 Gazebo 中生成 / 删除 cube 模型"""
    """负责在 Gazebo 中生成 / 删除目标物体（默认 cube）"""
    def __init__(self,
                 model_name="cylinder",
                 pkg_name="panda_pick_place",
                 sdf_rel_path="models/cylinder.sdf"):
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
        """在 Gazebo 中生成 cube"""
        rospy.loginfo("[GazeboCubeManager] 等待 /gazebo/spawn_sdf_model 服务...")
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=20.0)
        except rospy.ROSException:
            rospy.logerr(
                "[GazeboCubeManager] 等待 /gazebo/spawn_sdf_model 超时，请确认 Gazebo 已启动"
            )
            return False

        spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        try:
            sdf_xml = self._load_sdf()
        except Exception as e:
            rospy.logerr(f"[GazeboCubeManager] 读取 cube.sdf 失败: {e}")
            return False


        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0

        req = SpawnModelRequest()
        req.model_name = self.model_name
        req.model_xml = sdf_xml
        
        req.robot_namespace = ""
        req.initial_pose = pose
        req.reference_frame = reference_frame


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
            

        except rospy.ServiceException as e:
            rospy.logerr(f"[GazeboCubeManager] 调用 spawn_sdf_model 出错: {e}")
            return False

    def delete_cube(self):
        """删除 Gazebo 中的 cube 模型"""
        rospy.loginfo("[GazeboCubeManager] 尝试删除 Gazebo 中的方块 cube ...")
        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=10.0)
        except rospy.ROSException:
            rospy.logerr(
                "[GazeboCubeManager] 等待 /gazebo/delete_model 超时，请确认 Gazebo 正在运行"
            )
            return False

        delete_srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        req = DeleteModelRequest()
        req.model_name = self.model_name

        try:
            resp = delete_srv(req)
            if resp.success:
                rospy.loginfo("[GazeboCubeManager] ✓ 已删除 Gazebo 中的方块 cube")
                return True
            else:
                rospy.logwarn(
                    f"[GazeboCubeManager] ⚠ 删除方块 cube 失败: {resp.status_message}"
                )
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"[GazeboCubeManager] 调用 delete_model 出错: {e}")
            return False


class GazeboLinkAttacher:
    """封装 gazebo_ros_link_attacher 的 attach / detach 调用"""

    def __init__(self,
                 model_name_1="panda",
                 link_name_1="panda_link7",   # 原来是 panda_leftfinger
                 model_name_2="cube",
                 link_name_2="link"):
        rospy.loginfo(
            "等待 Gazebo link attacher 服务 /link_attacher_node/attach & detach ..."
        )
        rospy.wait_for_service("/link_attacher_node/attach")
        rospy.wait_for_service("/link_attacher_node/detach")
        self.attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach)
        self.detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach)
        rospy.loginfo("Gazebo link attacher 服务已连接")

        self.model_name_1 = model_name_1
        self.link_name_1 = link_name_1
        self.model_name_2 = model_name_2
        self.link_name_2 = link_name_2

    def attach(self):
        """在 Gazebo 中把方块粘到夹爪上"""
        rospy.loginfo(">>> Gazebo: attach 物体 到夹爪")
        req = AttachRequest()
        req.model_name_1 = self.model_name_1
        req.link_name_1 = self.link_name_1
        req.model_name_2 = self.model_name_2
        req.link_name_2 = self.link_name_2
        try:
            resp = self.attach_srv.call(req)
            if resp.ok:
                rospy.loginfo("Gazebo attach 成功")
                return True
            else:
                rospy.logwarn("Gazebo attach 返回失败")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Gazebo attach 服务调用出错: {e}")
            return False

    def detach(self):
        """在 Gazebo 中把方块和夹爪分离"""
        rospy.loginfo(">>> Gazebo: detach 物体 和夹爪")
        req = AttachRequest()
        req.model_name_1 = self.model_name_1
        req.link_name_1 = self.link_name_1
        req.model_name_2 = self.model_name_2
        req.link_name_2 = self.link_name_2
        try:
            resp = self.detach_srv.call(req)
            if resp.ok:
                rospy.loginfo("Gazebo detach 成功")
                return True
            else:
                rospy.logwarn("Gazebo detach 返回失败")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Gazebo detach 服务调用出错: {e}")
            return False
