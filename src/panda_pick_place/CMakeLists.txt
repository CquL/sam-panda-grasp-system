cmake_minimum_required(VERSION 3.0.2)
project(panda_pick_place)

## 查找catkin宏和库
find_package(catkin REQUIRED COMPONENTS
  rospy
  moveit_commander
  moveit_msgs
  geometry_msgs
  trajectory_msgs
  gazebo_ros
  panda_moveit_config
  tf
)

## 系统依赖
# find_package(Boost REQUIRED COMPONENTS system)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES panda_pick_place
  CATKIN_DEPENDS 
    rospy 
    moveit_commander 
    moveit_msgs 
    geometry_msgs 
    trajectory_msgs 
    gazebo_ros
    panda_moveit_config
    tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## 指定额外的头文件位置
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## 安装Python脚本
catkin_install_python(PROGRAMS
  scripts/pick_place_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 安装launch文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

## 安装模型文件
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  PATTERN ".svn" EXCLUDE
)

## 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

## 如需测试，在此添加