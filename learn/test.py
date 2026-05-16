#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 写法 A：路径全称
import core.core.motor
from core.core import lidar

core.core.motor.run_motor()

# 写法 B：从...导入（推荐，更简洁）
from core.core import motor
motor.run_motor()

# 写法 C：直接导入函数
from core.core.motor import run_motor
run_motor()

