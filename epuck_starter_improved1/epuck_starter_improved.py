from controller import Robot
import sys
import time
import epuck_starter_improved1

robot = Robot()
timestep = int(robot.getBasicTimeStep())

def load_logic():
    """读取外部文件并执行，永远加载最新版"""
    global logic_namespace
    logic_namespace = {}
    with open("epuck_starter_improved1.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, logic_namespace)

# 第一次载入
load_logic()
last_mod_time = 0

import os

while robot.step(timestep) != -1:

    # 检测文件是否被修改
    modified = os.path.getmtime("epuck_starter_improved1.py")
    if modified != last_mod_time:
        print("Reloading logic...")
        load_logic()
        last_mod_time = modified

        # 每次执行新版逻辑
        logic_namespace["run"](robot)
