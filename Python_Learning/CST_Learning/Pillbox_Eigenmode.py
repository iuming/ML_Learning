"""
Program Name: CST_Learning
IDE: PyCharm
File Name: Pillbox_Eigenmode
Author: mliu
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/11/18 13:51

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/11/18 13:51: Initial Create.

"""

import cst
from cst.interface import Project

def run_pillbox_eigenmode_simulation():
    # 打开一个新项目
    with cst.Project() as project:
        project_path = "E:\\Data2\\mliu\\Python\\ML_Learning\\Python_Learning\\CST_Learning\\CST_Projects\\Pillbox_Eigenmode.cst"
        project.save(project_path)

        # 几何建模
        create_geometry(project)

        # 设置边界条件和求解器
        set_boundary_conditions(project)
        setup_solver(project)

        # 保存并运行仿真
        project.save()
        project.simulate()

        # 提取结果
        extract_results(project)

def create_geometry(project):
    """创建Pillbox腔体的几何结构"""
    component = project.modeler.components["Component1"]
    material = project.materials["PEC"]

    # 创建一个圆柱体 (Pillbox)
    component.create_cylinder(
        material=material,
        center=(0, 0, 0),
        radius=50.0,  # 半径 (mm)
        height=100.0,  # 高度 (mm)
        axis="z",  # 沿Z轴生成
        name="Pillbox"
    )

def set_boundary_conditions(project):
    """设置边界条件"""
    boundaries = project.boundary_conditions
    boundaries.x_min = "electric"  # X轴最小边界为电场边界
    boundaries.x_max = "electric"  # X轴最大边界为电场边界
    boundaries.y_min = "electric"  # Y轴最小边界为电场边界
    boundaries.y_max = "electric"  # Y轴最大边界为电场边界
    boundaries.z_min = "electric"  # Z轴最小边界为电场边界
    boundaries.z_max = "electric"  # Z轴最大边界为电场边界

def setup_solver(project):
    """设置求解器为Eigenmode"""
    solver = project.solver.eigenmode
    solver.frequency_range = (1e9, 5e9)  # 设置频率范围为1 GHz到5 GHz
    solver.modes = 1  # 求解第一个模态

def extract_results(project):
    """提取仿真结果"""
    results = project.results
    eigenfrequencies = results.eigenfrequencies()
    print("Eigenfrequencies (GHz):", eigenfrequencies)

if __name__ == "__main__":
    run_pillbox_eigenmode_simulation()
