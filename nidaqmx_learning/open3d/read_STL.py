"""
Project Name: ML_Learning
File Name: read_STL.py
Author: Liu Ming
Created Time: Feb 28, 2025
Description:
This script demonstrates the reading and visualization of an STL file using the Open3D library. 
It includes the creation of a coordinate frame and a grid for better visualization.
Preparation:
- Install Open3D library: pip install open3d
How to Run:
- Execute the script in a Python environment: python read_STL.py
Modification Log:
- Modified Time: Feb 28, 2025
- Modified by: Liu Ming
    Modified Notes: Initial creation of the script.
"""

import open3d as o3d
import numpy as np

# 读取STL文件
mesh = o3d.io.read_triangle_mesh("nidaqmx_learning/open3d/BareCavity.STL")

# 检查是否成功读取
if not mesh.is_empty():
    print("Successfully read the STL file.")
else:
    print("Failed to read the STL file.")
    exit()

# 创建彩色坐标轴（红:X轴，绿:Y轴，蓝:Z轴）
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])

# 创建XY平面网格线（灰色）
grid_size = 2.0  # 网格覆盖范围：-grid_size 到 +grid_size
step = 0.5       # 网格线间距

# 生成网格顶点和线段
points = []
lines = []
line_count = 0

# X方向网格线（平行于Y轴）
for x in np.arange(-grid_size, grid_size + step, step):
    points.append([x, -grid_size, 0])
    points.append([x,  grid_size, 0])
    lines.append([line_count, line_count+1])
    line_count += 2

# Y方向网格线（平行于X轴）
for y in np.arange(-grid_size, grid_size + step, step):
    points.append([-grid_size, y, 0])
    points.append([ grid_size, y, 0])
    lines.append([line_count, line_count+1])
    line_count += 2

# 创建线框对象
grid_lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
grid_lines.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

# 显示所有对象（模型+坐标轴+网格）
o3d.visualization.draw_geometries(
    [mesh, coord_frame, grid_lines],
    zoom=0.7,
    front=[0.0, 0.0, -1.0],
    lookat=[0.0, 0.0, 0.0],
    up=[0.0, -1.0, 0.0],
    window_name="3D Model with Coordinate Grid"
)