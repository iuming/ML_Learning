"""
Project Name: Simple 3D Modeling Tool
File Name: distance.py
Author: Liu Ming
Created Date: Feb 28, 2025
Description:
This script implements a simple 3D modeling tool using Tkinter for the GUI and Matplotlib for 3D visualization. 
It allows users to import a 3D model in STL format, input satellite points with distances, and visualize the model 
along with the satellite points and their respective target areas. The intersection areas between the model and 
the spheres around the satellite points are highlighted with a darker color. Zoom functionality is added via buttons.
Prerequisites:
- Python 3.x
- Tkinter
- Matplotlib
- numpy-stl
- numpy
To run this script:
1. Ensure all prerequisites are installed.
2. Update the default model file path in the GUI or input the correct path.
3. Run the script using the command: python distance.py
Modification Log:
- Modified Time: Feb 28, 2025
- Modified By: Liu Ming
    Modified Notes: Initial creation of the script.
- Modified Time: Mar 1, 2025
- Modified By: Liu Ming
    Modified Notes: Added functionality to highlight intersection areas with a darker color.
- Modified Time: Mar 1, 2025
- Modified By: Liu Ming
    Modified Notes: Added zoom functionality with buttons for enhanced interaction.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from stl import mesh  # 需要安装 numpy-stl

class SimpleModelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple 3D Modeling Tool (Embedded)")
        self.root.geometry("800x700")

        # 数据存储
        self.model_data = None  # 存储导入的模型顶点和面
        self.points_data = []   # [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
        self.distances = [None, None, None]  # 每个点的距离
        self.zoom_level = 1.0  # 初始缩放级别

        # 创建GUI布局
        self.setup_gui()

    def setup_gui(self):
        # 左侧输入区域
        left_frame = ttk.Frame(self.root, width=200, relief="solid", borderwidth=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 模型导入参数
        ttk.Label(left_frame, text="Import Model").pack(pady=5)
        ttk.Label(left_frame, text="Model File Path:").pack()
        self.model_path = ttk.Entry(left_frame, width=15)
        self.model_path.insert(0, "nidaqmx_learning\\open3d\\BareCavity.STL")  # 默认路径
        self.model_path.pack(pady=2)
        ttk.Button(left_frame, text="Import Model", command=self.import_model).pack(pady=10)

        # 卫星点参数
        ttk.Label(left_frame, text="Satellite Points").pack(pady=5)

        # Point 1
        ttk.Label(left_frame, text="Point 1 (x, y, z):").pack()
        self.p1_x = ttk.Entry(left_frame, width=15)
        self.p1_x.insert(0, "100.0")
        self.p1_x.pack(pady=2)
        self.p1_y = ttk.Entry(left_frame, width=15)
        self.p1_y.insert(0, "0.0")
        self.p1_y.pack(pady=2)
        self.p1_z = ttk.Entry(left_frame, width=15)
        self.p1_z.insert(0, "0.0")
        self.p1_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 1:").pack()
        self.dist1 = ttk.Entry(left_frame, width=15)
        self.dist1.insert(0, "100.0")
        self.dist1.pack(pady=2)

        # Point 2
        ttk.Label(left_frame, text="Point 2 (x, y, z):").pack()
        self.p2_x = ttk.Entry(left_frame, width=15)
        self.p2_x.insert(0, "250.0")
        self.p2_x.pack(pady=2)
        self.p2_y = ttk.Entry(left_frame, width=15)
        self.p2_y.insert(0, "0.0")
        self.p2_y.pack(pady=2)
        self.p2_z = ttk.Entry(left_frame, width=15)
        self.p2_z.insert(0, "0.0")
        self.p2_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 2:").pack()
        self.dist2 = ttk.Entry(left_frame, width=15)
        self.dist2.insert(0, "100.0")
        self.dist2.pack(pady=2)

        # Point 3
        ttk.Label(left_frame, text="Point 3 (x, y, z):").pack()
        self.p3_x = ttk.Entry(left_frame, width=15)
        self.p3_x.insert(0, "200.0")
        self.p3_x.pack(pady=2)
        self.p3_y = ttk.Entry(left_frame, width=15)
        self.p3_y.insert(0, "200.0")
        self.p3_y.pack(pady=2)
        self.p3_z = ttk.Entry(left_frame, width=15)
        self.p3_z.insert(0, "200.0")
        self.p3_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 3:").pack()
        self.dist3 = ttk.Entry(left_frame, width=15)
        self.dist3.insert(0, "150.0")
        self.dist3.pack(pady=2)

        ttk.Button(left_frame, text="Generate Points", command=self.update_points).pack(pady=5)
        ttk.Button(left_frame, text="Generate Target Area", command=self.update_target_area).pack(pady=10)

        # 缩放按钮
        ttk.Label(left_frame, text="Zoom Controls").pack(pady=5)
        ttk.Button(left_frame, text="Zoom In", command=self.zoom_in).pack(pady=2)
        ttk.Button(left_frame, text="Zoom Out", command=self.zoom_out).pack(pady=2)

        # 右侧3D视图区域
        self.fig = plt.Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 初始空场景
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.canvas.draw()

    def get_point_values(self):
        try:
            p1 = (float(self.p1_x.get()), float(self.p1_y.get()), float(self.p1_z.get()))
            p2 = (float(self.p2_x.get()), float(self.p2_y.get()), float(self.p2_z.get()))
            p3 = (float(self.p3_x.get()), float(self.p3_y.get()), float(self.p3_z.get()))
            return [p1, p2, p3]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for points!")
            return None

    def get_distances(self):
        try:
            d1 = float(self.dist1.get())
            d2 = float(self.dist2.get())
            d3 = float(self.dist3.get())
            return [d1, d2, d3]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid distances!")
            return None

    def import_model(self):
        file_path = self.model_path.get()
        try:
            model = mesh.Mesh.from_file(file_path)
            vertices = model.vectors.reshape(-1, 3)  # 顶点
            faces = np.arange(len(vertices)).reshape(-1, 3)  # 简单面索引
            self.model_data = (vertices, faces)
            self.redraw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def is_inside_all_spheres(self, vertex, centers, radii):
        for center, radius in zip(centers, radii):
            if np.linalg.norm(vertex - center) > radius:
                return False
        return True

    def redraw_scene(self):
        self.ax.clear()

        # 绘制导入的模型
        if self.model_data:
            vertices, faces = self.model_data
            if self.points_data and all(d is not None for d in self.distances):
                centers = np.array(self.points_data)
                radii = np.array(self.distances)
                inside = [self.is_inside_all_spheres(v, centers, radii) for v in vertices]
                inside_indices = np.where(inside)[0]
                outside_indices = np.where(~np.array(inside))[0]

                if len(inside_indices) > 0:
                    self.ax.plot_trisurf(vertices[inside_indices, 0], vertices[inside_indices, 1], 
                                         faces, vertices[inside_indices, 2], color='darkblue', alpha=0.8)
                if len(outside_indices) > 0:
                    self.ax.plot_trisurf(vertices[outside_indices, 0], vertices[outside_indices, 1], 
                                         faces, vertices[outside_indices, 2], color='blue', alpha=0.8)
            else:
                self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='blue', alpha=0.8)

        # 绘制卫星点
        if self.points_data:
            x_vals, y_vals, z_vals = zip(*self.points_data)
            self.ax.scatter(x_vals, y_vals, z_vals, color='red', s=50, label='Satellites')

        # 绘制目标区域（球体）
        if self.points_data and all(d is not None for d in self.distances):
            for p, d in zip(self.points_data, self.distances):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = d * np.outer(np.cos(u), np.sin(v)) + p[0]
                y = d * np.outer(np.sin(u), np.sin(v)) + p[1]
                z = d * np.outer(np.ones(np.size(u)), np.cos(v)) + p[2]
                self.ax.plot_wireframe(x, y, z, color='green', alpha=0.3)

        # 设置坐标轴范围（考虑缩放）
        all_x, all_y, all_z = [], [], []
        if self.model_data:
            vertices, _ = self.model_data
            all_x.extend(vertices[:, 0])
            all_y.extend(vertices[:, 1])
            all_z.extend(vertices[:, 2])
        if self.points_data:
            x_vals, y_vals, z_vals = zip(*self.points_data)
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)
        if all(d is not None for d in self.distances):
            for p, d in zip(self.points_data, self.distances):
                all_x.extend([p[0] - d, p[0] + d])
                all_y.extend([p[1] - d, p[1] + d])
                all_z.extend([p[2] - d, p[2] + d])

        if all_x:
            self.ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            self.ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            self.ax.set_zlim(min(all_z) - 1, max(all_z) + 1)
        else:
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(-2, 2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.canvas.draw()

    def update_points(self):
        points = self.get_point_values()
        if not points:
            return
        self.points_data = points
        self.redraw_scene()

    def update_target_area(self):
        distances = self.get_distances()
        if not distances or not self.points_data:
            messagebox.showerror("Error", "Please generate points first and enter valid distances!")
            return
        self.distances = distances
        self.redraw_scene()

    def zoom_in(self):
        """放大视图"""
        self.zoom_level *= 1.1  # 放大10%
        self.adjust_zoom()

    def zoom_out(self):
        """缩小视图"""
        self.zoom_level /= 1.1  # 缩小10%
        self.adjust_zoom()

    def adjust_zoom(self):
        """根据缩放级别调整坐标轴范围"""
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_zlim = self.ax.get_zlim()

        # 计算当前视图中心
        center_x = (current_xlim[0] + current_xlim[1]) / 2
        center_y = (current_ylim[0] + current_ylim[1]) / 2
        center_z = (current_zlim[0] + current_zlim[1]) / 2

        # 根据缩放级别调整范围
        new_xlim = [center_x - (center_x - current_xlim[0]) / self.zoom_level,
                    center_x + (current_xlim[1] - center_x) / self.zoom_level]
        new_ylim = [center_y - (center_y - current_ylim[0]) / self.zoom_level,
                    center_y + (current_ylim[1] - center_y) / self.zoom_level]
        new_zlim = [center_z - (center_z - current_zlim[0]) / self.zoom_level,
                    center_z + (current_zlim[1] - center_z) / self.zoom_level]

        # 应用新的范围
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.ax.set_zlim(new_zlim)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleModelingApp(root)
    root.mainloop()