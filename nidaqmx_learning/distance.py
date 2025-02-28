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
        self.model_path.insert(0, "model.stl")  # 默认值，需替换为实际路径
        self.model_path.pack(pady=2)
        ttk.Button(left_frame, text="Import Model", command=self.import_model).pack(pady=10)

        # 卫星点参数
        ttk.Label(left_frame, text="Satellite Points").pack(pady=5)

        # Point 1
        ttk.Label(left_frame, text="Point 1 (x, y, z):").pack()
        self.p1_x = ttk.Entry(left_frame, width=15)
        self.p1_x.insert(0, "0.0")
        self.p1_x.pack(pady=2)
        self.p1_y = ttk.Entry(left_frame, width=15)
        self.p1_y.insert(0, "0.0")
        self.p1_y.pack(pady=2)
        self.p1_z = ttk.Entry(left_frame, width=15)
        self.p1_z.insert(0, "0.0")
        self.p1_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 1:").pack()
        self.dist1 = ttk.Entry(left_frame, width=15)
        self.dist1.insert(0, "1.0")
        self.dist1.pack(pady=2)

        # Point 2
        ttk.Label(left_frame, text="Point 2 (x, y, z):").pack()
        self.p2_x = ttk.Entry(left_frame, width=15)
        self.p2_x.insert(0, "1.0")
        self.p2_x.pack(pady=2)
        self.p2_y = ttk.Entry(left_frame, width=15)
        self.p2_y.insert(0, "1.0")
        self.p2_y.pack(pady=2)
        self.p2_z = ttk.Entry(left_frame, width=15)
        self.p2_z.insert(0, "1.0")
        self.p2_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 2:").pack()
        self.dist2 = ttk.Entry(left_frame, width=15)
        self.dist2.insert(0, "1.0")
        self.dist2.pack(pady=2)

        # Point 3
        ttk.Label(left_frame, text="Point 3 (x, y, z):").pack()
        self.p3_x = ttk.Entry(left_frame, width=15)
        self.p3_x.insert(0, "2.0")
        self.p3_x.pack(pady=2)
        self.p3_y = ttk.Entry(left_frame, width=15)
        self.p3_y.insert(0, "2.0")
        self.p3_y.pack(pady=2)
        self.p3_z = ttk.Entry(left_frame, width=15)
        self.p3_z.insert(0, "0.0")
        self.p3_z.pack(pady=2)
        ttk.Label(left_frame, text="Distance 3:").pack()
        self.dist3 = ttk.Entry(left_frame, width=15)
        self.dist3.insert(0, "1.0")
        self.dist3.pack(pady=2)

        ttk.Button(left_frame, text="Generate Points", command=self.update_points).pack(pady=5)
        ttk.Button(left_frame, text="Generate Target Area", command=self.update_target_area).pack(pady=10)

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
            # 加载STL文件
            model = mesh.Mesh.from_file(file_path)
            vertices = model.vectors.reshape(-1, 3)  # 顶点
            faces = np.arange(len(vertices)).reshape(-1, 3)  # 简单面索引，可能需要调整
            self.model_data = (vertices, faces)
            self.redraw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def redraw_scene(self):
        self.ax.clear()

        # 绘制导入的模型
        if self.model_data:
            vertices, faces = self.model_data
            self.ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='blue', alpha=0.8)

        # 绘制卫星点
        if self.points_data:
            x_vals, y_vals, z_vals = zip(*self.points_data)
            self.ax.scatter(x_vals, y_vals, z_vals, color='red', s=50, label='Satellites')

        # 绘制目标区域（每个点一个球体）
        if self.points_data and all(d is not None for d in self.distances):
            for p, d in zip(self.points_data, self.distances):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = d * np.outer(np.cos(u), np.sin(v)) + p[0]
                y = d * np.outer(np.sin(u), np.sin(v)) + p[1]
                z = d * np.outer(np.ones(np.size(u)), np.cos(v)) + p[2]
                self.ax.plot_wireframe(x, y, z, color='green', alpha=0.3)

        # 设置坐标轴范围
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

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleModelingApp(root)
    root.mainloop()