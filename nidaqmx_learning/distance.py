import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class SimpleModelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple 3D Modeling Tool (Embedded)")
        self.root.geometry("800x500")

        # 用于存储当前的长方体和点数据
        self.cuboid_data = None  # (pos_x, pos_y, pos_z, size_x, size_y, size_z)
        self.points_data = []    # [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

        # Create GUI layout
        self.setup_gui()

    def setup_gui(self):
        # Left input area
        left_frame = ttk.Frame(self.root, width=200, relief="solid", borderwidth=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Cuboid parameters
        ttk.Label(left_frame, text="Cuboid Parameters").pack(pady=5)

        ttk.Label(left_frame, text="Length (x):").pack()
        self.size_x = ttk.Entry(left_frame, width=15)
        self.size_x.insert(0, "1.0")
        self.size_x.pack(pady=2)

        ttk.Label(left_frame, text="Width (y):").pack()
        self.size_y = ttk.Entry(left_frame, width=15)
        self.size_y.insert(0, "1.0")
        self.size_y.pack(pady=2)

        ttk.Label(left_frame, text="Height (z):").pack()
        self.size_z = ttk.Entry(left_frame, width=15)
        self.size_z.insert(0, "1.0")
        self.size_z.pack(pady=2)

        ttk.Label(left_frame, text="Position (x, y, z):").pack(pady=5)
        self.pos_x = ttk.Entry(left_frame, width=15)
        self.pos_x.insert(0, "0.0")
        self.pos_x.pack(pady=2)
        self.pos_y = ttk.Entry(left_frame, width=15)
        self.pos_y.insert(0, "0.0")
        self.pos_y.pack(pady=2)
        self.pos_z = ttk.Entry(left_frame, width=15)
        self.pos_z.insert(0, "0.0")
        self.pos_z.pack(pady=2)

        ttk.Button(left_frame, text="Generate Cuboid", command=self.update_box).pack(pady=10)

        # Point parameters
        ttk.Label(left_frame, text="Point Coordinates").pack(pady=5)

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

        ttk.Button(left_frame, text="Generate Points", command=self.update_points).pack(pady=10)

        # Right 3D view area
        self.fig = plt.Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial empty scene
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.canvas.draw()

    def get_input_values(self):
        try:
            size_x = float(self.size_x.get())
            size_y = float(self.size_y.get())
            size_z = float(self.size_z.get())
            pos_x = float(self.pos_x.get())
            pos_y = float(self.pos_y.get())
            pos_z = float(self.pos_z.get())
            return size_x, size_y, size_z, pos_x, pos_y, pos_z
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for cuboid!")
            return None

    def get_point_values(self):
        try:
            p1 = (float(self.p1_x.get()), float(self.p1_y.get()), float(self.p1_z.get()))
            p2 = (float(self.p2_x.get()), float(self.p2_y.get()), float(self.p2_z.get()))
            p3 = (float(self.p3_x.get()), float(self.p3_y.get()), float(self.p3_z.get()))
            return [p1, p2, p3]
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for points!")
            return None

    def redraw_scene(self):
        # 清空视图并重新绘制所有内容
        self.ax.clear()

        # 绘制长方体（如果存在）
        if self.cuboid_data:
            pos_x, pos_y, pos_z, size_x, size_y, size_z = self.cuboid_data
            self.ax.bar3d(pos_x, pos_y, pos_z, size_x, size_y, size_z, color='blue', alpha=1.0)  # 蓝色，完全不透明

        # 绘制点（如果存在）
        if self.points_data:
            x_vals, y_vals, z_vals = zip(*self.points_data)
            self.ax.scatter(x_vals, y_vals, z_vals, color='red', s=50)

        # 设置坐标轴范围
        all_x = []
        all_y = []
        all_z = []
        if self.cuboid_data:
            pos_x, pos_y, pos_z, size_x, size_y, size_z = self.cuboid_data
            all_x.extend([pos_x, pos_x + size_x])
            all_y.extend([pos_y, pos_y + size_y])
            all_z.extend([pos_z, pos_z + size_z])
        if self.points_data:
            x_vals, y_vals, z_vals = zip(*self.points_data)
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)

        if all_x:  # 如果有内容，调整范围
            self.ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            self.ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            self.ax.set_zlim(min(all_z) - 1, max(all_z) + 1)
        else:  # 否则使用默认范围
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(-2, 2)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.canvas.draw()

    def update_box(self):
        values = self.get_input_values()
        if not values:
            return
        
        # 更新长方体数据
        self.cuboid_data = values
        self.redraw_scene()

    def update_points(self):
        points = self.get_point_values()
        if not points:
            return
        
        # 更新点数据
        self.points_data = points
        self.redraw_scene()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleModelingApp(root)
    root.mainloop()