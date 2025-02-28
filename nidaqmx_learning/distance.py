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
        self.root.geometry("800x400")

        # Create GUI layout
        self.setup_gui()

    def setup_gui(self):
        # Left input area
        left_frame = ttk.Frame(self.root, width=200, relief="solid", borderwidth=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="Cuboid Parameters").pack(pady=5)

        # Size input
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

        # Position input
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

        # Generate button
        ttk.Button(left_frame, text="Generate Cuboid", command=self.update_box).pack(pady=10)

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
            messagebox.showerror("Error", "Please enter valid numbers!")
            return None

    def update_box(self):
        values = self.get_input_values()
        if not values:
            return
        
        size_x, size_y, size_z, pos_x, pos_y, pos_z = values

        # Clear current view
        self.ax.clear()

        # Draw cuboid (using bar3d to represent a simple cuboid)
        self.ax.bar3d(pos_x, pos_y, pos_z, size_x, size_y, size_z, color='gray', alpha=0.8)

        # Set axis range
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        max_range = max(size_x, size_y, size_z) + max(abs(pos_x), abs(pos_y), abs(pos_z)) + 1
        self.ax.set_xlim(pos_x - 1, pos_x + size_x + 1)
        self.ax.set_ylim(pos_y - 1, pos_y + size_y + 1)
        self.ax.set_zlim(pos_z - 1, pos_z + size_z + 1)

        # Update canvas
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleModelingApp(root)
    root.mainloop()