"""
Project Name: Cylinder Target Localization
File Name: RMSE_test.py
Author: Liu Ming
Created Time: March 2nd, 2025

Description:
This script creates a GUI application using Tkinter to visualize the localization of probes on a cylindrical target.
It calculates the Root Mean Square Error (RMSE) between the probe distances and the actual distances, and plots both
a 3D visualization of the cylinder and probes, and a 2D contour plot of the RMSE values.

Preparation:
- Ensure Python is installed on your system.
- Install the required libraries: tkinter, numpy, matplotlib, scipy

Run Instructions:
1. Run the script using the command: python RMSE_test.py
2. Input the cylinder height, radius, and number of probes.
3. Update the probe inputs and set default parameters if needed.
4. Click "Plot and Calculate RMSE" to visualize the results.

Modification Log:
- March 2nd, 2025: Initial creation by Liu Ming
- YYYY-MM-DD: [Modification Notes]
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import griddata  # Using SciPy's griddata

# Global variable to store references to probe input fields
probe_entries = []

# RMSE calculation function
def calculate_rmse(z_phi, d_qi):
    return np.sqrt(np.mean((z_phi - d_qi) ** 2))

# Plot 3D graph
def plot_3d(cylinder_height, cylinder_radius, probe_coords, distances):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, cylinder_height, 100)
    theta, z = np.meshgrid(theta, z)
    x = cylinder_radius * np.cos(theta)
    y = cylinder_radius * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=0.5, color='blue', label='Cylinder')

    # Draw probe positions
    for i, (x, y, z) in enumerate(probe_coords):
        ax.scatter(x, y, z, color='red', s=50, label=f'Probe {i+1}' if i == 0 else None)
        ax.text(x, y, z, f'{distances[i]:.2f}', color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Cylinder and Probes')
    ax.legend()
    return fig

# Plot 2D RMSE graph
def plot_rmse(z_values, phi_values, rmse_values, cylinder_height):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create grid: x-axis is Phi (0° to 360°), y-axis is Z (0 to cylinder_height)
    xi = np.linspace(0, 360, 100)
    yi = np.linspace(0, cylinder_height, 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolation, using 'cubic' method and filling with mean value to cover the entire area
    zi = griddata((phi_values, z_values), rmse_values, (xi, yi), method='cubic', fill_value=np.nanmean(rmse_values))

    # Draw contour plot, increase levels for smooth transition
    contour = ax.contourf(xi, yi, zi, cmap='viridis', levels=20)
    plt.colorbar(contour, label='RMSE (mm)')

    # Set axis labels
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('RMSE Contour Plot')
    return fig

# Update probe input fields
def update_probe_inputs():
    global probe_entries
    for widget in probe_frame.winfo_children():
        widget.destroy()

    num_probes = int(num_probes_entry.get())
    probe_entries = []
    for i in range(num_probes):
        tk.Label(probe_frame, text=f'Probe {i+1} (x, y, z):').grid(row=i, column=0, padx=5, pady=5)
        x_entry = tk.Entry(probe_frame, width=5)
        x_entry.grid(row=i, column=1, padx=5, pady=5)
        y_entry = tk.Entry(probe_frame, width=5)
        y_entry.grid(row=i, column=2, padx=5, pady=5)
        z_entry = tk.Entry(probe_frame, width=5)
        z_entry.grid(row=i, column=3, padx=5, pady=5)
        tk.Label(probe_frame, text='Distance d_i:').grid(row=i, column=4, padx=5, pady=5)
        d_entry = tk.Entry(probe_frame, width=5)
        d_entry.grid(row=i, column=5, padx=5, pady=5)
        probe_entries.append([x_entry, y_entry, z_entry, d_entry])

# Calculate and plot
def calculate_and_plot():
    cylinder_height = float(height_entry.get())
    cylinder_radius = float(radius_entry.get())
    num_probes = int(num_probes_entry.get())

    probe_coords = []
    distances = []
    z_values = []
    phi_values = []
    rmse_values = []

    for i in range(num_probes):
        x = float(probe_entries[i][0].get())
        y = float(probe_entries[i][1].get())
        z = float(probe_entries[i][2].get())
        d = float(probe_entries[i][3].get())

        probe_coords.append((x, y, z))
        distances.append(d)

        # Calculate Phi angle (0° to 360°)
        phi = (np.degrees(np.arctan2(y, x)) + 360) % 360
        phi_values.append(phi)

        # Calculate RMSE (example, adjust as needed)
        rmse = np.abs(z - d)
        rmse_values.append(rmse)
        z_values.append(z)

    # Plot 3D graph
    fig_3d = plot_3d(cylinder_height, cylinder_radius, probe_coords, distances)
    canvas_3d = FigureCanvasTkAgg(fig_3d, master=plot_frame)
    canvas_3d.draw()
    canvas_3d.get_tk_widget().pack()

    # Plot 2D RMSE graph
    fig_rmse = plot_rmse(z_values, phi_values, rmse_values, cylinder_height)
    canvas_rmse = FigureCanvasTkAgg(fig_rmse, master=rmse_frame)
    canvas_rmse.draw()
    canvas_rmse.get_tk_widget().pack()

    # Display RMSE
    rmse_label.config(text=f'RMSE: {np.mean(rmse_values):.2f} mm')

# Set default parameters
def set_default_parameters():
    height_entry.delete(0, tk.END)
    height_entry.insert(0, "100")
    radius_entry.delete(0, tk.END)
    radius_entry.insert(0, "10")
    num_probes_entry.delete(0, tk.END)
    num_probes_entry.insert(0, "3")
    update_probe_inputs()
    fill_default_probe_data()

# Fill default probe data
def fill_default_probe_data():
    num_probes = int(num_probes_entry.get())
    default_probes = [
        (10, 0, 0, 50),   # Probe 1
        (-10, 0, 50, 50), # Probe 2
        (0, 10, 100, 50)  # Probe 3
    ]

    for i in range(num_probes):
        probe_entries[i][0].delete(0, tk.END)
        probe_entries[i][0].insert(0, default_probes[i][0])
        probe_entries[i][1].delete(0, tk.END)
        probe_entries[i][1].insert(0, default_probes[i][1])
        probe_entries[i][2].delete(0, tk.END)
        probe_entries[i][2].insert(0, default_probes[i][2])
        probe_entries[i][3].delete(0, tk.END)
        probe_entries[i][3].insert(0, default_probes[i][3])

# Create main window
root = tk.Tk()
root.title("Cylinder Target Localization")

# Left input frame
input_frame = ttk.LabelFrame(root, text="Input Parameters")
input_frame.pack(side=tk.LEFT, padx=10, pady=10)

ttk.Label(input_frame, text="Cylinder Height (mm):").grid(row=0, column=0, padx=5, pady=5)
height_entry = ttk.Entry(input_frame)
height_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Cylinder Radius (mm):").grid(row=1, column=0, padx=5, pady=5)
radius_entry = ttk.Entry(input_frame)
radius_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Number of Probes:").grid(row=2, column=0, padx=5, pady=5)
num_probes_entry = ttk.Entry(input_frame)
num_probes_entry.grid(row=2, column=1, padx=5, pady=5)

update_button = ttk.Button(input_frame, text="Update Probe Inputs", command=update_probe_inputs)
update_button.grid(row=3, column=0, columnspan=2, pady=10)

default_button = ttk.Button(input_frame, text="Set Default Parameters", command=set_default_parameters)
default_button.grid(row=4, column=0, columnspan=2, pady=10)

plot_button = ttk.Button(input_frame, text="Plot and Calculate RMSE", command=calculate_and_plot)
plot_button.grid(row=5, column=0, columnspan=2, pady=10)

# Probe input frame
probe_frame = ttk.LabelFrame(root, text="Probe Coordinates and Distances")
probe_frame.pack(side=tk.LEFT, padx=10, pady=10)

# 3D plot frame
plot_frame = ttk.LabelFrame(root, text="3D Visualization")
plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# RMSE plot frame
rmse_frame = ttk.LabelFrame(root, text="RMSE Contour Plot")
rmse_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# RMSE display
rmse_label = ttk.Label(root, text="RMSE: ")
rmse_label.pack(pady=10)

# Initialize default parameters
set_default_parameters()

# Start main loop
root.mainloop()