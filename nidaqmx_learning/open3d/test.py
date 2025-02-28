"""
Project Name: ML_Learning
File Name: test.py
Author: Liu Ming
Created Time: Feb 28, 2025
Description:
This script demonstrates the creation and visualization of a simple point cloud using the Open3D library. 
It includes a custom function to handle key callbacks for rotating the view and printing the cursor position.
Preparation:
- Install Open3D library: pip install open3d
How to Run:
- Execute the script in a Python environment: python test.py
- Use the 'W' and 'S' keys to rotate the view.
- Press 'P' to print the cursor position.
Modification Log:
- Modified Time: Feb 28, 2025
- Modified by: Liu Ming
    Modified Notes: Initial creation of the script.
"""

import open3d as o3d


# Create a simple point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Visualize the point cloud
def custom_draw_geometry_with_key_callback(pcd):
    def rotate_view(vis, angle):
        ctr = vis.get_view_control()
        ctr.rotate(angle, 0.0)
        return False

    def print_cursor_position(vis):
        print("Cursor position:", vis.get_cursor_position())
        return False

    key_to_callback = {
        ord("W"): lambda vis: rotate_view(vis, 10.0),
        ord("S"): lambda vis: rotate_view(vis, -10.0),
        ord("P"): print_cursor_position
    }

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

custom_draw_geometry_with_key_callback(pcd)
