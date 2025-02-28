"""
Project Name: ML_Learning
File Name: menu.py
Author: Liu Ming
Created Time: Feb 28, 2025
Description:
This script demonstrates a simple menu-driven application using the Open3D library.
It allows users to load and visualize point cloud files.
Preparation:
- Install Open3D library: pip install open3d
How to Run:
- Execute the script in a Python environment: python menu.py
Modification Log:
- Modified Time: Feb 28, 2025
- Modified by: Liu Ming
    Modified Notes: Initial creation of the script.
"""


import open3d as o3d

def create_menu():
    print("Open3D Menu")
    print("1. Load Point Cloud")
    print("2. Visualize Point Cloud")
    print("3. Exit")

def load_point_cloud():
    file_path = input("Enter the path to the point cloud file: ")
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])

def main():
    while True:
        create_menu()
        choice = input("Enter your choice: ")
        if choice == '1':
            pcd = load_point_cloud()
        elif choice == '2':
            if 'pcd' in locals():
                visualize_point_cloud(pcd)
            else:
                print("No point cloud loaded. Please load a point cloud first.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()