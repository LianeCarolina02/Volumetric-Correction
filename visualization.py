
import open3d as o3d
import copy
import numpy as np
import surface_acquisition

def visualize_mesh_digital_twin(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def visualize_point_cloud_digital_twin(mesh_path, number):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=number)
    pcd.paint_uniform_color([1, 0.706, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def visualize_save_surface_digital_twin(patient, pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    o3d.io.write_point_cloud(f"Pacients/{patient}/Final_Surface.ply", pcd)

def visualize_surface_scan_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def visualize_surface_and_all(pcd_path, mesh_path):
    print("Reading Mesh...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("Sampling Mesh...")
    pcd_1 = mesh.sample_points_uniformly(number_of_points=1000000)
    pcd_1_rotated = surface_acquisition.rotation_pcd(pcd_1)
    pcd_1_rotated.paint_uniform_color([0.5, 1, 0.5])

    print("Reading surface point cloud...")
    pcd_2= o3d.io.read_point_cloud(pcd_path)
    pcd_2.paint_uniform_color([1,0.5,0.5])

    x_max,_,_,_,_,_ = surface_acquisition.bouding_points(pcd_1_rotated)

    pcd_2.translate([2.5*x_max,0, -200])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(pcd_1_rotated)
    vis.add_geometry(pcd_2)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    folder = "Pacients"
    number = 61

    patient = "BR0" + f"{number}"

    digital_twin = f"{folder}/{patient}/Segment_4.stl"
    surface_digital_twin = f"{folder}/{patient}/Surface.ply"






