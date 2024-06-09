
import open3d as o3d
import copy
import numpy as np


def sample_mesh(mesh_path, number):
    print("::       Reading mesh")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("::       Sampling Mesh")
    pcd = mesh.sample_points_uniformly(number_of_points=number)
    return pcd


def rotation_pcd(pcd):
    print("::       Rotation of the pcd")
    theta = np.radians(90)  # Convert degrees to radians
    rotation_matrix = np.array([[1,             0,                  0               ],
                                [0,             -np.cos(theta),     np.sin(theta)   ],
                                [0,             -np.sin(theta),     -np.cos(theta)  ]])

    rotated_pcd = pcd.rotate(rotation_matrix)
    z_min_1 = rotated_pcd.get_min_bound()[2]
    x_max = rotated_pcd.get_max_bound()[0]
    x_min = rotated_pcd.get_min_bound()[0]
    half_x_1 = (x_max + x_min)/2
    print(f"::      z_min and half_x: {z_min_1, half_x_1}")
    vector = [-half_x_1, 0, -z_min_1]
    rotated_pcd.translate(vector)
    z_min_2 = rotated_pcd.get_min_bound()[2]
    x_max = rotated_pcd.get_max_bound()[0]
    x_min = rotated_pcd.get_min_bound()[0]
    half_x_2 = (x_max + x_min)/2
    print(f"::      z_min and half_x: {z_min_2, half_x_2}")
    return rotated_pcd, rotation_matrix, vector

def computation_diameter(pcd):
    print("::       Computing Diameter")
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print(f"::      Diameter of the pcd is: {diameter}")
    return diameter

def bouding_points(pcd):
    print("::       Computing limits")
    x_max = pcd.get_max_bound()[0]
    y_max = pcd.get_max_bound()[1]
    z_max = pcd.get_max_bound()[2]

    x_min = abs(pcd.get_min_bound()[0])
    y_min = abs(pcd.get_min_bound()[1])
    z_min = abs(pcd.get_min_bound()[2])

    print(f"::      Bounding Limits Max:[{x_max, y_max, z_max}")
    print(f"::      Bounding Limits Min:[{x_min, y_min, z_min}]")

    return x_max, y_max, z_max, x_min, y_min, z_min

def hidden_points_removal(all_pcd, camera, color, diameter):
    radius = diameter / 2 * 100
    _, pt_map_front = all_pcd.hidden_point_removal(np.array(camera), radius)  # Pass camera as numpy array
    pcd_remotion = all_pcd.select_by_index(pt_map_front)
    pcd_remotion.paint_uniform_color(color)

    return pcd_remotion

def final_pcd(all_pcd, cameras, colors, patient):
    diameter = computation_diameter(all_pcd)

    merged_pcd = o3d.geometry.PointCloud()
    for i, (camera, color) in enumerate(zip(cameras, colors)):
        print(f"::      Point Cloud Number {i + 1}-th Done")
        camera_coords = cameras[camera]
        merged_pcd +=  hidden_points_removal(all_pcd, camera_coords, color, diameter)

    print("::       Finish Merged")
    # merged_pcd.paint_uniform_color(colors[0])
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(merged_pcd)
    render_options = visualizer.get_render_option()
    render_options.point_size = 5
    visualizer.run()
    visualizer.destroy_window()
    o3d.io.write_point_cloud(f"Pacients/{patient}/Surface.ply", merged_pcd)
    return merged_pcd

def visualize_hpr(all_pcd, cameras, colors):
    diameter = computation_diameter(all_pcd)
    line_set = box(all_pcd)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    for i, (camera, color) in enumerate(zip(cameras, colors)):
        print(f"::      Point Cloud Number {i + 1}-th Done")
        camera_coords = cameras[camera]  # Retrieve camera coordinates
        pcd = hidden_points_removal(all_pcd, camera_coords, color, diameter)  # Pass camera coordinates
        visualizer.add_geometry(pcd)
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        camera_sphere.translate(camera_coords)
        camera_sphere.paint_uniform_color(color)
        visualizer.add_geometry(camera_sphere)
    
    visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
    visualizer.add_geometry(line_set)
    render_options = visualizer.get_render_option()
    render_options.point_size = 5
    visualizer.run()
    visualizer.destroy_window()

def visualize_surface_and_all(surface, original_pcd):
    surface_copy = copy.deepcopy(surface)
    original_pcd.paint_uniform_color([0.5, 1, 0.5])
    surface.paint_uniform_color([1,0.5,0.5])

    x_max,_,_,_,_,_ = bouding_points(surface_copy)

    surface_copy.translate([0,0, -200])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(original_pcd)
    vis.add_geometry(surface_copy)
    vis.run()
    vis.destroy_window()


def box(pcd):
    print("::       Computation of the box")
    half_diameter = computation_diameter(pcd) / 3
    vertices = [[-half_diameter, -half_diameter, -half_diameter],
                [half_diameter, -half_diameter, -half_diameter],
                [half_diameter, -half_diameter, half_diameter],
                [-half_diameter, -half_diameter, half_diameter],
                [-half_diameter, half_diameter, -half_diameter],
                [half_diameter, half_diameter, -half_diameter],
                [half_diameter, half_diameter, half_diameter],
                [-half_diameter, half_diameter, half_diameter]]

    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    black_color = [0, 0, 0]
    line_set.colors = o3d.utility.Vector3dVector([black_color] * len(edges))
    return line_set













# pcd = mesh.sample_points_uniformly(number_of_points=900)
# hull, _ = pcd.compute_convex_hull()
# hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# hull_ls.paint_uniform_color((1, 0, 0))
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color((1, 0.706, 0))


# o3d.visualization.draw_geometries([mesh, hulls])

# alpha = 0.001
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# # mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
