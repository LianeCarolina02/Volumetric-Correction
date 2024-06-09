
import open3d as o3d
import copy
import numpy as np
import surface_acquisition
import cv2

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
    mesh.paint_uniform_color([0, 0.651, 0.929])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def visualize_pcd_twin(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.paint_uniform_color([1, 0.706, 0])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(pcd)
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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    print(f"1st Point cloud: yellow gold\nTranformation: {transformation} \n2nd Point cloud: blue")
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.get_render_option().mesh_show_back_face = False
    vis.run()


def capture_video(mesh_path=None, pcd_path=None, pcd=None, output_file="this.avi", width=640, height=480, fps=30):
    if mesh_path is not None:
        pcd = o3d.io.read_triangle_mesh(mesh_path)
        pcd.compute_vertex_normals()
    elif pcd_path is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    

    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    vis.add_geometry(pcd)
    view_control = vis.get_view_control()

    # Capture frames
    for _ in range(360):  # Rotate 360 degrees
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        view_control.rotate(0.0, 10.0)

        # Capture the current frame
        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    # Release resources
    vis.destroy_window()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    folder = "Pacients"
    number = 74

    patient = "BR0" + f"{number}"

    digital_twin = f"{folder}/{patient}/Segment_4.stl"
    surface_digital_twin = "Pacients/BR074/Final_Surface.ply"
    _try_ = "Pacients/BR074/try.stl"
    surface_scan = f"Pacients/{patient}/{number}/{number}.obj"

    # visualize_mesh_digital_twin(_try_)
    # visualize_surface_scan_mesh(surface_scan)
    visualize_pcd_twin(surface_digital_twin)

    # capture_video(mesh_path=digital_twin, pcd_path=None, output_file="original_mesh.avi", width=1920, height=1080, fps=30)