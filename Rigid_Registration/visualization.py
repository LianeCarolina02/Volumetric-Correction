import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import prepare_dataset as prd

def draw_point_cloud(pcd):
    points = np.asarray(pcd.points)
    color_map = plt.get_cmap('Blues')

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    colors = color_map(((1 - (points[:, 2] - z_min) / (z_max - z_min)) + 0.5))[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], width=1600, 
                                  height=1200)


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


def save_image(source, target, transformation, filename):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.run()
    
    vis.capture_screen_image(filename)
    
    vis.destroy_window()




