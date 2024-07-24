import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import prepare_dataset as prd




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




