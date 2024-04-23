import datetime
import open3d as o3d
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os

def get_current_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m__%H_%M_%S")
    return formatted_datetime

Breast_ascii = "Manequin/Mannequin_Breast_ASCII.ply"
Torso_ascii = "Manequin/Mannequin_Torso_ASCII.ply"

def preprocessing(pcd, voxel_size = 0.01):
    # print(":: Downsampling with voxel %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    
    radius_normal = voxel_size * 2
    # print(":: Normal with search radius %.3f." % radius_normal)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Computation of FPFH feature with search radius %.3f." % radius_feature)

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_pcd, target_pcd, voxel_size=0.01):
    target = o3d.io.read_point_cloud(target_pcd)
    source = o3d.io.read_point_cloud(source_pcd)

    # print(":: Prepare Source Dataset")
    source_down, source_fpfh = preprocessing(source, voxel_size)
    # print(":: Prepare Target Dataset")
    target_down, target_fpfh = preprocessing(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh

if __name__ == '__main__':
    BREAST = "Manequin/Mannequin_Breast_ASCII.ply"
    FASCIA = "Manequin/Mannequin_Fascia_ASCII.ply"
    TORSO = "Manequin/Mannequin_Torso_ASCII.ply"
    voxel_size = 0.01

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(BREAST, TORSO, voxel_size=0.01)

    source = o3d.io.read_point_cloud(TORSO)

    radius_normal = voxel_size * 2

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    maximum = pcd_fpfh.data.max()
    minimum = pcd_fpfh.data.min()

    folder = "Features"
    for i in range(len(pcd_fpfh.data)):
        fpfh = np.asarray(pcd_fpfh.data[i,:])
        colormap = 'inferno'

        fpfh_colors = plt.get_cmap(colormap )(
                (fpfh - minimum) / (maximum - minimum))
        #fpfh_colors = plt.get_cmap(colormap)(fpfh)
        fpfh_colors = fpfh_colors[:, :3]


        fpfh_pcd =o3d.geometry.PointCloud()
        fpfh_pcd.points = source.points
        fpfh_pcd.normals = source.normals
        fpfh_pcd.colors = o3d.utility.Vector3dVector(fpfh_colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        vis.add_geometry(fpfh_pcd)
        vis.run()
        
        vis.capture_screen_image(f"{folder}/Feature_{i}_torso.png")
        
        vis.destroy_window()


    # num_rows = 6
    # num_cols = 6

    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 15))
    # folder_path = "Features"

    # selected_filenames = [f"Feature_{i}_torso.png" for i in range(33)]

    # image_files = [os.path.join(folder_path, file) for file in selected_filenames if file in os.listdir(folder_path)]

    # axes = axes.flatten()

    # for i, ax in enumerate(axes):
    #     if i < len(image_files):
    #         img = mpimg.imread(image_files[i])
    #         cropped_img = img[100:1100, 500:1100]
    #         ax.imshow(cropped_img)
    #         ax.axis('off')  # Hide axis
    #         ax.set_title(f"Feature {i}")
    #     else:
    #         ax.axis('off')


    # plt.tight_layout()
    # plt.savefig(f"{folder_path }/TORSO_1.png")
    # plt.show()



