import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy


def fpfh(pcd, voxel_size):
    print("Computing Fast Points Feature Histogram...")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    radius_normal = voxel_size * 5
    downpcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downpcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print("FPFH Done")
    return pcd_fpfh.data

def save_feature_image(pcd, pcd_fpfh_data,voxel_size, folder, patient):
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    chosen_fpfh = pcd_fpfh_data[[5, 6, 15, 16, 26, 27, 28], :]
    maximum = chosen_fpfh.max()
    minimum = chosen_fpfh.min()
    num_features = len(chosen_fpfh)

    for i in range(num_features):
        print(f"Feature number {i}-th")
        fpfh = np.asarray(chosen_fpfh[i,:])
        colormap = 'inferno'

        fpfh_colors = plt.get_cmap(colormap )(
                (fpfh - minimum) / (maximum - minimum))

        fpfh_colors = fpfh_colors[:, :3]


        fpfh_pcd =o3d.geometry.PointCloud()
        fpfh_pcd.points = downpcd.points
        fpfh_pcd.normals = downpcd.normals
        fpfh_pcd.colors = o3d.utility.Vector3dVector(fpfh_colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        vis.add_geometry(fpfh_pcd)
        vis.run()
        
        # vis.capture_screen_image(f"{folder}/{patient}_Feature_{i}.png")
        vis.destroy_window()

