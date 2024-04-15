import numpy as np
import open3d as o3d
#from stl import mesh
import matplotlib.pyplot as plt
import copy
import os
import datetime

current_datetime = datetime.datetime.now()
datetime = current_datetime.strftime("%d_%m__%H_%M_%S")

Breast_ascii = "Manequin/Mannequin_Breast_ASCII.ply"
Torso_ascii = "Manequin/Mannequin_Torso_ASCII.ply"

def preprocessing(pcd, voxel_size = 0.01):
    print(":: Downsampling with voxel %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    
    radius_normal = voxel_size * 2
    print(":: Normal with search radius %.3f." % radius_normal)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Computation of FPFH feature with search radius %.3f." % radius_feature)

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size=0.01):
    Breast = "Manequin/Mannequin_Breast_ASCII.ply"
    Torso = "Manequin/Mannequin_Torso_ASCII.ply"

    target = o3d.io.read_point_cloud(Torso)
    source = o3d.io.read_point_cloud(Breast)

    print(":: Prepare Source Dataset")
    source_down, source_fpfh = preprocessing(source, voxel_size)
    print(":: Prepare Target Dataset")
    target_down, target_fpfh = preprocessing(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh
def draw_point_cloud(pcd):
    points = np.asarray(pcd.points)
    color_map = plt.get_cmap('Blues')

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    colors = color_map(((1 - (points[:, 2] - z_min) / (z_max - z_min)) + 0.5))[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], width=1600, 
                                  height=1200, point_show_normal=False, 
                                  mesh_show_wireframe=False, mesh_show_back_face=False)


def draw_registration_result(source, target, transformation, filename):
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.get_render_option().point_show_normal = False
    vis.get_render_option().mesh_show_wireframe = False
    vis.get_render_option().mesh_show_back_face = False
    vis.run()
    
    vis.capture_screen_image(filename)
    
    # Close the visualizer
    vis.destroy_window()



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, distance):
    distance_threshold = voxel_size * distance
    print(":: RANSAC registration")
    print(":: distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == '__main__':
    voxel_size = 0.01
    distance = 5
    original = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size = voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size=voxel_size,distance=distance)
    print(f"\n Result Ransac: \n \n Transformation: \n {result_ransac.transformation} \n \n Fitness: {result_ransac.fitness} \n \n RMSE: {result_ransac.inlier_rmse}")

    draw_registration_result(source, target, result_ransac.transformation, f"output_folder/registration_vs{voxel_size}_{datetime}.png")
