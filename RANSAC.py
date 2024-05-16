import numpy as np
import open3d as o3d
#from stl import mesh
import matplotlib.pyplot as plt
import prepare_dataset as prd
import visualization as vis
import time


def global_registration_ransac(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, distance):
    distance_threshold = voxel_size * distance
    # print(":: RANSAC registration")
    # print(":: distance threshold %.3f." % distance_threshold)

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
    
    # print(f"\n Result Ransac: \n \n Transformation: \n {result.transformation} \n \n Fitness: {result.fitness} \n \n RMSE: {result.inlier_rmse}")
    return result

if __name__ == '__main__':
    voxel_size = 0.01
    distance = 5
    original = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    sigma = 0.01
    # folder = "output_folder/Noise"
    folder = "output_folder/original"
    threshold = 0.02 #perceber

    source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(Breast, Torso, voxel_size = voxel_size)

    # ransac = global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh,voxel_size=voxel_size,distance=distance)
    
    
    source_point_cloud = o3d.io.read_point_cloud(Breast)

    # Create a condition
    condition = source_fpfh.data[6, :] > 150


    

