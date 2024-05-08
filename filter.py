import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return inlier_cloud


def guided_filter(pcd, radius, epsilon):

    pcd_copy = copy.deepcopy(pcd)
    kdtree = o3d.geometry.KDTreeFlann(pcd_copy)
    points_copy = np.array(pcd_copy.points)
    points = np.asarray(pcd_copy.points)
    num_points = len(pcd_copy.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd_copy.points[i], radius)
        if k < 5:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b

    pcd_copy.points = o3d.utility.Vector3dVector(points_copy)
    pcd_copy.paint_uniform_color([0.8, 0.8, 0.8])
    pcd.paint_uniform_color([1, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, pcd_copy])

    return 

def median_filter(pcd, radius):
    pcd_copy = copy.deepcopy(pcd)
    kdtree = o3d.geometry.KDTreeFlann(pcd_copy)
    points_copy = np.array(pcd_copy.points)
    points = np.asarray(pcd_copy.points)
    num_points = len(pcd_copy.points)

    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(pcd_copy.points[i], radius)
        if k < 12:
            continue
        
        neighbors = points[idx, :]
        median = np.median(neighbors, axis=0)

        points_copy[i] = median


    pcd_copy.points = o3d.utility.Vector3dVector(points_copy)
    pcd_copy.paint_uniform_color([1, 0, 0])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    
    o3d.visualization.draw_geometries([pcd_copy, pcd])

    pcd_copy.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pcd_copy])

    return pcd_copy