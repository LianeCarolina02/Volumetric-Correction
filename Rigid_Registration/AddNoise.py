import copy
import numpy as np
import open3d as o3d
import visualization as vis
from scipy.stats import multivariate_normal


def save_ply(pcd, folder, filename):
    filename = f"{folder}/{filename}.ply"  # Specify the filename with appropriate extension
    o3d.io.write_point_cloud(filename, pcd)

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)  # Adjusted standard deviation
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def noise(pcd, mu):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    chosen_points_idx = np.random.choice(len(points), size=50, replace=False)
    chosen_points = points[chosen_points_idx]
    for point in chosen_points:
        radius = np.random.uniform(0.005, 0.05)
        distances = np.linalg.norm(point - points, axis=1)
        points_in_radius = points[distances < radius]
        sigma = np.random.uniform(0.0005, 0.003)
        noise = np.random.normal(mu, sigma, size=points_in_radius.shape)
        points[distances < radius] += noise  
    noisy_pcd.points = o3d.utility.Vector3dVector(points)  
    return noisy_pcd