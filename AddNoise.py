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

mu, sigma = 0, 0.001  # mean and standard deviation
voxel_size = 0.01

Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Torso = "Manequin/Mannequin_Torso_ASCII.ply"
Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"

target = o3d.io.read_point_cloud(Torso)
source = o3d.io.read_point_cloud(Breast)

# source_noisy = apply_noise(source, mu, sigma)

folder = "Noise_ply"
filename_0 = f"Breast_local_1_Noise"
filename_1 = f"Breast_Noise_{sigma}"

# save_ply(source_noisy, folder, filename)

Result_name_0 = f"{folder}/{filename_0}.ply" 
Result_name_1 = f"{folder}/{filename_1}.ply" 

Result_0 = o3d.io.read_point_cloud(Result_name_0)
Result_1 = o3d.io.read_point_cloud(Result_name_1)

vis.draw_registration_result(Result_0, Result_1, transformation=np.identity(4))