import copy
import numpy as np
import open3d as o3d
import RANSAC



voxel_size = 0.01
Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Torso = "Manequin/Mannequin_Torso_ASCII.ply"
Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"

target = o3d.io.read_point_cloud(Torso)
# source = o3d.io.read_point_cloud(Breast)
source = o3d.io.read_point_cloud(Fascia)

def save_ply(pcd, sigma):
    filename = f"Noise_ply/Breast_Noise_{sigma}.ply"  # Specify the filename with appropriate extension
    o3d.io.write_point_cloud(filename, pcd)

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)  # Adjusted standard deviation
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


mu, sigma = 0, 0.0001  # mean and standard deviation
source_noisy = apply_noise(source, mu, sigma)
save_ply(source_noisy, sigma)
RANSAC.draw_point_cloud(source_noisy)