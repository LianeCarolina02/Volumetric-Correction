import copy
import numpy as np
import open3d as o3d
import visualization as vis


def save_ply(pcd, folder, filename):
    filename = f"{folder}/{filename}.ply"  # Specify the filename with appropriate extension
    o3d.io.write_point_cloud(filename, pcd)

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)  # Adjusted standard deviation
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

mu, sigma = 0, 0.0005  # mean and standard deviation
voxel_size = 0.01

Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Torso = "Manequin/Mannequin_Torso_ASCII.ply"
Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"

target = o3d.io.read_point_cloud(Torso)
source = o3d.io.read_point_cloud(Breast)

source_noisy = apply_noise(source, mu, sigma)

folder = "Noise_ply"
filename = f"Breast_Noise_{sigma}"

save_ply(source_noisy, folder, filename)

Result_name = f"{folder}/{filename}.ply" 
Result = o3d.io.read_point_cloud(Result_name)

vis.draw_point_cloud(Result)