import copy
import numpy as np
import open3d as o3d
import open

voxel_size = 0.01
Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Torso = "Manequin/Mannequin_Torso_ASCII.ply"

target = o3d.io.read_point_cloud(Torso)
source = o3d.io.read_point_cloud(Breast)

def use_o3d(pts, write_text):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud("Manequin/Breast_noise.ply", pcd, write_ascii=write_text)

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma * 0.001, size=points.shape)  # Adjusted standard deviation
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


mu, sigma = 0, 1  # mean and standard deviation
source_noisy = apply_noise(source, mu, sigma)
use_o3d(source_noisy, True)
open.draw_point_cloud(source_noisy)