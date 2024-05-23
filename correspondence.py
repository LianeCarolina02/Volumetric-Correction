import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import icp
import prepare_dataset as prd
import studies

BREAST = "Manequin/Mannequin_Breast_ASCII.ply"
FASCIA = "Manequin/Mannequin_Fascia_ASCII.ply"
TORSO = "Manequin/Mannequin_Torso_ASCII.ply"
VOXEL_SIZE = 0.01

np.set_printoptions(threshold=sys.maxsize)

def compute_correspondences(source, target, threshold, transformation):
    """
    Compute correspondences between source and target point clouds.
    """
    source_point = np.asarray(source.points)
    target_point = np.asarray(target.points)
    print(f"\n No. Points of the Source: {len(source_point)} \n")
    print(f"\n No. Points of the Target: {len(target_point)} \n")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    ev_points = np.asarray(evaluation.correspondence_set)
    return ev_points


import numpy as np
import matplotlib.pyplot as plt

def distances(source_down, target_down, transformation):
    """
    Compute Euclidean distances between corresponding points in two point clouds.
    """
    correspondences = compute_correspondences(source_down, target_down, 0.02, transformation)

    source_down.transform(transformation)
    corresponding_points_pcd1 = []
    corresponding_points_pcd2 = []

    for correspondence in correspondences:
        point_index_source = int(correspondence[0])
        point_index_target = int(correspondence[1])
        corresponding_points_pcd1.append(source_down.points[point_index_source])
        corresponding_points_pcd2.append(target_down.points[point_index_target])

    euclidean_distances = np.sqrt(np.sum((np.array(corresponding_points_pcd2) - np.array(corresponding_points_pcd1))**2, axis=1))
    
    hist, bins = np.histogram(euclidean_distances, bins=40)
    return hist, bins

