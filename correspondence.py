import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import icp
import prepare_dataset as prd

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
    print(f"\n No. Points of the Source: {len(source_point)} \n")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    ev_points = np.asarray(evaluation.correspondence_set)
    return ev_points

def plot_correspondences(source, target, source_down, target_down, transformation):
    """
    Plot correspondences between source and target point clouds.
    """
    correspondences = compute_correspondences(source_down, target_down, 0.02, transformation)
    correspondences[:, 1] += len(np.asarray(source_down.points))

    source_aligned_down = source_down.transform(transformation)
    source_aligned_down_translated = source_aligned_down.translate([0.25, 0, 0])
    target_down_translated = target_down.translate([-0.25, 0, 0])

    source_aligned = source.transform(transformation)

    # Translate the source and target point clouds
    source_aligned_translated = source_aligned.translate([0.25, 0, 0])
    target_translated = target.translate([-0.25, 0, 0])
    source_temp = copy.deepcopy(source_aligned_translated)
    target_temp = copy.deepcopy(target_translated)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Reverse the order of points for the line set
    reversed_correspondences = correspondences[:, ::-1]

    # Translate the lines to match the reversed order of points
    source_points_translated = np.asarray(source_aligned_down_translated.points)
    target_points_translated = np.asarray(target_down_translated.points)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack((source_points_translated, target_points_translated))),
        lines=o3d.utility.Vector2iVector(reversed_correspondences)
    )

    # Visualize
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set], width=1600, height=1200)

import numpy as np
import matplotlib.pyplot as plt

# Define the distances function
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

if __name__ == '__main__':

    BREAST = "Manequin/Mannequin_Breast_ASCII.ply"
    FASCIA = "Manequin/Mannequin_Fascia_ASCII.ply"
    TORSO = "Manequin/Mannequin_Torso_ASCII.ply"
    VOXEL_SIZE = [0.01, 0.01, 0.01, 0.01, 0.015, 0.02]  # Assuming you have this defined

    SIGMAS = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    SOURCES_BREAST = [BREAST] + [f"Noise_ply/Breast_Noise_{sigma}.ply" for sigma in SIGMAS]
    SOURCES_FASCIA = [FASCIA] + [f"Noise_ply/Fascia_Noise_{sigma}.ply" for sigma in SIGMAS]

    source_types = [SOURCES_BREAST, SOURCES_FASCIA]

    for source_type in source_types:
        all_histograms = []
        all_bins = []

        for i, idx in enumerate(source_type):
            source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(idx, TORSO, voxel_size=VOXEL_SIZE[i])

            transformation = icp.vanilla_icp(source_down, target_down, 0.02).transformation

            hist, bins = distances(source_down, target_down, transformation)
            
            all_histograms.append(hist)
            all_bins.append(bins)

        all_histograms = all_histograms[::-1]
        all_bins = all_bins[::-1]

        for bins, histogram in zip(all_bins, all_histograms):
            plt.hist(bins[:-1], weights=histogram, bins=bins, alpha=0.8) # Plot each histogram with transparency and reversed order

        names = [f"Original", "Noise 0.0001", "Noise 0.0005", "Noise 0.001", "Noise 0.005", "Noise 0.01"]
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Histograms of Euclidean Distances for {source_type}')
        plt.legend(names[::-1])  # Add legend with labels for each histogram
        plt.show()
