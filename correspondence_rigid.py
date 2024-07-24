import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import RANSAC

import icp
import prepare_dataset as prd


def evaluation_function(source, target, threshold):
    """
    Compute correspondences between source and target point clouds.
    """
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, np.identity(4))
    ev_points = np.asarray(evaluation.correspondence_set)
    return ev_points


def percentage_of_correspondences(source, target, threshold, percentage):

    matrix_correspondences = evaluation_function(source, target, threshold)

    num_rows_to_keep = int(np.round(matrix_correspondences.shape[0] * percentage, 0))

    indices_to_keep = np.random.choice(matrix_correspondences.shape[0], num_rows_to_keep, replace=False)

    matrix_correspondences = matrix_correspondences[indices_to_keep]
        
    return matrix_correspondences

def select_clusters_based_on_histogram(threshold_histogram, feature_target, feature_source, print_shapes=False):

    cluster_1_target = np.where(feature_target < threshold_histogram)[0]
    cluster_2_target = np.where(feature_target >= threshold_histogram)[0]

    cluster_1_source = np.where(feature_source < threshold_histogram)[0]
    cluster_2_source = np.where(feature_source >= threshold_histogram)[0]

    if print_shapes:

        print("Cluster sizes:")
        print("Cluster 1 Source:", cluster_1_source.shape)
        print("Cluster 2 Source:", cluster_2_source.shape)
        print("Cluster 1 Target:", cluster_1_target.shape)
        print("Cluster 2 Target:", cluster_2_target.shape)


    return cluster_1_source, cluster_1_target, cluster_2_source, cluster_2_target

def filtered_correspondence_set_based_on_clusters(cluster_2_source, cluster_2_target, correspondence_set):
    correspondence_set_copy = copy.deepcopy(correspondence_set)

    mask = np.zeros(correspondence_set_copy.shape[0], dtype=bool)

    source_indices = correspondence_set_copy[:, 0]
    mask |= np.isin(source_indices, cluster_2_source)

    target_indices = correspondence_set_copy[:, 1]
    mask |= np.isin(target_indices, cluster_2_target)

    filtered_correspondence_set = correspondence_set_copy[mask]

    return filtered_correspondence_set

    
def plot_manequim_correspondences(source_path, target_path, voxel_size):
    """
    Plot correspondences between source and target point clouds.

    """
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    threshold = 2 * voxel_size

    transformation = rigid_registration_manequim(source, target, voxel_size, threshold)

    source.transform(transformation)
    
    correspondences = percentage_of_correspondences(source, target, threshold, percentage=0.001)
    correspondences_copy = copy.deepcopy(correspondences)
    correspondences[:, 1] += len(np.asarray(source.points))

    source.translate([0, 0, 0.15])
    target.translate([0, 0, -0.15])

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Reverse the order of points for the line set
    reversed_correspondences = correspondences[:, ::-1]

    # Translate the lines to match the reversed order of points
    source_points_translated = np.asarray(source.points)
    target_points_translated = np.asarray(target.points)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack((source_points_translated, target_points_translated))),
        lines=o3d.utility.Vector2iVector(reversed_correspondences)
    )

    source_idx_points = correspondences_copy[:, 0]
    target_idx_points = correspondences_copy[:, 1]

    source_chosen_points = source_points_translated[source_idx_points]
    target_chosen_points = target_points_translated[target_idx_points]
    
    spheres_source = []
    spheres_target = []

    for s_point in source_chosen_points:
        sphere_source = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere_source.translate(s_point)
        sphere_source.paint_uniform_color([1, 0, 0])  # Red color to highlight
        spheres_source.append(sphere_source)

    for t_point in target_chosen_points:
        sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        sphere_target.translate(t_point)
        sphere_target.paint_uniform_color([0, 0, 1])  # Blue color to highlight
        spheres_target.append(sphere_target)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.add_geometry(line_set)
    for sphere_source in spheres_source:
        vis.add_geometry(sphere_source)
    for sphere_target in spheres_target:
        vis.add_geometry(sphere_target)
    
    vis.run()
    vis.destroy_window()

import numpy as np
import matplotlib.pyplot as plt


def rigid_registration_manequim(source, target, voxel_size, threshold, visualize_rigid_transformation=False):

    source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(source, target, voxel_size)
    global_transformation = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size, distance=5)
    refinement_transformation = icp.vanilla_icp(source_down, target_down, threshold, global_transformation.transformation)
    transformation = refinement_transformation.transformation

    if visualize_rigid_transformation:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        vis.run

    return transformation

def manequim_correspondences_histogram_distances(source_path, target_path, voxel_size, display_histogram):
    """
    Compute Euclidean distances between corresponding points in two point clouds.
    """

    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    threshold = 2 * voxel_size

    transformation = rigid_registration_manequim(source, target, voxel_size, threshold)

    source.transform(transformation)

    correspondences = evaluation_function(source, target, threshold)

    corresponding_points_pcd1 = []
    corresponding_points_pcd2 = []

    for correspondence in correspondences:
        point_index_source = int(correspondence[0])
        point_index_target = int(correspondence[1])
        corresponding_points_pcd1.append(source.points[point_index_source])
        corresponding_points_pcd2.append(target.points[point_index_target])

    euclidean_distances = np.sqrt(np.sum((np.array(corresponding_points_pcd2) - np.array(corresponding_points_pcd1))**2, axis=1))
    
    hist, bins = np.histogram(euclidean_distances, bins=200)

    if display_histogram:
        plt.hist(bins[:-1], weights=hist, bins=bins, alpha=0.8) # Plot each histogram with transparency and reversed order
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Histograms of Euclidean Distances')
        plt.show()

    return hist, bins


