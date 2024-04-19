import open3d as o3d
import numpy as np
import icp
import prepare_dataset as prd
import sys
import copy

Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Breast_Noise = "Noise_ply/Breast_Noise_0.005.ply" #isto está errado
Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"
Torso = "Manequin/Mannequin_Torso_ASCII.ply"
voxel_size = 0.01

np.set_printoptions(threshold=sys.maxsize)

def compute_correspondences(source, target, threshold, transformation):
    source_point = np.asarray(source.points)
    print(f"\n No. Points of the Source: {len(source_point)} \n")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, transformation)
    ev_points = np.asarray(evaluation.correspondence_set)
    return ev_points

def plot_correspondences(source, target, source_down, target_down, transformation):

    correspondences = compute_correspondences(source_down, target_down, 0.02, transformation)
    correspondences[:, 1] += len(np.asarray(source_down.points))

    corresponding_points_pcd1 = []
    corresponding_points_pcd2 = []

    for correspondence in correspondences:
        point_index_source = int(correspondence[0])
        point_index_target = int(correspondence[1])
        corresponding_points_pcd1.append(source.points[point_index_source])
        corresponding_points_pcd2.append(target.points[point_index_target])

    lines = []
    for i in range(len(corresponding_points_pcd1)):
        line = [[corresponding_points_pcd1[i][0], corresponding_points_pcd2[i][0]],
                [corresponding_points_pcd1[i][1], corresponding_points_pcd2[i][1]],
                [corresponding_points_pcd1[i][2], corresponding_points_pcd2[i][2]]]
        lines.append(line)


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

source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(Breast_Noise, Torso, voxel_size = voxel_size)

# Assuming your icp.vanilla_icp() function returns a transformation matrix
transformation = icp.vanilla_icp(source_down, target_down, 0.02).transformation
plot_correspondences(source, target, source_down, target_down, transformation)
