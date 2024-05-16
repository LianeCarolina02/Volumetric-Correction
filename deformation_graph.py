import open3d as o3d
import numpy as np
import visualization as vis 
import pprint
import filter
import RANSAC
import matplotlib.cm as cm
import prepare_dataset as prd
import matplotlib.pyplot as plt


def graph_and_nodes(pcd, voxel_size_1, voxel_size_2):

    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size_1)

    pcd_points = np.asarray(downpcd.points)
    num_points = pcd_points.shape[0]

    nodes = pcd.voxel_down_sample(voxel_size=voxel_size_2)
    nodes_points = np.asarray(nodes.points)
    num_nodes = nodes_points.shape[0]
    nodes.paint_uniform_color([0.8,0.2,0.8])

    print(f"The original Point Cloud has {num_points} \nThe Number of nodes are {num_nodes}\nThe Ratio is {round(num_nodes/num_points *100, 2)}%")

    distances = nodes.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radius = 3 * avg_distance

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        nodes, o3d.utility.DoubleVector([radius, radius * 2]))

    mesh.paint_uniform_color([0,1,1])

    graph = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    return graph, nodes

def lines_from_graph(graph):
    lines = np.asarray(graph.lines)
    return lines

def visualize_graph(nodes, graph):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(nodes)
    vis.add_geometry(graph)
    vis.run()
    vis.destroy_window()



def scan_and_twin(pcd_source, pcd_target, transformation):
    source_temp = copy.deepcopy(pcd_source)
    target_temp = copy.deepcopy(pcd_target)
    source_temp.transform(transformation)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.run()
    vis.destroy_window()

import icp
import copy


def first_transformation(digital_twin, surface_scan):
    _, _, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(digital_twin, surface_scan, voxel_size = 10)

    source_down_copy = copy.deepcopy(source_down)
    transformation_ransac = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh,voxel_size = 10,distance=50).transformation

    source_down.transform(transformation_ransac)

    icp_result = icp.vanilla_icp(source_down, target_down, 10)
    correspondence = np.asarray(icp_result.correspondence_set)

    transformation_icp = icp_result.transformation

    transformation = transformation_icp @ transformation_ransac

    return source_down_copy, target_down, transformation, correspondence

def read_point_clouds(id):
    patient = "BR0" + f"{id}"
    surface_digital_twin = f"Pacients/{patient}/Final_Surface.ply"
    surface_scan = f"Pacients/{patient}/{id}/{id}.obj"

    mesh_surface_scan = o3d.io.read_triangle_mesh(surface_scan)

    pcd_scan = mesh_surface_scan.sample_points_uniformly(number_of_points=500000)
    pcd_digital_twin = o3d.io.read_point_cloud(surface_digital_twin)

    return pcd_scan, pcd_digital_twin

def plot_histogram_distances(patients_ids):
    distances_dict = {}
    for id in patients_ids:
        pcd_scan, pcd_digital_twin = read_point_clouds(id=id)
        sourcedown, targetdown, first_trans, correspondence_set = first_transformation(pcd_digital_twin, pcd_scan)
        sourcedown.transform(first_trans)
        correspondence_set_copy = copy.deepcopy(correspondence_set)

        distances = []
        for source_index, target_index in correspondence_set_copy:
            source_point = np.asarray(sourcedown.points)[source_index]
            target_point = np.asarray(targetdown.points)[target_index]
            distance = np.linalg.norm(source_point - target_point)
            distances.append(distance)

        # Store distances in the dictionary
        distances_dict[id] = distances

    # Plot combined histogram
    plt.figure(figsize=(10, 6))
    for patient, distances in distances_dict.items():
        plt.hist(distances, bins=50, alpha=0.7, label=patient)

    plt.title('Histogram of Distances from Corresponding Points in Breast')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correspondences(ids):
    """
    Plot correspondences between source and target point clouds.
    """

    target, source = read_point_clouds(id=ids)
    source_down, target_down, transformation, correspondence = first_transformation(source, target)

    num_rows_to_keep = correspondence.shape[0] // 40

    # Randomly select rows to keep
    indices_to_keep = np.random.choice(correspondence.shape[0], num_rows_to_keep, replace=False)

    # Keep only the selected rows
    correspondence = correspondence[indices_to_keep]
    
    correspondence_copy = copy.deepcopy(correspondence)
    correspondence[:, 1] += len(np.asarray(source_down.points))

    source_aligned_down = source_down.transform(transformation)
    source_aligned_down_translated = source_aligned_down.translate([0, 300, -200])
    target_down_translated = target_down.translate([0, 0, 0])

    source_aligned = source.transform(transformation)

    # Translate the source and target point clouds
    source_aligned_translated = source_aligned.translate([0, 300, -200])
    target_translated = target.translate([0, 0, 0])
    source_temp = copy.deepcopy(source_aligned_translated)
    target_temp = copy.deepcopy(target_translated)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Reverse the order of points for the line set
    reversed_correspondences = correspondence[:, ::-1]

    # Translate the lines to match the reversed order of points
    source_points_translated = np.asarray(source_aligned_down_translated.points)
    target_points_translated = np.asarray(target_down_translated.points)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack((source_points_translated, target_points_translated))),
        lines=o3d.utility.Vector2iVector(reversed_correspondences)
    )
    
    source_idx_points = correspondence_copy[:, 0]
    target_idx_points = correspondence_copy[:, 1]

    source_chosen_points = source_points_translated[source_idx_points]
    target_chosen_points = target_points_translated[target_idx_points]

    # Create sphere meshes for correspondence points
    spheres_source = []
    spheres_target = []

    for s_point in source_chosen_points:
        sphere_source = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere_source.translate(s_point)
        sphere_source.paint_uniform_color([1, 0, 0])  # Red color to highlight
        spheres_source.append(sphere_source)

    for t_point in target_chosen_points:
        sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere_target.translate(t_point)
        sphere_target.paint_uniform_color([0, 0, 1])  # Blue color to highlight
        spheres_target.append(sphere_target)

    # Visualize
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


if __name__ == "__main__":
    ids = [61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 76]

    target, source = read_point_clouds(id=62)

    source_down, target_down, transformation, correspondence = first_transformation(source, target)

    source.transform(transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, 20, np.identity(4))
    correspondence_set = np.asarray(evaluation.correspondence_set)

    distances = []
    target_temp_points = []
    print(correspondence_set.shape)

    for source_index, target_index in correspondence_set:
        source_point = np.asarray(source.points)[source_index]
        target_point = np.asarray(target.points)[target_index]
        distance = np.linalg.norm(target_point - source_point)  # Euclidean distance
        distances.append(distance)
        target_temp_points.append(target_point)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(target_temp_points)

    max_distance = max(distances)
    colors = [[d / max_distance, 0, 1 - d / max_distance] for d in distances]  # Blue for closer points, red for farther points

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(target_temp_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

        



    # graph_twin, nodes_twin = graph_and_nodes(source, 1, 10)
    # graph_scan, nodes_scan = graph_and_nodes(target, 1, 10)

    # visualize_graph(nodes_twin, graph_twin)
    # visualize_graph(nodes_scan, graph_scan)

    # plot_histogram_distances(ids)

    # plot_correspondences(74)