import open3d as o3d
import numpy as np
import visualization as vis 
import pprint
import filter
import RANSAC
import matplotlib.cm as cm
import prepare_dataset as prd
import visualization as vis
import matplotlib.pyplot as plt
import surface_acquisition as sa
import matplotlib.colors as mcolors

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
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
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
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
    vis.run()
    vis.destroy_window()

import icp
import copy


def first_transformation(digital_twin, surface_scan, downsample=10):
    _, _, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(digital_twin, surface_scan, voxel_size = downsample)

    source_down_copy = copy.deepcopy(source_down)
    transformation_ransac = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh,voxel_size = downsample,distance=50).transformation

    source_down.transform(transformation_ransac)

    icp_result = icp.vanilla_icp(source_down, target_down, 20)
    correspondence = np.asarray(icp_result.correspondence_set)

    transformation_icp = icp_result.transformation

    transformation = transformation_icp @ transformation_ransac

    return source_down_copy, target_down, transformation, correspondence

def read_point_clouds(id, scan_2=False):
    patient = "BR0" + f"{id}"
    surface_digital_twin = f"Pacients/{patient}/Final_Surface.ply"
    tumor_path = f"Pacients/BR0{id}/Segment_1.stl"
    simplified_mesh_path = f"Pacients/BR0{id}/try.stl"

    if id == 74 and scan_2:
        surface_scan = "Pacients/BR074/74/Scan 2.obj"
    else:
        surface_scan = f"Pacients/{patient}/{id}/{id}.obj"

    mesh_surface_scan = o3d.io.read_triangle_mesh(surface_scan)
    mesh_tumor = o3d.io.read_triangle_mesh(tumor_path)
    simplified_mesh = o3d.io.read_triangle_mesh(simplified_mesh_path)

    pcd_scan = mesh_surface_scan.sample_points_uniformly(number_of_points=500000)
    pcd_tumor = mesh_tumor.sample_points_uniformly(number_of_points=10000)
    pcd_simplified_mesh = simplified_mesh.sample_points_uniformly(number_of_points=200000)
    _, rotation, vector = sa.rotation_pcd(pcd_simplified_mesh)
    pcd_tumor.rotate(rotation)
    pcd_tumor.translate(vector)

    pcd_digital_twin = o3d.io.read_point_cloud(surface_digital_twin)

    if id in {63, 64, 66, 67, 69, 71, 73, 74, 76}:
        pcd_scan.rotate([[1,0,0],
                         [0,np.cos(90), -np.sin(90)],
                         [0,np.sin(90), np.cos(90)]])
        
        pcd_scan.rotate([[-np.cos(45),np.sin(45),0],
                         [-np.sin(45),-np.cos(45),0],
                         [0,0,1]])
    elif id in {61, 62, 65, 68, 74}:
        pcd_scan.rotate([[1,0,0],
                         [0,-np.cos(-5), np.sin(-5)],
                         [0,-np.sin(-5), -np.cos(-5)]])
        
        pcd_scan.rotate([[np.cos(135),0,-np.sin(135)],
                         [0,1,0],
                         [np.sin(135),0,np.cos(135)  ]])
        
        pcd_scan.rotate([[1,0,0],
                         [0,0,1],
                         [0,-1,0]])

    return pcd_scan, pcd_digital_twin, pcd_tumor

def plot_histogram_distances(patients_ids):
    distances_dict = {}
    distances_dict_down = {}

    for id in patients_ids:
        target, source,_ = read_point_clouds(id=id)
        source_down, target_down, first_trans, correspondence_down = first_transformation(source, target)
        source.transform(first_trans)
        source_down.transform(first_trans)

        evaluation = o3d.pipelines.registration.evaluate_registration(source, target, 20, np.identity(4))
        correspondence_set = np.asarray(evaluation.correspondence_set)
        
        correspondence_set_copy = copy.deepcopy(correspondence_set)
        correspondence_set_down_copy = copy.deepcopy(correspondence_down)

        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        source_points_down = np.asarray(source_down.points)
        target_points_down = np.asarray(target_down.points)

        distances = []
        distances_down = []
        target_normals = np.asarray(target.normals)
        target_normals_down = np.asarray(target_down.normals)

        for i, (source_index, target_index) in enumerate(correspondence_set_copy):
            source_point = source_points[source_index]
            target_point = target_points[target_index]
            distance = np.linalg.norm(source_point - target_point)
            if np.dot(source_point - target_point, target_normals[i]) > 0:
                distances.append(distance)
            else: 
                distances.append(-distance)

        # Store distances in the dictionary
        distances_dict[id] = distances

        for i, (source_index, target_index) in enumerate(correspondence_set_down_copy):
            source_point = source_points_down[source_index]
            target_point = target_points_down[target_index]
            distance = np.linalg.norm(source_point - target_point)
            if np.dot(source_point - target_point, target_normals_down[i]) > 0:
                distances_down.append(distance)
            else: 
                distances_down.append(-distance)

        # Store distances in the dictionary
        distances_dict[id] = distances
        distances_dict_down[id] = distances_down

    # Plot combined histogram
    plt.figure(figsize=(10, 6))
    hist_sums = {}
    for idx, patient_id in enumerate(distances_dict):
        # hist_full, bins_full = np.histogram(distances_dict[patient_id], bins=200)
        hist_down, bins_down = np.histogram(distances_dict_down[patient_id], bins=50)

        # max_hist_full = np.max(distances_dict[patient_id])
        # min_hist_full = np.min(distances_dict[patient_id])
        # max_hist_down = np.max(distances_dict_down[patient_id])
        # min_hist_down = np.min(distances_dict_down[patient_id])

        # bin_size_full = (max_hist_full - min_hist_full)/200
        # bin_size_down = (max_hist_down - min_hist_down)/200

        # print(f"bin size: {bin_size_full}")
        # print(f"bin size: {bin_size_down}")
        
        # sum_full = np.sum(hist_full)
        # sum_down = np.sum(hist_down)

        # Store the integer sums in the dictionary
        # hist_sums[patient_id] = {
        #     'sum_full': int(sum_full),
        #     'sum_down': int(sum_down)
        # }

        # print(f'Patient {patient_id} - Full Histogram Sum: {sum_full}')
        # print(f'Patient {patient_id} - Downsampled Histogram Sum: {sum_down}')

        # plt.hist(bins_full[:-1], bins_full, weights=hist_full/sum_full, alpha=0.7, label=f'{patient_id} - evaluation from open3d')
        plt.hist(bins_down[:-1], bins_down, weights=hist_down, alpha=0.7, label=f'{patient_id} - icp')



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
    for id in ids:
        target, source,_ = read_point_clouds(id=id)
        source_down, target_down, transformation, correspondence = first_transformation(source, target, downsample=10)

        num_rows_to_keep = correspondence.shape[0] // 40

        # Randomly select rows to keep
        indices_to_keep = np.random.choice(correspondence.shape[0], num_rows_to_keep, replace=False)

        # Keep only the selected rows
        correspondence = correspondence[indices_to_keep]
        
        correspondence_copy = copy.deepcopy(correspondence)
        correspondence[:, 1] += len(np.asarray(source_down.points))

        source_aligned_down = source_down.transform(transformation)
        source_aligned_down_translated = source_aligned_down.translate([0, 0,300])
        target_down_translated = target_down.translate([0, 0, 0])

        source_aligned = source.transform(transformation)

        # Translate the source and target point clouds
        source_aligned_translated = source_aligned.translate([0, 0,300])
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
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
        for sphere_source in spheres_source:
            vis.add_geometry(sphere_source)
        for sphere_target in spheres_target:
            vis.add_geometry(sphere_target)
        
        vis.run()
        vis.destroy_window()
        return source_temp + target_temp + line_set + spheres_source + spheres_target

def plot_distances(ids, evaluation_correspondence=True):
    for id in ids:
        target, source,_ = read_point_clouds(id=id)

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_down, target_down, transformation, correspondence_icp = first_transformation(source, target)

        source_temp.transform(transformation)

        source.transform(transformation)

        source_down_copy = copy.deepcopy(source_down)
        source_down_copy.transform(transformation)
        

        if evaluation_correspondence:
            evaluation = o3d.pipelines.registration.evaluate_registration(source, target, 15, np.identity(4))
            correspondence_set = np.asarray(evaluation.correspondence_set)
            target_normals = np.asarray(target.normals)
            source_points = np.asarray(source.points)
            target_points = np.asarray(target.points)
        else:
            correspondence_set = correspondence_icp
            target_normals = np.asarray(target_down.normals)
            source_points = np.asarray(source_down.points)
            target_points = np.asarray(target_down.points)

        distances = []
        target_temp_points = []

        for i, (source_index, target_index) in enumerate(correspondence_set):
            source_point = source_points[source_index]
            target_point = target_points[target_index]
            distance = np.linalg.norm(source_point - target_point) 
            
            if np.dot(source_point - target_point, target_normals[i]) > 0:
                distances.append(distance)
            else: 
                distances.append(-distance)

            target_temp_points.append(target_point)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(target_temp_points)
        distances = np.array(distances)
        distances_pos_log = (distances[distances >= 0])
        distances_neg_log = (distances[distances < 0])
        max_dist = max(np.max(distances_pos_log), np.max(distances_neg_log))
        min_dist = min(np.min(distances_pos_log), np.min(distances_neg_log))

        red = np.array([[1, 0, 0]])
        blue = np.array([[0, 0, 1]])
        white = np.array([[1, 1, 1]])

        colors_pos = ((distances_pos_log - 0) / (max_dist - 0)).reshape(-1, 1) @ red + \
                    ((max_dist - distances_pos_log) / (max_dist - 0)).reshape(-1, 1) @ white
        
        colors_neg = ((distances_neg_log - min_dist) / (0 - min_dist)).reshape(-1, 1) @ white + \
                    ((0 - distances_neg_log) / (0 - min_dist)).reshape(-1, 1) @ blue
        
        print(distances)
        colors = np.zeros((distances.shape[0], 3))
        colors[distances >= 0] = colors_pos
        colors[distances < 0] = colors_neg

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(target_temp_points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)


        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        source_temp.translate([500, 0, 0])
        target_temp.translate([500,0,0])
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
        vis.add_geometry(point_cloud)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
        vis.run()
        vis.destroy_window()
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue, White, Red
        cmap_name = 'custom_color_map'
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

        # Create some data
        print(min_dist)
        print(max_dist)
        data = np.random.randint(min_dist, max_dist, size=(10,10))

        # Plot the data with a colorbar
        plt.imshow(data, cmap=custom_cmap, vmin=min_dist, vmax=max_dist)
        cbar = plt.colorbar()

        # Define the ticks and labels for the colorbar
        cbar.set_ticks([min_dist, 0, max_dist])
        cbar.set_ticklabels([str(min_dist), str(0), str(max_dist)])

        plt.show()
        


if __name__ == "__main__":
    all_ids = [61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 76]

    all_idsx = [[61], [62], [63], [64], [65], [66], [67], [68], [69], [71], [73], [74], [76]]
    ids = [74]#, 68]#, 74]
    # graph_twin, nodes_twin = graph_and_nodes(source, 1, 10)
    # graph_scan, nodes_scan = graph_and_nodes(target, 1, 10)

    # visualize_graph(nodes_twin, graph_twin)
    # visualize_graph(nodes_scan, graph_scan)

    # plot_distances(ids, True)
    # pcd = plot_correspondences(ids)
    plot_histogram_distances(ids)