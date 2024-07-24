
import open3d as o3d
import copy
import numpy as np
import surface_acquisition
import cv2
import surface_acquisition as surface
import deformation_graph as dg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def capture_video(mesh_path=None, pcd_path=None, pcd=None, output_file="this.avi", width=640, height=480, fps=30):
    if mesh_path is not None:
        pcd = o3d.io.read_triangle_mesh(mesh_path)
        pcd.compute_vertex_normals()
    elif pcd_path is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        

    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    # view_control.set_zoom(0.35)

    # Capture frames
    for _ in range(3000):  # Rotate 360 degrees
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        view_control.rotate(5.0, 0.0)

        # Capture the current frame
        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    # Release resources
    vis.destroy_window()
    out.release()
    cv2.destroyAllWindows()


def capture_frame(vis, out):
    # Capture the current screen
    image = vis.capture_screen_float_buffer(False)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
    out.write(image)

def video_set_up(name:str, fps=30.0, width=1920, height=1080):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f'{name}.avi', fourcc, fps, (width, height))
    return out


def plot_errors(errors, alpha):
    plt.figure(figsize=(10, 6))

    errors = np.array(errors)
    total_errors = errors[:,0]
    align_errors = errors[:,1]
    reg_errors = errors[:,2]
    # rot_errors = errors[:, 3]

    plt.plot(total_errors, marker='o', linestyle='-', label=f'Total Error alpha={alpha}')
    plt.plot(align_errors, marker='x', linestyle='--', label=f'Align Error alpha={alpha}')
    plt.plot(reg_errors, marker='s', linestyle='-.', label=f'Reg Error alpha={alpha}')
    # plt.plot(rot_errors, marker='d', linestyle=':', label=f'Rot Error alpha={alphas[i]}, beta={betas[i]}')

    plt.title('Error Vector Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Error Value')
    plt.grid(True)
    plt.legend()
    plt.show()



def plot_after_non_rigid_registration(source_transformed, tumor_points, original_tumor, target, graph, EG, original_source, show_twin=True):
    
    original_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    original_tumor.paint_uniform_color([0,0,1])

    target_copy = copy.deepcopy(target)
    target_copy.paint_uniform_color([0, 0.651, 0.929])
    tumor_copy_1 = copy.deepcopy(original_tumor)
    tumor_copy_2 = copy.deepcopy(original_tumor)

    target_copy.translate((500,0,0))
    tumor_copy_1.translate([500,0,0])

    new_source_transformed = o3d.geometry.PointCloud()
    new_source_transformed.points = o3d.utility.Vector3dVector(source_transformed)
    new_source_transformed.paint_uniform_color([1, 0.706, 0])

    pcd_vg = o3d.geometry.PointCloud()
    pcd_vg.points = o3d.utility.Vector3dVector(graph)

    new_pcd_tumor = o3d.geometry.PointCloud()
    new_pcd_tumor.points = o3d.utility.Vector3dVector(tumor_points)
    new_pcd_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    new_pcd_tumor.paint_uniform_color((1,0,0))

    edges = dg.create_edges(graph, EG)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(graph)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    new_pcd_twin_copy = copy.deepcopy(new_source_transformed)
    new_pcd_twin_copy.translate((500,0,0))

    new_pcd_tumor_copy = copy.deepcopy(new_pcd_tumor)
    new_pcd_tumor_copy.translate((500,0,0))
    

    if show_twin:
        o3d.visualization.draw_geometries([target_copy, original_source, pcd_vg, line_set, new_pcd_twin_copy, new_pcd_tumor_copy, new_pcd_tumor, tumor_copy_1,tumor_copy_2])
    else:
        o3d.visualization.draw_geometries([target_copy, pcd_vg, line_set, new_pcd_twin_copy, new_pcd_tumor_copy, new_pcd_tumor, tumor_copy_1,tumor_copy_2])

def plot_tumor(pcd_tumor_original, all_tumor_final_points, bs):
    geometries = []  # List to store all point clouds
    colors = [[1,0,0], [0,1,0], [1,1,0], [0,1,1]]  # List to store colors for legend
    color_name = ["Red", "Green", "Yellow", "Cyan"]
    
    # Plot each point cloud with a different color
    for idx, tumor_final_points in enumerate(all_tumor_final_points):
        pcd_tumor = o3d.geometry.PointCloud()
        pcd_tumor.points = o3d.utility.Vector3dVector(tumor_final_points)
        color = colors[idx]
        print(f"b = {bs[idx]} have the color {color_name[idx]}")
        pcd_tumor.paint_uniform_color(color)
        pcd_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        geometries.append(pcd_tumor)
        colors.append(color)
    
    # Add the original tumor point cloud with a different color
    pcd_tumor_original.paint_uniform_color((0, 0, 1))
    geometries.append(pcd_tumor_original)
    colors.append((0, 0, 1))
    
    # Visualize all point clouds
    o3d.visualization.draw_geometries(geometries)


def visualize_graph(VG, EG, pcd_tumor, show_tumor=False, visualize_open3d=False):
    # Create Open3D point cloud from VG
    pcd_vg = o3d.geometry.PointCloud()
    pcd_vg.points = o3d.utility.Vector3dVector(VG)

    if visualize_open3d:
        # Create Open3D lineset from EG
        edges = dg.create_edges(VG, EG)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(VG)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        sphere_meshes = []
        for point in VG:
            sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2)  # Adjust radius as needed
            sphere_mesh.translate(point)
            sphere_mesh.paint_uniform_color((1,0,1))
            sphere_meshes.append(sphere_mesh)

        # Visualize both point cloud and edges
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        vis.add_geometry(pcd_vg)
        vis.add_geometry(line_set)

        for sphere in sphere_meshes:
            vis.add_geometry(sphere)

        if show_tumor:
            vis.add_geometry(pcd_tumor)
        vis.run()
        vis.destroy_window()

    return pcd_vg


def plot_correspondences_alignment_term(pcd_source, graph, target, correspondences):
    """
    Plot correspondences between source and target point clouds.

    Parameters:
    - source: Source point cloud (open3d.geometry.PointCloud)
    - target: Target point cloud (open3d.geometry.PointCloud)
    - correspondences: Numpy array of correspondences (Nx1 array)
                       Each index refers to the index of the source points that connects to the point of the target display in its position.
    """
    
    correspondence = np.array([[i, correspondences[i]] for i in range(len(correspondences))])

    correspondence_copy = copy.deepcopy(correspondence)
    correspondence[:, 1] += len(np.asarray(graph.points))

    # Translate the source and target point clouds
    graph_translated = graph.translate([0, 0, 300])
    pcd_source = pcd_source.translate([0,0,300])
    target_temp = copy.deepcopy(target)
    pcd_source_temp = copy.deepcopy(pcd_source)
    pcd_source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Reverse the order of points for the line set
    reversed_correspondences = correspondence[:, ::-1]

    # Translate the lines to match the reversed order of points
    graph_points_translated = np.asarray(graph_translated.points)
    target_points_translated = np.asarray(target.points)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack((graph_points_translated, target_points_translated))),
        lines=o3d.utility.Vector2iVector(reversed_correspondences)
    )
    
    graph_idx_points = correspondence_copy[:, 0]
    target_idx_points = correspondence_copy[:, 1]

    graph_chosen_points = graph_points_translated[graph_idx_points]
    target_chosen_points = target_points_translated[target_idx_points]

    # Create sphere meshes for correspondence points
    spheres_source = []
    spheres_target = []

    for s_point in graph_chosen_points:
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
    vis.add_geometry(target_temp)
    vis.add_geometry(pcd_source_temp)
    vis.add_geometry(line_set)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
    for sphere_source in spheres_source:
        vis.add_geometry(sphere_source)
    for sphere_target in spheres_target:
        vis.add_geometry(sphere_target)
    
    vis.run()
    vis.destroy_window()



def plot_histogram_distances_new_and_old(pcd_target, s_points, pcd_original, font):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(s_points)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

    evaluation_original = o3d.pipelines.registration.evaluate_registration(pcd_original, pcd_target, 20, np.identity(4))
    correspondence_set_original = np.asarray(evaluation_original.correspondence_set)

    evaluation_new = o3d.pipelines.registration.evaluate_registration(source, pcd_target, 20, np.identity(4))
    correspondence_set_new = np.asarray(evaluation_new.correspondence_set)

    original_points = np.asarray(pcd_original.points)
    source_points = np.asarray(source.points)
    target_points = np.asarray(pcd_target.points)

    distances_new = []
    distances_original = []

    for source_index, target_index in correspondence_set_original:
        source_point = original_points[source_index]
        target_point = target_points[target_index]
        distance = np.linalg.norm(source_point - target_point)
        distances_original.append(distance)

    for source_index, target_index in correspondence_set_new:
        source_point = source_points[source_index]
        target_point = target_points[target_index]
        distance = np.linalg.norm(source_point - target_point)
        distances_new.append(distance)
    
    print(len(distances_new))
    print(len(distances_original))


    plt.figure(figsize=(10, 6))
    hist_new, bins_new = np.histogram(distances_new, bins=200)
    hist_old, bins_old = np.histogram(distances_original, bins=200)

    plt.hist(bins_new[:-1], bins_new, weights=hist_new, alpha=0.7, color='darkseagreen', edgecolor='olivedrab', linewidth=0.5)
    plt.hist(bins_old[:-1], bins_old, weights=hist_old, alpha=0.7, color='thistle', edgecolor='purple', linewidth=0.5)

    mean_new = np.mean(distances_new)
    mean_old = np.mean(distances_original)

    plt.axvline(mean_new, color='olivedrab', linestyle='dashed', linewidth=2)
    plt.axvline(mean_old, color='purple', linestyle='dashed', linewidth=2)

    plt.text(mean_new + 2, plt.ylim()[1]*0.9, f'Mean: {mean_new:.2f}', color='olivedrab', fontsize=font, ha='center')
    plt.text(mean_old + 2, plt.ylim()[1]*0.8, f'Mean: {mean_old:.2f}', color='purple', fontsize=font, ha='center')

    plt.title('Histogram of Distances from Corresponding Points in Breast', fontsize=font)
    plt.xlabel('Distance (mm)', fontsize=font)
    plt.ylabel('# Points', fontsize=font)

    plt.xticks([0, 5, 10, 15, 20], fontsize=font)
    plt.yticks([0, 8000], fontsize=font)

    plt.show()

def plot_red_blues_non_rigid_registration(s_points, t, original):

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(s_points)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(t)

    evaluation_original = o3d.pipelines.registration.evaluate_registration(original, t, 15, np.identity(4))
    correspondence_set_original = np.asarray(evaluation_original.correspondence_set)

    evaluation_new = o3d.pipelines.registration.evaluate_registration(source, t, 15, np.identity(4))
    correspondence_set_new = np.asarray(evaluation_new.correspondence_set)

    target_normals = np.asarray(t.normals)
    original_points = np.asarray(original.points)
    source_points = np.asarray(source.points)
    target_points = np.asarray(t.points)

    distances_original = []
    distances_new = []
    target_temp_points_new = []
    target_temp_points_original = []

    for i, (original_index, target_index) in enumerate(correspondence_set_original):
        original_point = original_points[original_index]
        target_point = target_points[target_index]
        distance = np.linalg.norm(original_point - target_point) 
        
        if np.dot(original_point - target_point, target_normals[i]) >= 0:
            distances_original.append(distance)
        else: 
            distances_original.append(-distance)
        target_temp_points_original.append(target_point)


    distances_original = np.array(distances_original)
    distances_pos_original = (distances_original[distances_original >= 0])
    distances_neg_original = (distances_original[distances_original < 0])
    max_dist_original = max(np.max(distances_pos_original), np.max(distances_neg_original))
    min_dist_original = min(np.min(distances_pos_original), np.min(distances_neg_original))

    for i, (source_index, target_index) in enumerate(correspondence_set_new):
        source_point = source_points[source_index]
        target_point = target_points[target_index]
        distance = np.linalg.norm(source_point - target_point) 
        
        if np.dot(source_point - target_point, target_normals[i]) >= 0:
            distances_new.append(distance)
        else: 
            distances_new.append(-distance)

        target_temp_points_new.append(target_point)


    distances_new = np.array(distances_new)
    distances_pos_new = (distances_new[distances_new >= 0])
    distances_neg_new = (-distances_new[distances_new < 0])



    # if distances_pos_new.size > 0 and distances_neg_new.size > 0:
    #     max_dist = max(np.max(distances_pos_new), np.max(distances_neg_new))
    #     min_dist = min(np.min(distances_pos_new), np.min(distances_neg_new))
    # else:
    #     max_dist = 2*(10**-10)
    #     min_dist = 1*(10**-10)


    red = np.array([[1, 0, 0]])
    blue = np.array([[0, 0, 1]])
    white = np.array([[1, 1, 1]])

    colors_pos_new = ((distances_pos_new - 0) / (max_dist_original - 0)).reshape(-1, 1) @ red + \
                ((max_dist_original - distances_pos_new) / (max_dist_original - 0)).reshape(-1, 1) @ white
    
    colors_neg_new = ((distances_neg_new - min_dist_original) / (0 - min_dist_original)).reshape(-1, 1) @ white + \
                ((0 - distances_neg_new) / (0 - min_dist_original)).reshape(-1, 1) @ blue
    
    # print(distances_new)

    colors_new = np.zeros((distances_new.shape[0], 3))
    colors_new[distances_new >= 0] = colors_pos_new
    colors_new[distances_new < 0] = colors_neg_new

    colors_pos_original = ((distances_pos_original - 0) / (max_dist_original - 0)).reshape(-1, 1) @ red + \
                ((max_dist_original - distances_pos_original) / (max_dist_original - 0)).reshape(-1, 1) @ white
    
    colors_neg_original = ((distances_neg_original - min_dist_original) / (0 - min_dist_original)).reshape(-1, 1) @ white + \
                ((0 - distances_neg_original) / (0 - min_dist_original)).reshape(-1, 1) @ blue
    
    # print(distances_original)
    colors_original = np.zeros((distances_original.shape[0], 3))
    colors_original[distances_original >= 0] = colors_pos_original
    colors_original[distances_original < 0] = colors_neg_original

    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(target_temp_points_original)
    point_cloud_original.colors = o3d.utility.Vector3dVector(colors_original)

    point_cloud_new = o3d.geometry.PointCloud()
    point_cloud_new.points = o3d.utility.Vector3dVector(target_temp_points_new)
    point_cloud_new.colors = o3d.utility.Vector3dVector(colors_new)

    # Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.translate([500, 0, 0])
    target_temp.translate([500,0,0])
    point_cloud_original.translate ([-500,0,0])
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.add_geometry(point_cloud_new)
    vis.add_geometry(point_cloud_original)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0))
    vis.run()
    vis.destroy_window()


    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue, White, Red
    cmap_name = 'custom_color_map'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create some data
    print(min_dist_original)
    print(max_dist_original)
    data = np.random.randint(min_dist_original, max_dist_original, size=(10,10))

    # Plot the data with a colorbar
    plt.imshow(data, cmap=custom_cmap, vmin=min_dist_original, vmax=max_dist_original)
    cbar = plt.colorbar()

    # Define the ticks and labels for the colorbar
    cbar.set_ticks([min_dist_original, 0, max_dist_original])
    cbar.set_ticklabels([f'{np.round(min_dist_original,0):.2f}', f'{0:.2f}', f'{np.round(max_dist_original,0):.2f}'])
    

    plt.show()


def plot_displacements(old_points, new_points, clip_point_cloud=True, clip_value=18):
    displacements = np.linalg.norm(old_points - new_points, axis=1)

    if clip_point_cloud:
        displacements = np.clip(displacements, None, clip_value)
    
    return displacements

def visualize_point_clouds_open3d(old_points, new_points, clip_point_cloud=True, clip_value=18):
    displacements = plot_displacements(old_points, new_points, clip_point_cloud=clip_point_cloud, clip_value=clip_value)
    norm_displacements = (displacements - displacements.min()) / (displacements.max() - displacements.min())
    
    colormap = plt.get_cmap('inferno_r')
    color_source = colormap(norm_displacements)[:, :3]  # Get RGB values from the colormap

    
    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(new_points)
    point_cloud.colors = o3d.utility.Vector3dVector(color_source)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

    max = np.max(displacements)
    data = np.random.randint(0, 15, size=(10, 10))
    plt.imshow(data, cmap=colormap, vmin=0, vmax=max)
    cbar = plt.colorbar()

    # Define the ticks and labels for the colorbar based on the capped displacement values
    cbar.set_ticks([0, np.round(max/2,0), max])
    cbar.set_ticklabels([f'{0:.2f}', f'{np.round(max/2,0):.2f}', f'{max:.2f}'])

    plt.title('Colormap for Displacement Values')
    plt.xlabel('X Label')
    plt.ylabel('Y Label')

    plt.show()

def visualize_histogram_displacements(old_points, new_points, clip_point_cloud=True, clip_value=18, point_cloud_name="Source"):
    displacements = plot_displacements(old_points, new_points, clip_point_cloud=clip_point_cloud, clip_value=clip_value)

    plt.figure(figsize=(10, 6))
    mean_displacement = np.mean(displacements)

    plt.hist(displacements, bins=200, alpha=0.7, color='blue', edgecolor='black')

    plt.axvline(mean_displacement, color='red', linestyle='dashed', linewidth=1)

    plt.text(mean_displacement + 0.5, plt.ylim()[1] * 0.9, f'Mean: {mean_displacement:.2f}', color='red')

    plt.title(f'Histogram of Displacemnets of {point_cloud_name} Points')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Nº Points')
    plt.grid(True)
    plt.show()



# if __name__ == '__main__':
#     folder = "Pacients"
#     number = 74

#     patient = "BR0" + f"{number}"

#     digital_twin = f"{folder}/{patient}/Segment_4.stl"
#     surface_digital_twin = "Pacients/BR074/Final_Surface.ply"
#     _try_ = "Pacients/BR074/try.stl"
#     surface_scan = f"Pacients/{patient}/{number}/{number}.obj"
#     tumor = "Pacients/BR074/Segment_1.stl"

#     mesh = o3d.io.read_triangle_mesh(digital_twin)
#     pcd_1 = mesh.sample_points_uniformly(number_of_points=1000000)
#     pcd_original, rotation, vector = surface_acquisition.rotation_pcd(pcd_1)
#     pcd_final = o3d.io.read_point_cloud(surface_digital_twin)
#     mesh_tumor = o3d.io.read_triangle_mesh(tumor)
#     pcd_tumor = mesh_tumor.sample_points_uniformly(number_of_points=100000)
#     pcd_tumor.rotate(rotation)
#     pcd_tumor.translate(vector)

#     pcd_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
#     pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

#     pcd_original.paint_uniform_color([0.9, 0.4, 0])
#     pcd_final.paint_uniform_color([1, 0.706, 0])
#     pcd_tumor.paint_uniform_color((0,0,1))

#     pcd_tumor_copy = copy.deepcopy(pcd_tumor)
#     pcd_original.translate((-500,0,0))
#     pcd_tumor_copy.translate((-500,0,0))

#     pcd_o = pcd_original + pcd_tumor_copy
#     pcd_f = pcd_final + pcd_tumor

#     # vis = o3d.visualization
#     # mat = vis.rendering.MaterialRecord()
#     # mat.shader = 'defaultLit'
#     # mat.base_color = [0.8, 0, 0, 1.0]

#     # mat_tumor = vis.rendering.MaterialRecord()
#     # # mat_tumor.shader = 'defaultLitTransparency'
#     # mat_tumor.shader = 'defaultLitSSR'
#     # mat_tumor.base_color = [1, 1, 1, 0]
#     # mat_tumor.base_roughness = 0
#     # mat_tumor.base_reflectance = 0
#     # mat_tumor.base_clearcoat = 0
#     # mat_tumor.thickness = 3
#     # mat_tumor.transmission = 2
#     # mat_tumor.absorption_distance = 20
#     # mat_tumor.absorption_color = [1, 0, 0]

#     # geoms = [{'name': 'original', 'geometry': pcd_f, 'material': mat_tumor}, 
#     #          {'name': 'tumor', 'geometry': pcd_tumor, 'material': mat}]
#     # vis.draw(geoms)

#     material = o3d.visualization.rendering.Material()
#     material.base_color = np.array([1.0, 0.0, 0.0, 1.0])  # RGBA color

#     # Apply the material to the mesh
#     pcd_tumor.paint_uniform_color(material.base_color)

#     # Create a visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     # Add the mesh to the visualizer
#     vis.add_geometry(mesh)

#     # Run the visualizer
#     vis.run()
#     vis.destroy_window()
