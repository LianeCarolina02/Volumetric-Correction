import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import studies
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import studies
import cv2
import sys
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation
np.set_printoptions(threshold=sys.maxsize)


    
def capture_frame(vis):
    # Capture the current screen
    image = vis.capture_screen_float_buffer(False)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
    out.write(image)

def plot(source_transformed, tumor_points, original_tumor, target, graph, EG, original_source, show_twin=True):
    
    original_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    original_tumor.paint_uniform_color([0,0,1])

    target_copy = copy.deepcopy(target)
    tumor_copy_1 = copy.deepcopy(original_tumor)
    tumor_copy_2 = copy.deepcopy(original_tumor)

    target_copy.translate((500,0,0))
    tumor_copy_1.translate([500,0,0])

    target_copy.paint_uniform_color([1, 0.706, 0])

    new_source_transformed = o3d.geometry.PointCloud()
    new_source_transformed.points = o3d.utility.Vector3dVector(source_transformed)
    new_source_transformed.paint_uniform_color((1,0,0))

    pcd_vg = o3d.geometry.PointCloud()
    pcd_vg.points = o3d.utility.Vector3dVector(graph)

    new_pcd_tumor = o3d.geometry.PointCloud()
    new_pcd_tumor.points = o3d.utility.Vector3dVector(tumor_points)
    new_pcd_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    new_pcd_tumor.paint_uniform_color((1,1,0))

    edges = create_edges(graph, EG)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(graph)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    new_pcd_twin_copy = copy.deepcopy(new_source_transformed)
    new_pcd_twin_copy.translate((500,0,0))

    new_pcd_tumor_copy = copy.deepcopy(new_pcd_tumor)
    new_pcd_tumor_copy.translate((500,0,0))
    

    if show_twin:
        o3d.visualization.draw_geometries([target_copy, new_source_transformed, original_source, pcd_vg, line_set, new_pcd_twin_copy, new_pcd_tumor_copy, new_pcd_tumor, tumor_copy_1,tumor_copy_2])
    else:
        o3d.visualization.draw_geometries([target_copy, new_source_transformed, pcd_vg, line_set, new_pcd_twin_copy, new_pcd_tumor_copy, new_pcd_tumor, tumor_copy_1,tumor_copy_2])

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



def downsample_open_3d(pcd, DOWNSAMPLED):
    pcd_twin_copy = copy.deepcopy(pcd)
    start_time_0 = time.time()
    pcd_twin_downsampled_other = pcd_twin_copy.voxel_down_sample(voxel_size=DOWNSAMPLED)
    print(np.round(time.time() - start_time_0, 4))

    points_1 = pcd_twin_downsampled_other.points

    sphere_meshes = []
    for point in points_1:
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2)  # Adjust radius as needed
        sphere_mesh.translate(point)
        sphere_mesh.paint_uniform_color((1,0,1))
        sphere_meshes.append(sphere_mesh)

    print(len(points_1))
    o3d.visualization.draw_geometries([pcd]+ sphere_meshes)

def compute_euclidean_distances(VG):
    # Compute pairwise Euclidean distances
    distances_euclidean = np.linalg.norm(VG[:, np.newaxis] - VG[np.newaxis, :], axis=-1)
    return distances_euclidean

def graph_acquisition(pcd, VOXEL_SIZE:int, RADIUS:int):
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd_downsampled.paint_uniform_color([1, 0.706, 0])

    pcd_points = np.asarray(pcd_downsampled.points)

    pca = PCA(n_components=1)
    pca.fit(pcd_points)
    largest_eigenvector = pca.components_[0]

    projections = np.dot(pcd_points - pca.mean_, largest_eigenvector)

    sorted_indices = np.argsort(projections)[::-1]
    print(f"Number of Initial Points for Graph acquisition {len(sorted_indices)} with Downsample {VOXEL_SIZE} and Radius {RADIUS}")

    VG = []
    count = 0
    for i, index in enumerate(sorted_indices):
        vi = pcd_points[index]
        if not VG or all(np.linalg.norm(vi - vj) >= RADIUS for vj in VG):
            VG.append(vi)
            count += 1
    
    # Compute pairwise geodesic distances using VG
    distances_euclidean = compute_euclidean_distances(np.array(VG))

    # Construct edge set EG by connecting vertices whose geodesic distance is smaller than R
    EG = set()
    for i in range(len(VG)):
        for j in range(i+1, len(VG)):
            if distances_euclidean[i][j] <= RADIUS + 15:
                EG.add((i, j))
                EG.add((j, i))

    return VG, EG

def convert_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def visualization_graph_and_pcd(pcd, pcd_graph):
    sphere_meshes = []
    for point in pcd_graph:
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2)  # Adjust radius as needed
        sphere_mesh.translate(point)
        sphere_mesh.paint_uniform_color((1,0,1))
        sphere_meshes.append(sphere_mesh)

    o3d.visualization.draw_geometries([pcd] + sphere_meshes)


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

def compute_weights(source_points, graph_points, sigma=1.0):
    weights = np.zeros((len(source_points), len(graph_points)))
    for i, sp in enumerate(source_points):
        distances = np.linalg.norm(graph_points - sp, axis=1)
        weights[i] = np.exp(-distances**2 / (2 * sigma**2))
        weights[i] /= np.sum(weights[i])  # Normalize weights to sum to 1
    return weights

def find_neighbors(pcd_source, pcd_graph, R):
    # Create KDTree from graph points
    graph_kdtree = o3d.geometry.KDTreeFlann(pcd_graph)
    source_points = np.asarray(pcd_source.points)
    
    # Dictionary to store neighbors
    neighbors_dict = {}
    
    for i, source_point in enumerate(source_points):
        _, idx, _ = graph_kdtree.search_radius_vector_3d(source_point, R)
        neighbors_dict[i] = idx
    # print(neighbors_dict)
    return neighbors_dict

def compute_matrix_F(source_points, graph_points, neighbors_dict, a, b):
    n = len(source_points)
    r = len(graph_points)
    F = np.zeros((n, 4*r))
    
    for i, source_point in enumerate(source_points):
        neighbor_indices = neighbors_dict.get(i, [])
        sum_weights = sum([np.exp(-((np.linalg.norm(np.array(source_point) - np.array(graph_points[neighbor])))**(a)/b)) for neighbor in neighbor_indices])

        for j in neighbor_indices:
            diff = np.array(source_point) - np.array(graph_points[j])
            distance = np.linalg.norm(diff)
            if sum_weights != 0:
                normalized_distance = (np.exp(-(distance**(a)/b))) / sum_weights
                F[i, 4*j:4*(j+1)] = normalized_distance * np.append(diff, 1)
    return F

def compute_matrix_Y(graph_points, EG):
    Y = []
    for edge in EG:
        i, j = edge
        diff_vector = graph_points[j] - graph_points[i]
        Y.append(diff_vector)
    return np.array(Y)

def compute_matrix_P(source_points, graph_points, neighbors_dict, a, b):
    n = len(source_points)
    P = np.zeros((n, 3))
    
    for i, source_point in enumerate(source_points):
        neighbor_indices = neighbors_dict.get(i, [])
        sum_weights = sum([np.exp(-((np.linalg.norm(np.array(source_point) - np.array(graph_points[neighbor])))**(a)/b)) for neighbor in neighbor_indices])
        for j in neighbor_indices:
            diff = np.array(source_point) - np.array(graph_points[j])
            distance = np.linalg.norm(diff)
            if sum_weights != 0:
                normalized_distance = (np.exp(-(distance**(a)/b))) / sum_weights
                P[i] += normalized_distance * np.array(graph_points[j])
        
    return P

def compute_matrix_U(source_points, target_tree):
    _, indices = target_tree.query(source_points)

    U = np.zeros((len(source_points), 3))

    U = target_points[indices]
    
    return U, indices

def visualize_graph(VG, EG, pcd, show_tumor=False):
    # Create Open3D point cloud from VG
    pcd_vg = o3d.geometry.PointCloud()
    pcd_vg.points = o3d.utility.Vector3dVector(VG)

    # Create Open3D lineset from EG
    edges = create_edges(VG, EG)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(VG)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # Visualize both point cloud and edges
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=1200)
    vis.add_geometry(pcd_vg)
    vis.add_geometry(line_set)
    if show_tumor:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def create_edges(VG, EG):
    lines = []
    for edge in EG:
        lines.append([edge[0], edge[1]])
    return np.array(lines)

def compute_matrix_B(EG, graph_points):
    num_edges = len(EG)
    num_graph_points = len(graph_points)
    B = np.zeros((num_edges, 4 * num_graph_points))
    for idx, edge in enumerate(EG):
        i, j = edge
        # Construct row for index i
        diff_vector = graph_points[j] - graph_points[i]
        row_i = np.concatenate((diff_vector, [1]))
        B[idx, i*4:(i+1)*4] = row_i
        # Construct row for index j
        B[idx, j*4:(j+1)*4] = [0, 0, 0, 1]
    return B

def compute_matrix_X(graph_points, int_value):
    r = len(graph_points)
    X = int_value * np.ones((4*r, 3))
    return X


def compute_matrix_J(graph_points):
    r = len(graph_points)
    J = np.eye(4 * r)  # Create an identity matrix with dimensions 4r x 4r
    for i in range(3, 4 * r, 4):
        J[i, i] = 0  # Set diagonal elements with row indices such that % 4 == 3 to 0
    return J


def closest_rotation_matrix(A):
    r = Rotation.from_matrix(A)
    R = r.as_matrix()
    return R

def compute_matrix_Z(X):
    Z = np.zeros_like(X)
    for i in range(0, X.shape[0], 4):
        A_i = X[i:i+3, :3]  # Extract rotation matrix A_i
        R = closest_rotation_matrix(A_i)  # Find closest rotation matrix to A_i
        Z[i:i+3, :3] = R
        Z[i+3, :3] = np.zeros(3)  # Substitute t_i with zero vector
    return Z

def frobenius_norm_squared(X):
    frobenius_norm_sq = np.linalg.norm(X, 'fro') ** 2
    return frobenius_norm_sq

def compute_error(E_align, E_reg, a):
    error = frobenius_norm_squared(E_align) + a * frobenius_norm_squared(E_reg) #+ b * frobenius_norm_squared(E_rot)
    return error

def gradient(X_matrix, F_matrix, P_matrix, U_matrix, B_matrix, Y_matrix):
    E_align = F_matrix.T @ (F_matrix @ X_matrix + P_matrix - U_matrix) 
    E_reg = B_matrix.T @ (B_matrix @ X_matrix - Y_matrix) 
    # E_rot= J_matrix @ X_matrix - Z_matrix
    return E_align, E_reg#, E_rot

def SGD(source_points, graph_points, target_points, tumor_points, neighbors_source, neighbors_tumor, EG, I_max, alpha:int, bs:list, ass:list, lb, epsilon, capture=False):
    time_1 = time.time()
    all_tumor_points = []
    n = len(source_points)
    
    if capture:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source_points)
        pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

        pcd_tumor = o3d.geometry.PointCloud()
        pcd_tumor.points = o3d.utility.Vector3dVector(tumor_points)
        pcd_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        
        vis.add_geometry(pcd_source)
        vis.add_geometry(pcd_tumor)
        view_control = vis.get_view_control()
    all_F_tumor = []
    all_P_tumor = []
    for idx in range(len(bs)):
        F_tumor = compute_matrix_F(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        P_tumor = compute_matrix_P(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        all_F_tumor.append(F_tumor)
        all_P_tumor.append(P_tumor)

    F = compute_matrix_F(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors_source, a=1/2, b=1)
    P = compute_matrix_P(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors_source, a=1/2, b=1)

    B = compute_matrix_B(EG, graph_points=graph_points)
    Y = compute_matrix_Y(graph_points=graph_points, EG=EG)
    # J = compute_matrix_J(graph_points=graph_points)

    target_tree = cKDTree(target_points)
    error = np.inf
    errors = []
    X = compute_matrix_X(graph_points=graph_points, int_value=-0.0000001)
    time_2 = time.time()
    print(f"Initialization Time: {time_2 - time_1}")

    for iteration in range(I_max):
        U = compute_matrix_U(source_points=source_points, target_tree=target_tree)[0]
        # Z = compute_matrix_Z(X)
        
        E_a, E_re = gradient(X_matrix=X, F_matrix=F, P_matrix=P, U_matrix=U, B_matrix=B, Y_matrix=Y)

        G = 2 * (E_a + alpha * E_re) # + betas[parameter] * E_ro)
        X_new = X - lb * G

        new_error = compute_error(E_a, E_re, alpha) / n

        source_points = F @ X_new + P

        if capture:
            tumor_points = all_F_tumor[0] @ X_new + all_P_tumor[0]
            pcd_source.points = o3d.utility.Vector3dVector(source_points)
            pcd_tumor.points = o3d.utility.Vector3dVector(tumor_points)
            vis.update_geometry(pcd_source)
            vis.update_geometry(pcd_tumor)
            vis.poll_events()
            vis.update_renderer()
            view_control.rotate(-10.0, 0.0)
            
            capture_frame(vis)

        if np.abs(error - new_error) < epsilon:
            break
        error = new_error
        f_E_a = frobenius_norm_squared(E_a) / n
        f_E_re = alpha * frobenius_norm_squared(E_re) / n
        # f_E_ro = betas[parameter] * frobenius_norm_squared(E_ro) / n
        errors.append([error, f_E_a, f_E_re])#, f_E_ro])
        X = X_new
        print(f"Iteration {iteration}, Total Error: {new_error}, Align Error: {f_E_a}, Reg Error: {f_E_re}")#, Rot Error: {f_E_ro}")

    if capture:
        vis.destroy_window()
        out.release()
        cv2.destroyAllWindows()

    for idx in range(len(bs)):
        tumor_points = all_F_tumor[idx] @ X + all_P_tumor[idx]
        all_tumor_points.append(tumor_points)

    print(f"SVD Time: {time.time() - time_2}")
    return source_points, all_tumor_points, errors

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



def plot_distances_non_rigid_transformation(s_points, t, original):

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
    cbar.set_ticklabels([str(min_dist_original), str(0), str(max_dist_original)])

    plt.show()

def plot_histogram_distances_new_and_old(pcd_target, s_points, pcd_original):
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

    sum_new = np.sum(hist_new)
    sum_old = np.sum(hist_old)

    plt.hist(bins_new[:-1], bins_new, weights=hist_new/sum_new, alpha=0.7, label=f'New Distances')
    plt.hist(bins_old[:-1], bins_old, weights=hist_old/sum_old, alpha=0.7, label=f'1st Transformation')

    plt.title('Histogram of Distances from Corresponding Points in Breast')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Nº Points')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    downsample = 2
    width, height = 1920, 1080  # Set the desired resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'tumor_vox_2_step_1.avi', fourcc, 30.0, (width, height))
    tumor = "Pacients/BR074/Segment_1.stl"

    start_time = time.time()
    id = 74
    R = 15
    VOXEL_SIZE = 10

    pcd_scan, pcd_twin, pcd_tumor = studies.read_point_clouds(id, scan_2=False)
    _, _, transformation,_ = studies.first_transformation(pcd_twin, pcd_scan)
    pcd_twin.transform(transformation)
    pcd_tumor.transform(transformation)

    VG_TWIN, EG_TWIN = graph_acquisition(pcd_twin, VOXEL_SIZE, R)
    print("Nº Vertices:", len(VG_TWIN))
    print("Nº Edges:", len(EG_TWIN))

    # visualize_graph(VG=VG_TWIN, EG=EG_TWIN, pcd=pcd_tumor)

    pcd_twin = pcd_twin.voxel_down_sample(voxel_size=downsample)

    graph_points = np.array(VG_TWIN)
    source_points = np.asarray(pcd_twin.points)
    target_points = np.array(pcd_scan.points)
    tumor_points = np.array(pcd_tumor.points)

    print(f"Nº Source points: {len(source_points)}")
    print(f"Nº Target points: {len(target_points)}")
    print(f"Nº Tumor points: {len(tumor_points)}")

    pcd_vg = convert_to_point_cloud(VG_TWIN)

    neighbors_dictionary = find_neighbors(pcd_twin, pcd_vg, 20)
    neighbors_tumor = find_neighbors(pcd_tumor, pcd_vg, 100)
    print("Neighbors acquired")

    max_iterations = 1000
    alpha = 0.1
    bs = [1]  
    ass = [1/2]
    epsilon = 0.5
    lb = 0.0001

    new_source_points, all_new_tumor_points, erros = SGD(source_points=source_points, graph_points=graph_points, target_points=target_points,tumor_points=tumor_points, neighbors_source=neighbors_dictionary, neighbors_tumor= neighbors_tumor, EG=EG_TWIN, I_max=max_iterations, alpha=alpha, bs=bs, ass=ass, lb=lb, epsilon=epsilon, capture= True)

    print(f"Total Time: {time.time() - start_time}")

    plot_distances_non_rigid_transformation(s_points=new_source_points, t=pcd_scan, original=pcd_twin)
    
    # plot(new_source_points, all_new_tumor_points[0], pcd_tumor, pcd_scan, graph_points, EG_TWIN, pcd_twin, show_twin=True)

    # plot_histogram_distances_new_and_old(pcd_scan, new_source_points, pcd_twin)

    # plot_tumor(pcd_tumor, all_new_tumor_points, bs)

    # plot_errors(erros, alpha)
