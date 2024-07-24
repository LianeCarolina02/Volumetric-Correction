import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import studies
import copy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import studies
import cv2
import sys
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation
import visualization
np.set_printoptions(threshold=sys.maxsize)



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

    edge_count = np.zeros(len(VG), dtype=int)
    for edge in EG:
        edge_count[edge[0]] += 1

    # Compute the average number of edges per node
    average_edges_per_node = np.mean(edge_count)

    print(f"Average number of edges per node: {average_edges_per_node}")
    return VG, EG

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

    # Compute the number of neighbors for each point
    num_neighbors = [len(neighbors) for neighbors in neighbors_dict.values()]

    # Compute the average number of neighbors per point
    average_neighbors_per_point = np.mean(num_neighbors)

    print(f"Average number of neighbors per point: {average_neighbors_per_point}")
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

def compute_matrix_U(source_points, target_tree, target_points):
    _, indices = target_tree.query(source_points)

    U = np.zeros((len(source_points), 3))

    U = target_points[indices]
    
    return U, indices


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

def SGD_with_capture(source_points, graph_points, target_points, tumor_points, neighbors_source, neighbors_tumor, EG, I_max, alpha:int, bs:list, ass:list, lb, epsilon, out, capture=False):
    time_1 = time.time()
    all_tumor_points = []
    n = len(source_points)
    
    if capture:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.get_render_option().background_color = [0, 0, 0]

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source_points)
        pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        pcd_source.paint_uniform_color([1, 0.706, 0])

        edges = create_edges(graph_points, EG)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(graph_points)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        pcd_tumor = o3d.geometry.PointCloud()
        pcd_tumor.points = o3d.utility.Vector3dVector(tumor_points)
        pcd_tumor.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        pcd_tumor.paint_uniform_color((0,0,1))
        pcd_tumor_copy = copy.deepcopy(pcd_tumor)

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_points)
        pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
        pcd_target.paint_uniform_color([0, 0.651, 0.929])

        vis.add_geometry(pcd_source)
        
        # vis.add_geometry(pcd_tumor)
        vis.add_geometry(line_set)
        pcd_tumor_copy.paint_uniform_color((1,0,0))
        vis.add_geometry(pcd_tumor_copy)
        # vis.add_geometry(pcd_target)

        view_control = vis.get_view_control()
        view_control.rotate(1000, -100)
        view_control.set_zoom(0.5)

        visualization.capture_frame(vis, out)
    all_F_tumor = []
    all_P_tumor = []
    for idx in range(len(bs)):
        F_tumor = compute_matrix_F(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        P_tumor = compute_matrix_P(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        all_F_tumor.append(F_tumor)
        all_P_tumor.append(P_tumor)
        print("Done F_tumor")

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
        U = compute_matrix_U(source_points=source_points, target_tree=target_tree, target_points=target_points)[0]
        # Z = compute_matrix_Z(X)
        
        E_a, E_re = gradient(X_matrix=X, F_matrix=F, P_matrix=P, U_matrix=U, B_matrix=B, Y_matrix=Y)

        G = 2 * (E_a + alpha * E_re) # + betas[parameter] * E_ro)
        X_new = X - lb * G

        new_error = compute_error(E_a, E_re, alpha) / n

        source_points = F @ X_new + P

        if capture:
            tumor_points = all_F_tumor[0] @ X_new + all_P_tumor[0]
            pcd_source.points = o3d.utility.Vector3dVector(source_points)
            pcd_tumor_copy.points = o3d.utility.Vector3dVector(tumor_points)
            vis.update_geometry(pcd_source)
            vis.update_geometry(pcd_tumor_copy)
            # vis.update_geometry(pcd_target)
            vis.poll_events()
            vis.update_renderer()
            
            visualization.capture_frame(vis, out)

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



def SGD_without_capture(source_points, graph_points, target_points, tumor_points, neighbors_source, neighbors_tumor, EG, I_max, alpha:int, bs:list, ass:list, lb, epsilon):
    time_1 = time.time()
    all_tumor_points = []
    n = len(source_points)

    all_F_tumor = []
    all_P_tumor = []
    for idx in range(len(bs)):
        F_tumor = compute_matrix_F(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        P_tumor = compute_matrix_P(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=ass[idx], b=bs[idx])
        all_F_tumor.append(F_tumor)
        all_P_tumor.append(P_tumor)
        print("Done F_tumor")

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
        U = compute_matrix_U(source_points=source_points, target_tree=target_tree, target_points=target_points)[0]
        # Z = compute_matrix_Z(X)
        
        E_a, E_re = gradient(X_matrix=X, F_matrix=F, P_matrix=P, U_matrix=U, B_matrix=B, Y_matrix=Y)

        G = 2 * (E_a + alpha * E_re) # + betas[parameter] * E_ro)
        X_new = X - lb * G

        new_error = compute_error(E_a, E_re, alpha) / n

        source_points = F @ X_new + P

        if np.abs(error - new_error) < epsilon:
            break
        error = new_error
        f_E_a = frobenius_norm_squared(E_a) / n
        f_E_re = alpha * frobenius_norm_squared(E_re) / n
        # f_E_ro = betas[parameter] * frobenius_norm_squared(E_ro) / n
        errors.append([error, f_E_a, f_E_re])#, f_E_ro])
        X = X_new
        print(f"Iteration {iteration}, Total Error: {new_error}, Align Error: {f_E_a}, Reg Error: {f_E_re}")#, Rot Error: {f_E_ro}")

    for idx in range(len(bs)):
        tumor_points = all_F_tumor[idx] @ X + all_P_tumor[idx]
        all_tumor_points.append(tumor_points)

    print(f"SVD Time: {time.time() - time_2}")
    return source_points, all_tumor_points, errors, iteration


def save_point_cloud_in_pickle(points, name:str):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)


if __name__ == "__main__":

    # out = visualization.video_set_up("original", fps=30, width=1920, height=1080)

    DOWNSAMPLE = 2
    TUMOR_PATH = "Pacients/BR074/Segment_1.stl"
    ID = 74
    RADIUS = 15
    VOXEL_SIZE = 10

    pcd_scan, pcd_twin, pcd_tumor = studies.read_point_clouds(ID, scan_2=False)
    _, _, transformation,_ = studies.first_transformation(pcd_twin, pcd_scan)
    pcd_twin.paint_uniform_color([1, 0.706, 0])
    pcd_scan.paint_uniform_color([0, 0.651, 0.929])
    pcd_twin.transform(transformation)
    pcd_tumor.transform(transformation)

    VG_TWIN, EG_TWIN = graph_acquisition(pcd_twin, VOXEL_SIZE, RADIUS)
    print("# Graph Points:", len(VG_TWIN))
    print("# Edges:", len(EG_TWIN))
    pcd_vg = visualization.visualize_graph(VG=VG_TWIN, EG=EG_TWIN, pcd_tumor=pcd_tumor, show_tumor=True)

    graph_points = np.array(VG_TWIN)
    source_points = np.asarray(pcd_twin.points)
    target_points = np.array(pcd_scan.points)
    tumor_points = np.array(pcd_tumor.points)

    print(f"# Source points: {len(source_points)}")
    print(f"# Target points: {len(target_points)}")
    print(f"# Tumor points: {len(tumor_points)}")

    pcd_twin = pcd_twin.voxel_down_sample(voxel_size=2)
    source_points = np.asarray(pcd_twin.points)
    print(f"# Source points: {len(source_points)}")

    neighbors_dictionary = find_neighbors(pcd_twin, pcd_vg, 20)
    neighbors_tumor = find_neighbors(pcd_tumor, pcd_vg, 150)
    print("Neighbors acquired")

    max_iterations = 5000
    alpha = 0.1
    coeficient_weights_b = [1]  
    coeficient_weights_a = [1/2]
    epsilon = 0.001
    lb = 0.0001

    new_source_points, all_new_tumor_points, erros, it = SGD_without_capture(source_points=source_points, graph_points=graph_points, target_points=target_points, tumor_points=tumor_points, 
                                                                       neighbors_source=neighbors_dictionary, neighbors_tumor= neighbors_tumor, 
                                                                       EG=EG_TWIN, 
                                                                       I_max=max_iterations, alpha=alpha, bs=coeficient_weights_b, ass=coeficient_weights_a, lb=lb, epsilon=epsilon)

    save_point_cloud_in_pickle(new_source_points, f"Pickles/source_points_it{it}_alpha{alpha}_lb{lb}_epsilon{epsilon}")

    for i, (a, b) in enumerate(zip(coeficient_weights_a, coeficient_weights_b)):
        save_point_cloud_in_pickle(all_new_tumor_points[i], f"Pickles/tumor_points_a{a}_b{b}")