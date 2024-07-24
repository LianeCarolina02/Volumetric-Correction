import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import studies
import copy
import time
import os
import pandas as pd
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

def write_to_file(message):
    with open('log.txt', 'a') as file:
        file.write(message + '\n')

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

    initial_time = time.time()
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
            if distances_euclidean[i][j] <= RADIUS * 2:
                EG.add((i, j))
                EG.add((j, i))

    edge_count = np.zeros(len(VG), dtype=int)
    for edge in EG:
        edge_count[edge[0]] += 1

    # Compute the average number of edges per node
    average_edges_per_node = np.mean(edge_count)

    print(f"Average number of edges per node: {average_edges_per_node}")
    final_time = time.time() - initial_time
    return VG, EG, final_time

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
    return neighbors_dict, average_neighbors_per_point

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



def SGD_without_capture(source_points, graph_points, target_points, tumor_points, 
                        neighbors_source, neighbors_tumor, EG, 
                        I_max, alpha:int, bs:list, ass:list, lb, epsilon,
                        graph_time, more_time, downsample, radius, resolution, n_vertices, n_edges, n_source, 
                        n_neighbors_source, n_neighbors_tumor, name_cols):
    
    initial_time_matrices_tumor = time.time()
    all_tumor_points = []
    all_F_tumor = []
    all_P_tumor = []
    computation_matrixes_time_tumor = []

    for i in bs:
        F_tumor = compute_matrix_F(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=1/2, b=i)
        P_tumor = compute_matrix_P(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=1/2, b=i)
        all_F_tumor.append(F_tumor)
        all_P_tumor.append(P_tumor)
        computation_matrixes_time_tumor.append(time.time() - initial_time_matrices_tumor)
        print(f'Coeficient_b_{i}')
    
    print(f'all_F_tumor_after_b: {len(all_F_tumor)}')
    
    for i in ass:
        F_tumor = compute_matrix_F(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=i, b=1)
        P_tumor = compute_matrix_P(source_points=tumor_points, graph_points=graph_points, neighbors_dict=neighbors_tumor, a=i, b=1)
        all_F_tumor.append(F_tumor)
        all_P_tumor.append(P_tumor)
        computation_matrixes_time_tumor.append(time.time() - initial_time_matrices_tumor)
        print(f'Coeficient_a_{i}')

    initial_time_matrices_source = time.time()
    print(f'all_F_tumor_after_b_a: {len(all_F_tumor)}')

    F = compute_matrix_F(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors_source, a=1/2, b=1)
    P = compute_matrix_P(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors_source, a=1/2, b=1)
    computation_matrixes_time_source = time.time() - initial_time_matrices_source

    B = compute_matrix_B(EG, graph_points=graph_points)
    Y = compute_matrix_Y(graph_points=graph_points, EG=EG)

    target_tree = cKDTree(target_points)
    error = np.inf
    errors = []
    X = compute_matrix_X(graph_points=graph_points, int_value=-0.0000001)

    for iteration in range(I_max):
        U = compute_matrix_U(source_points=source_points, target_tree=target_tree, target_points=target_points)[0]
        E_a, E_re = gradient(X_matrix=X, F_matrix=F, P_matrix=P, U_matrix=U, B_matrix=B, Y_matrix=Y)

        G = 2 * (E_a + alpha * E_re)
        X_new = X - lb * G

        new_error = compute_error(E_a, E_re, alpha) / n_source

        source_points = F @ X_new + P

        if np.abs(error - new_error) < epsilon:
            break

        if new_error > error + 10000:
            raise ValueError(f'FAILED')

        error = new_error
        f_E_a = frobenius_norm_squared(E_a) / n_source
        f_E_re = alpha * frobenius_norm_squared(E_re) / n_source
        errors.append([error, f_E_a, f_E_re])
        X = X_new
        print(f"Iteration {iteration}, Total Error: {new_error}")

        algorithm_time = time.time() - initial_time_matrices_source

    for idx in range(len(bs)):
        print(f'Tumor_points_index_of_F{idx}')
        print(f'Value of b_{bs[idx]} and a_1/2')
        tumor_points = all_F_tumor[idx] @ X + all_P_tumor[idx]
        print("done tumor_points")
    
        total_time = (time.time() - initial_time_matrices_tumor) + graph_time + more_time
        
        values_cols = [graph_time, downsample, radius, n_vertices, n_edges, resolution, computation_matrixes_time_source, computation_matrixes_time_tumor[idx], 
                       algorithm_time, total_time, iteration, bs[idx], 1/2, lb, n_source, n_neighbors_source, n_neighbors_tumor]
        
        save_row_to_csv('Studies.csv', name_cols, values_cols)
        print("done saving csv")

        save_pickle(source_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}', 'source_pcd')
        print("done saving pickles")
        save_pickle(graph_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}' , 'graph_pcd')
        print("done saving pickles")
        save_pickle(errors, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}' , 'erros')
        print("done saving pickles")
        save_pickle(tumor_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}' ,f'tumor_pcd_B{bs[idx]}_A0_5')
        print("done saving pickles")

    
    for idx in range(len(ass)):
        print(f'Tumor_points_index_of_F{idx + len(bs)}')
        print(f'Value of b_1 and a_{ass[idx]}')
        tumor_points = all_F_tumor[idx + len(bs)] @ X + all_P_tumor[idx + len(bs)]
    
        total_time = (time.time() - initial_time_matrices_tumor) + graph_time + more_time

        
        values_cols = [graph_time, downsample, radius, n_vertices, n_edges, resolution, computation_matrixes_time_source, computation_matrixes_time_tumor[idx + len(bs)], 
                       algorithm_time, total_time, iteration, 1, ass[idx], lb, n_source, n_neighbors_source, n_neighbors_tumor]
        
        save_row_to_csv('Studies.csv', name_cols, values_cols)

        save_pickle(source_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}', 'source_pcd')
        save_pickle(graph_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}', 'graph_pcd')
        save_pickle(errors, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}' , 'erros')
        save_pickle(tumor_points, f'Pickles/id_{ID}/RADIUS_{radius}/DOWNSAMPLE_{downsample}/RESOLUTION{resolution}/Step_Size{lb}' , f'tumor_pcd_B1_A{ass[idx]}')
    
    write_to_file(f'FINISHED- ID:{ID} RADIUS:{RADIUS} DOWNSAMPLE:{DOWNSAMPLE} RESOLUTION:{VOXEL_SIZE} Step_Size:{lb}')


    return source_points, all_tumor_points, errors, iteration


def save_pickle(pickle_object, folder: str, file_name: str):
    if not os.path.exists(folder):
        os.makedirs(folder)

    full_path = os.path.join(folder, f'{file_name}.pkl')
    print(full_path)

    with open(full_path, 'wb') as f:
        pickle.dump(pickle_object, f)

def save_row_to_csv(file_path, col_names, values):
    """
    Save a list of values to a CSV file at the specified row and column names.
    
    Parameters:
    - file_path: str, path to the CSV file.
    - row_name: str, the name of the row.
    - col_names: list of str, the names of the columns.
    - values: list, the values to save in the columns.
    """
    if len(col_names) != len(values):
        raise ValueError("The length of column names and values must be the same.")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=col_names)
    
    for col_name in col_names:
        if col_name not in df.columns:
            df[col_name] = pd.Series(dtype='object')
    
    # Create a new row with the specified values
    new_row = dict(zip(col_names, values))
    
    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the DataFrame back to the CSV file
    df.to_csv(file_path, index=False)


if __name__ == "__main__":

    # out = visualization.video_set_up("original", fps=30, width=1920, height=1080)
    IDS = [61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 76]
    name_cols = ['graph_time', 'Downsample', 'Radius', 'NGraphPoints', 'NGraphEdges', 'Resolution',
                 'initial_matrixes_time_source', 'initial_matrixes_time_tumor', 'algorithm_time', 'TotalTime', 'iterations', 
                 'B_Weights', 'A_Weights', 'Step_Size', 'NSourcePoints', 'NNeighborsSource', 'NNeighborsTumor'
                 ]

    for ID in IDS:

        ALL_RADIUS = [15, 20, 30]
        ALL_DOWNSAMPLE = [10, 15]
        VOXEL_SIZES = [0.1, 0.5, 1, 2]
        TUMOR_PATH = f"Pacients/BR0{ID}/Segment_1.stl"

        for RADIUS in ALL_RADIUS:
            for DOWNSAMPLE in ALL_DOWNSAMPLE:

                _time_ = np.round(time.time(), 4)

                pcd_scan, pcd_twin, pcd_tumor = studies.read_point_clouds(ID, scan_2=False)
                _, _, transformation,_ = studies.first_transformation(pcd_twin, pcd_scan)
                pcd_twin.transform(transformation)
                pcd_tumor.transform(transformation)

                VG_TWIN, EG_TWIN, graph_time = graph_acquisition(pcd_twin, DOWNSAMPLE, RADIUS)

                additional_time = time.time()

                pcd_vg = visualization.visualize_graph(VG=VG_TWIN, EG=EG_TWIN, pcd_tumor=pcd_tumor)

                for VOXEL_SIZE in VOXEL_SIZES:
                    pcd_twin = pcd_twin.voxel_down_sample(voxel_size=VOXEL_SIZE)

                    source_points = np.asarray(pcd_twin.points)
                    graph_points = np.array(VG_TWIN)
                    target_points = np.array(pcd_scan.points)
                    tumor_points = np.array(pcd_tumor.points)

                    n_source = len(source_points)
                    n_vertices = len(VG_TWIN)
                    n_edges = len(EG_TWIN)

                    neighbors_dictionary, n_neighbors_source = find_neighbors(pcd_twin, pcd_vg, 20)
                    neighbors_tumor, n_neighbors_tumor = find_neighbors(pcd_tumor, pcd_vg, 150)

                    max_iterations = 2000
                    coeficient_weights_b = [1/100, 1/20, 1/2, 1, 2, 8, 20]
                    coeficient_weights_a = [1/40, 1/20, 1/2, 1, 2, 8, 20]
                    alpha = 0.1
                    epsilon = 0.1
                    LAMBDAS = [10**-7, 10**-6, 10**-5, 10**-4]

                    for lb in LAMBDAS:
                        write_to_file(f'STARTED- ID:{ID} RADIUS:{RADIUS} DOWNSAMPLE:{DOWNSAMPLE} RESOLUTION:{VOXEL_SIZE} Step_Size:{lb}')
                        print(f"ID:{ID} RADIUS:{RADIUS} DOWNSAMPLE:{DOWNSAMPLE} RESOLUTION:{VOXEL_SIZE} Step_Size:{lb}")

                        additional_time_1 = time.time() - additional_time
                        try:
                            NEW_SOURCE_POINTS, ALL_TUMOR_POINTS, ERROS, IT = SGD_without_capture(source_points, graph_points, target_points, tumor_points,
                                                                                            neighbors_dictionary, neighbors_tumor, EG_TWIN,
                                                                                            max_iterations, alpha, coeficient_weights_b, coeficient_weights_a,
                                                                                            lb, epsilon,
                                                                                            graph_time, additional_time_1, DOWNSAMPLE, RADIUS, VOXEL_SIZE, n_vertices, n_edges, n_source, 
                                                                                            n_neighbors_source, n_neighbors_tumor, name_cols)
                        
                        except Exception as e:
                            print(ValueError)
                            write_to_file(f'FAILED- ID:{ID} RADIUS:{RADIUS} DOWNSAMPLE:{DOWNSAMPLE} RESOLUTION:{VOXEL_SIZE} Step_Size:{lb}')


                    