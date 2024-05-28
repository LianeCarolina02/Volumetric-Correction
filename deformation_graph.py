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
import sys
from scipy.spatial.transform import Rotation
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

def compute_matrix_F(source_points, graph_points, neighbors_dict):
    n = len(source_points)
    r = len(graph_points)
    F = np.zeros((n, 4*r))
    
    for i, source_point in enumerate(source_points):
        neighbor_indices = neighbors_dict.get(i, [])
        for j in neighbor_indices:
            diff = np.array(source_point) - np.array(graph_points[j])
            distance = np.linalg.norm(diff)
            sum_weights = sum([np.exp(-np.linalg.norm(np.array(source_point) - np.array(graph_points[neighbor]))**(1/2)) for neighbor in neighbor_indices])
            if sum_weights != 0:
                normalized_distance = (np.exp(-distance**(1/2))) / sum_weights
                F[i, 4*j:4*(j+1)] = normalized_distance * np.append(diff, 1)
    return F

def compute_matrix_Y(graph_points, EG):
    Y = []
    for edge in EG:
        i, j = edge
        diff_vector = graph_points[j] - graph_points[i]
        Y.append(diff_vector)
    return np.array(Y)

def compute_matrix_P(source_points, graph_points, neighbors_dict):
    n = len(source_points)
    P = np.zeros((n, 3))
    
    for i, source_point in enumerate(source_points):
        neighbor_indices = neighbors_dict.get(i, [])
        for j in neighbor_indices:
            diff = np.array(source_point) - np.array(graph_points[j])
            distance = np.linalg.norm(diff)
            sum_weights = sum([np.exp(-np.linalg.norm(np.array(source_point) - np.array(graph_points[neighbor]))**(1/2)) for neighbor in neighbor_indices])
            if sum_weights != 0:
                normalized_distance = np.exp(-distance**(1/2)) / sum_weights
                P[i] += normalized_distance * np.array(graph_points[j])
        
    return P

def compute_matrix_U(source_points, target_tree):
    _, indices = target_tree.query(source_points)

    U = np.zeros((len(source_points), 3))

    U = target_points[indices]
    
    return U, indices

def visualize_graph(VG, EG):
    # Create Open3D point cloud from VG
    pcd_vg = o3d.geometry.PointCloud()
    pcd_vg.points = o3d.utility.Vector3dVector(VG)

    # Create Open3D lineset from EG
    edges = create_edges(VG, EG)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(VG)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # Visualize both point cloud and edges
    o3d.visualization.draw_geometries([pcd_vg, line_set])

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
        t_i = X[i+3, :3]    # Extract translation vector t_i
        R = closest_rotation_matrix(A_i)  # Find closest rotation matrix to A_i
        Z[i:i+3, :3] = R
        Z[i+3, :3] = np.zeros(3)  # Substitute t_i with zero vector
    return Z

def frobenius_norm_squared(X):
    frobenius_norm_sq = np.linalg.norm(X, 'fro') ** 2
    return frobenius_norm_sq

def compute_error(E_align, E_reg, E_rot, a, b):
    error = frobenius_norm_squared(E_align) + a * frobenius_norm_squared(E_reg) + b * frobenius_norm_squared(E_rot)
    return error

def gradient(X_matrix, F_matrix, P_matrix, U_matrix, B_matrix, Y_matrix, Z_matrix, J_matrix, alpha, beta):
    E_align = F_matrix.T @ (F_matrix @ X_matrix + P_matrix - U_matrix) 
    E_reg = B_matrix.T @ (B_matrix @ X_matrix - Y_matrix) 
    E_rot= J_matrix @ X_matrix - Z_matrix
    return E_align, E_reg, E_rot

def SGD(source_points, graph_points, target_points, neighbors, EG, I_max, alphas:list, betas:list, lb, epsilon):
    all_errors = []
    for parameter in range(len(alphas)):
        F = compute_matrix_F(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors)
        print(f"Matrix F: \nRows: {F.shape[0]} \nCols: {F.shape[1] / 4} * 4 \n")
        P = compute_matrix_P(source_points=source_points, graph_points=graph_points, neighbors_dict=neighbors)
        print(f"Matrix P: \nRows: {P.shape[0]} \nCols: {P.shape[1]} \n")
        B = compute_matrix_B(EG, graph_points=graph_points)
        print(f"Matrix B: \nRows: {B.shape[0]} \nCols: {B.shape[1] / 4} * 4 \n")
        Y = compute_matrix_Y(graph_points=graph_points, EG=EG)
        print(f"Matrix Y: \nRows: {Y.shape[0]} \nCols: {Y.shape[1]} \n")
        J = compute_matrix_J(graph_points=graph_points)
        print(f"Matrix J: \nRows: {J.shape[0]/ 4} * 4 \nCols: {J.shape[1] / 4} * 4\n")

        target_tree = cKDTree(target_points)
        error = np.inf
        errors = []
        X = compute_matrix_X(graph_points=graph_points, int_value=-0.0000001)
        print(f"Matrix X: \nRows: {X.shape[0]/ 4} * 4 \nCols: {X.shape[1]}\n")
        for iteration in range(I_max):
            U = compute_matrix_U(source_points=source_points, target_tree=target_tree)[0]
            print(f"Matrix U: \nRows: {U.shape[0]} \nCols: {U.shape[1]}\n")
            Z = compute_matrix_Z(X)
            print(f"Matrix Z: \nRows: {Z.shape[0]/ 4} * 4 \nCols: {Z.shape[1]}\n")
            
            E_a, E_re, E_ro = gradient(X_matrix=X, F_matrix=F, P_matrix=P, U_matrix=U, B_matrix=B, Y_matrix=Y, Z_matrix=Z, J_matrix=J, alpha=alphas[parameter], beta=betas[parameter])
            # print(f"E_align: {E_a}")
            # print(f"E_reg: {E_re}")
            # print(f"E_ro: {E_ro}")

            G = 2 * (E_a + alphas[parameter] * E_re + betas[parameter] * E_ro)
            X_new = X - lb * G

            new_error = compute_error(E_a, E_re, E_ro, alphas[parameter], betas[parameter])

            source_points = F @ X_new + P
            if np.abs(error - new_error) < epsilon:
                break
            error = new_error
            f_E_a = frobenius_norm_squared(E_a)
            f_E_re = alphas[parameter] * frobenius_norm_squared(E_re)
            f_E_ro = betas[parameter] * frobenius_norm_squared(E_ro)
            errors.append([error, f_E_a, f_E_re, f_E_ro])
            X = X_new
            print(f"Iteration {iteration}, Total Error: {new_error}, Align Error: {f_E_a}, Reg Error: {f_E_re}, Rot Error: {f_E_ro}")

        all_errors.append(errors)

    return source_points, all_errors

def plot_errors(all_errors, alphas, betas):
    plt.figure(figsize=(10, 6))
    for i, errors in enumerate(all_errors):
        errors = np.array(errors)
        total_errors = errors[:, 0]
        align_errors = errors[:, 1]
        reg_errors = errors[:, 2]
        rot_errors = errors[:, 3]

        plt.plot(total_errors, marker='o', linestyle='-', label=f'Total Error alpha={alphas[i]}, beta={betas[i]}')
        plt.plot(align_errors, marker='x', linestyle='--', label=f'Align Error alpha={alphas[i]}, beta={betas[i]}')
        plt.plot(reg_errors, marker='s', linestyle='-.', label=f'Reg Error alpha={alphas[i]}, beta={betas[i]}')
        plt.plot(rot_errors, marker='d', linestyle=':', label=f'Rot Error alpha={alphas[i]}, beta={betas[i]}')

    plt.title('Error Vector Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Error Value')
    plt.grid(True)
    plt.legend()
    plt.show()





id = 74
R = 15
VOXEL_SIZE = 10

pcd_scan, pcd_twin = studies.read_point_clouds(id, scan_2=True)
_, _, transformation,_ = studies.first_transformation(pcd_twin, pcd_scan)
pcd_twin.transform(transformation)

VG_TWIN, EG_TWIN = graph_acquisition(pcd_twin, VOXEL_SIZE, R)
print("Nº Vertices:", len(VG_TWIN))
print("Nº Edges:", len(EG_TWIN))

# visualize_graph(VG=VG_TWIN, EG=EG_TWIN)

pcd_twin = pcd_twin.voxel_down_sample(voxel_size=5)

graph_points = np.array(VG_TWIN)
source_points = np.asarray(pcd_twin.points)
target_points = np.array(pcd_scan.points)

print(f"Nº Graph points: {len(graph_points)}")
print(f"Nº Source points: {len(source_points)}")
print(f"Nº Target points: {len(target_points)}")

pcd_vg = convert_to_point_cloud(VG_TWIN)

neighbors_dictionary = find_neighbors(pcd_twin, pcd_vg, 20)
print("Neighbors acquired")

max_iterations = 1000
alpha = [0.1]
beta = [0.00001]  
epsilon = 0.001
lb = 0.0001

new_source_points, erros = SGD(source_points=source_points, graph_points=graph_points, target_points=target_points, neighbors=neighbors_dictionary, EG=EG_TWIN, I_max=max_iterations, alphas=alpha, betas=beta, lb=lb, epsilon=epsilon)

plot_errors(erros, alpha, beta)

new_pcd_twin = o3d.geometry.PointCloud()
new_pcd_twin.points = o3d.utility.Vector3dVector(new_source_points)

new_pcd_twin.paint_uniform_color((1,0,0))
pcd_twin.paint_uniform_color((0.8,0.8,0.8))
pcd_vg = o3d.geometry.PointCloud()
pcd_vg.points = o3d.utility.Vector3dVector(VG_TWIN)

# Create Open3D lineset from EG
edges = create_edges(VG_TWIN, EG_TWIN)
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(VG_TWIN)
line_set.lines = o3d.utility.Vector2iVector(edges)

new_pcd_twin_copy = copy.deepcopy(new_pcd_twin)
new_pcd_twin_copy.translate((500,0,0))
pcd_scan.translate((500,0,0))
pcd_scan.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([new_pcd_twin, pcd_twin, pcd_vg, line_set, pcd_scan, new_pcd_twin_copy])




# correspondences = find_correspondences(source_points=source_points, target_points=target_points)
# print(len(correspondences))
# plot_correspondences_alignment_term(pcd_twin, pcd_vg, pcd_scan, correspondences=correspondences)