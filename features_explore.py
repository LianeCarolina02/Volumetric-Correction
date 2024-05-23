import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def save_features(pcd_path, pcd_type: str, folder: str, capture_screen=True):
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_fpfh = fpfh_matrix(pcd, voxel_size=voxel_downsample)
    maximum, minimum = bounded_values_fpfh(pcd_fpfh)
    
    for i in range(len(pcd_fpfh)):
        print(f"{i}-th Feature")
        fpfh = np.asarray(pcd_fpfh[i,:])
        colormap = 'inferno'

        fpfh_colors = plt.get_cmap(colormap )(
                (fpfh - minimum) / (maximum - minimum))
        fpfh_colors = fpfh_colors[:, :3]


        fpfh_pcd =o3d.geometry.PointCloud()
        fpfh_pcd.points = pcd.points
        fpfh_pcd.normals = pcd.normals
        fpfh_pcd.colors = o3d.utility.Vector3dVector(fpfh_colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1600, height=1200)
        vis.add_geometry(fpfh_pcd)
        vis.run()
        if capture_screen:
            vis.capture_screen_image(f"{folder}/Feature_{i}_{pcd_type}.png")
        vis.destroy_window()

def see_features(folder: str, pcd_type: str, num_rows: int, num_cols: int, features_list: list):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*3))
    axes = axes.flatten()

    selected_filenames = [f"Feature_{i}_{pcd_type}.png" for i in features_list]
    image_files = [os.path.join(folder, file) for file in selected_filenames if file in os.listdir(folder)]

    for i, ax in enumerate(axes):
        if i < len(image_files):
            img = mpimg.imread(image_files[i])

            if pcd_type == "torso":
                cropped_img = img[100:1100, 500:1100]
            elif pcd_type in {"breast", "fascia"}:
                cropped_img = img[100:1100, 300:1300]
            else:
                cropped_img = img

            ax.imshow(cropped_img)
            ax.axis('off')
            ax.set_title(f"Feature {features_list[i]}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{pcd_type}.png"))
    plt.show()


def fpfh_matrix(pcds, voxel_size):
    pcds_fpfh = []
    for pcd in pcds:
        radius_normal = voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        pcds_fpfh.append(pcd_fpfh.data)
    return pcds_fpfh

def bounded_values_fpfh(fpfh_matrix):
    maximum = fpfh_matrix.max()
    minimum = fpfh_matrix.min()
    return maximum, minimum

def normalize_fpfh(fpfh_matrixes):
    normalized_matrixes = []
    for fpfh_matrix in fpfh_matrixes:
        normalized_matrix = np.zeros_like(fpfh_matrix)

        for index, row in enumerate(fpfh_matrix):
            row_min = np.min(row)
            row_max = np.max(row)
            if row_max != row_min:  # To avoid division by zero
                normalized_matrix[index] = (row - row_min) / (row_max - row_min)
            else:
                normalized_matrix[index] = row 
        normalized_matrixes.append(normalized_matrix)

    return normalized_matrixes


def global_normalization(pcd_matrixes):
    normalized_matrixes = []

    for pcd_matrix in pcd_matrixes:
        min_val = np.min(pcd_matrix)
        max_val = np.max(pcd_matrix)
        normalized_matrix = (pcd_matrix - min_val) / (max_val - min_val)
        normalized_matrixes.append(normalized_matrix)

    return normalized_matrixes

def sort_by_variance(pcd_matrixes):
    sorted_matrixes = []
    sorted_indices = []

    for pcd_matrix in pcd_matrixes:

        row_variances = np.var(pcd_matrix, axis=1)

        weights = row_variances / np.sum(row_variances)

        weighted_sums = np.dot(weights, pcd_matrix)

        sorted_index = np.argsort(weighted_sums)[::-1]  # Descending order
        sorted_matrix = pcd_matrix[:, sorted_index]

        sorted_matrixes.append(sorted_matrix)
        sorted_indices.append(sorted_index)
    
    return sorted_matrixes, sorted_indices

def sorted_interesting_points(nipples_indexes, sorted_indices):
    nipple_sorted_indices = []

    for i in range(len(nipples_indexes)):
        sorted_index_sublist = [sorted_indices[i].tolist().index(idx) for idx in nipples_indexes[i] if idx in sorted_indices[i]]
        nipple_sorted_indices.append(sorted_index_sublist)
    
    return nipple_sorted_indices


def remove_low_variance_variables(pcd_matrixes, n):
    lines_remained_indexes = []
    for pcd_matrix in pcd_matrixes:
        row_variances = np.var(pcd_matrix, axis=1)
        lines_to_remove = np.argsort(row_variances)[:n]
        lines_remained_index = [i for i in range(len(pcd_matrix)) if i not in lines_to_remove]
        print("Indexes of lines that remained after removing lines with smaller variances:")
        print(lines_remained_index)
        pcd_matrix = np.delete(pcd_matrix, lines_to_remove, axis=0)
        lines_remained_indexes.append(lines_remained_index)

    return pcd_matrixes, lines_remained_indexes

def remove_low_variance_rows(pcd_matrixes, threshold=[0.001, 0.0001], initial_row=6):
    filtered_matrixes = []
    high_variance_indices = []
    new_rows = []

    for i, pcd_matrix in enumerate(pcd_matrixes):
        variances = np.var(pcd_matrix, axis=1)
        high_variance_index = np.where(variances > threshold[i])[0]
        print("Indexes of lines that remained after removing lines with smaller variances:")
        print(high_variance_index)
        filtered_matrix = pcd_matrix[high_variance_index, :]
        new_row = np.where(high_variance_index == initial_row)[0][0]
        filtered_matrixes.append(filtered_matrix)
        high_variance_indices.append(high_variance_index)
        new_rows.append(new_row)

    return filtered_matrixes, high_variance_indices, new_rows

def log_transform(pcd_matrixes):
    log_matrixes = []
    for pcd_matrix in pcd_matrixes:
        log_matrix = np.log1p(pcd_matrix)
        log_matrixes.append(log_matrix)
    return log_matrixes

def nipples(fpfh_originals, threshold = 125):
    interesting_points = []
    for fpfh_original in fpfh_originals:
        pcd_fpfh_copy = copy.deepcopy(fpfh_original)
        fpfh_6th = pcd_fpfh_copy[6, :]
        interesting_point = np.where(fpfh_6th >= threshold)[0]
        interesting_points.append(interesting_point)
    return interesting_points

def higher_variance_rows_sorted(pcd_matrices, name="Matrix", remaining_indexes=None):
    higher_variance_rows = []
    for pcd_matrix in pcd_matrices:
        variances = np.var(pcd_matrix, axis=1)
        higher_variance_row = np.argsort(variances)[::-1]
        higher_variance_rows.append(higher_variance_row)
    
    for idx, list_variables in enumerate(higher_variance_rows):
        if remaining_indexes is not None:
            new_list_variables = [remaining_indexes[idx][j] for j in list_variables]
            print(f"{name} {idx + 1} \n{new_list_variables}")
        else:
            print(f"{name} {idx + 1} \n{list_variables}")

    return higher_variance_rows

def plot_matrices_side_by_side(matrices, titles, nipples_idx_list, new_row_list):
    
    n = len(matrices) // 2
    fig, axes = plt.subplots(n, 2, figsize=(20, n * 5))

    for i in range(n):
        for j in range(2):
            ax = axes[i, j] if n > 1 else axes[j]
            matrix_idx = 2 * i + j
            im = ax.imshow(matrices[matrix_idx], cmap="inferno", aspect='auto')
            ax.set_title(titles[matrix_idx])
            fig.colorbar(im, ax=ax)
            ax.set_xlim(0, matrices[matrix_idx].shape[1])
            ax.scatter(nipples_idx_list[matrix_idx], [new_row_list[matrix_idx]] * len(nipples_idx_list[matrix_idx]), color='red', marker='o', label='Interesting Points')
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def apply_pca(matrices, nipples, names, pca_numbers = [0,1]):

    pca = PCA(n_components=0.95, svd_solver= 'full')  # Explained variance threshold of 90%
    pca_result = pca.fit(matrices[0].T)
    # num_components = pca_result.shape
    # print(f"Number of components for {names[i]}: {num_components}")

    num_matrices = len(matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(10 * num_matrices, 5))

    if num_matrices == 1:
        axes = [axes]

    for i, (matrix, nipple) in enumerate(zip(matrices, nipples)):
        # pca = PCA(n_components=0.95, svd_solver= 'full')  # Explained variance threshold of 90%
        # pca_result = pca.fit_transform(matrix.T)
        # num_components = pca_result.shape
        # print(f"Number of components for {names[i]}: {num_components}")
        pca_result = pca.transform(matrix.T)
        coords_nipples = pca_result[nipple]

        ax = axes[i]
        ax.scatter(pca_result[:, pca_numbers[0]], pca_result[:, pca_numbers[1]], c='lightcoral', marker='.')
        ax.scatter(coords_nipples[:, pca_numbers[0]], coords_nipples[:, pca_numbers[1]], c='purple', marker='.')
        ax.set_title(f'{names[i]}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

    plt.show()

def features_against_features(matrices, names, nipples, num_features):
    num_matrices = len(matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(10 * num_matrices, 5))

    for i, (matrix, nipple) in enumerate(zip(matrices, nipples)):
        ax = axes[i]
        ax.scatter(matrix[num_features[0], :], matrix[num_features[1], :], c='lightcoral', marker='.')
        ax.scatter(matrix[num_features[0], nipple], matrix[num_features[1], nipple], c='purple', marker='.')
        ax.set_title(f'{names[i]}')
        ax.set_xlabel(f'Feature {num_features[0]}-th')
        ax.set_ylabel(f'Feature {num_features[1]}-th')
    
def features_against_features_3d(matrices, names, nipples, num_features):
    num_matrices = len(matrices)
    
    for i, (matrix, nipple) in enumerate(zip(matrices, nipples)):
        # Extract the required features from the matrix
        points = matrix[num_features, :].T
        nipple_points = matrix[num_features][:, nipple].T
        
        # Create point clouds for the main points and the nipple points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0.6, 0.6])  # lightcoral color
        
        nipple_pcd = o3d.geometry.PointCloud()
        nipple_pcd.points = o3d.utility.Vector3dVector(nipple_points)
        nipple_pcd.paint_uniform_color([0.5, 0, 0.5])  # purple color
        
        # Create a visualizer and add point clouds
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=names[i], width=800, height=600)
        vis.add_geometry(pcd)
        vis.add_geometry(nipple_pcd)
    
        
        vis.run()
        vis.destroy_window()
    
def clustering(pcds_fpfh, num_features:list):
    num_clusters = 3

    # Fit K-means on the first matrix
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pcds_fpfh[1].T)

    # Get the centroids from the fitted model
    centroids = kmeans.cluster_centers_

    # Create subplots
    num_matrices = len(pcds_fpfh)
    fig, axs = plt.subplots(1, num_matrices, figsize=(16, 6))

    for idx, (ax, pcd_fpfh) in enumerate(zip(axs, pcds_fpfh)):
        # Predict the labels for each matrix using the fitted model
        labels = kmeans.predict(pcd_fpfh.T)

        # Visualize the clusters on the respective subplot
        for i in range(num_clusters):
            ax.scatter(pcd_fpfh.T[labels == i, num_features[0] ], pcd_fpfh.T[labels == i, num_features[1]], label=f'Cluster {i}')
        ax.scatter(centroids[:, num_features[0]], centroids[:, num_features[1]], s=300, c='red', label='Centroids')
        ax.set_title(f'Matrix {idx + 1}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Breast = "Rigid_Registration/Manequin/Mannequin_Breast_ASCII.ply"
    Fascia = "Rigid_Registration/Manequin/Mannequin_Fascia_ASCII.ply"
    Torso = "Rigid_Registration/Manequin/Mannequin_Torso_ASCII.ply"
    voxel_downsample = 0.01

    pcds = [Torso, Breast]

    pcd = o3d.io.read_point_cloud(pcd_path)

    pcd_fpfh = fpfh_matrix(pcd, voxel_size=voxel_downsample)
    # fpfh_normalized = normalize_fpfh(pcd_fpfh)            #Does not work
    fpfh_normalized = global_normalization(pcd_fpfh)
    log_fpfh = log_transform(fpfh_normalized)
    filtered_fpfh, remaining_idx, new_row = remove_low_variance_rows(log_fpfh, 0.001, 6)

    matrix_weighted, sorted_indices = sort_by_variance(filtered_fpfh)
    
    nipples_idx = nipples(pcd_fpfh, 100)

    # nipples_idx = [25832, 25833, 28640, 28641, 30549, 40308, 40313, 40315, 40316, 40317, 40318, 40320, \
    #                40321, 40322, 40323, 40324, 40326, 40327, 40333, 54466, 54471, 54472, 62170, 62171, \
    #                 62176, 62241, 62247, 75170, 75172, 75221, 75222, 75223, 75226, 75235, 75240, 75242, \
    #                     75243, 75244, 75270, 75271, 75273, 75274, 75275, 75276, 75279, 75288, 75290, 75291, \
    #                         75297, 75305, 75306, 89758, 89773, 89779, 89780, 89791, 89796, 89805, 89808, 89814, \
    #                             89815, 89834, 89885, 89932, 89936, 89950, 89987, 89999, 99950, 99966]

    
    nipple_sorted_indices = [sorted_indices.tolist().index(idx) for idx in nipples_idx if idx in sorted_indices]


    matrices = [filtered_fpfh, matrix_weighted]
    matrices_names = ["Remove variables", "Weighted"]

    # plot_color_map(matrices, matrices_names)

