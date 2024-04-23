import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import visualization as vis
import prepare_dataset as prd
import icp 
import RANSAC
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


start_time = time.time()

breast = "Manequin/Mannequin_Breast_ASCII.ply"
fascia = "Manequin/Mannequin_Fascia_ASCII.ply"
torso = "Manequin/Mannequin_Torso_ASCII.ply"
voxel_size = 0.01
threshold = 0.02


source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(breast, torso, voxel_size = voxel_size)
first_transformation = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size, distance=5)

end_time = time.time()
save_time = end_time - start_time

# vis.draw_registration_result(source, target, first_transformation.transformation)

start_time_1 = time.time()
source_down.transform(first_transformation.transformation)

vanilla_icp = icp.vanilla_icp(source_down, target_down, threshold)

transformation = vanilla_icp.transformation @ first_transformation.transformation

# vis.draw_registration_result(source, target, transformation)

# Features Extraction
radius_normal = voxel_size * 2

source.transform(transformation)

source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

radius_feature = voxel_size * 5

pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

fpfh_6th_source = np.asarray(pcd_fpfh_source.data[6,:])
fpfh_6th_target = np.asarray(pcd_fpfh_target.data[6,:])

# Important features

def retain_90_percent_variance(X):
    # Inicializa o PCA

    means = X.mean(axis=0)
    std = X.std(axis=0)
    std[std==0] = 1
    X = (X - means)/std

    pca = PCA()
    # Ajusta o PCA aos dados
    pca.fit(X)
    
    # Determina o número de componentes principais que retêm 90% da variância
    total_variance = np.sum(pca.explained_variance_)
    target_variance = 0.9 * total_variance
    retained_variance = 0
    num_components = 0
    for variance in pca.explained_variance_:
        retained_variance += variance
        num_components += 1
        if retained_variance >= target_variance:
            break
    
    # Reduz a dimensão para o número de componentes principais encontrados
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca

X_reduced = retain_90_percent_variance(pcd_fpfh_source.data.T).T
# X_reduced = fpfh_6th_source 

# Clustering interesting features
X_ordered = np.sort(fpfh_6th_source)
Y = np.arange(1, len(X_ordered))

gm = GaussianMixture(n_components=3, random_state=0).fit_predict(fpfh_6th_source.reshape(-1,1))
# kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X_reduced[0].reshape(-1,1))

colors = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 0], [0.5, 0.5, 0.5]])[gm]  # Blue for 0, Red for 1

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = source.points
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
# Visualization
colormap = 'inferno'

fpfh_colors = plt.get_cmap(colormap )(
        (X_reduced[0] - X_reduced[0].min()) / (X_reduced[0].max()- X_reduced[0].min()))

fpfh_colors = fpfh_colors[:, :3]

fpfh_pcd =o3d.geometry.PointCloud()
fpfh_pcd.points = source.points
fpfh_pcd.normals = source.normals
fpfh_pcd.colors = o3d.utility.Vector3dVector(fpfh_colors)

vis = o3d.visualization.Visualizer()
vis.create_window(width=1600, height=1200)
vis.add_geometry(fpfh_pcd)
vis.run()

# vis.capture_screen_image(f"{folder}/Feature_{i}_torso.png")

vis.destroy_window()

