import datetime
import open3d as o3d


def get_current_datetime():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m__%H_%M_%S")
    return formatted_datetime

Breast_ascii = "Manequin/Mannequin_Breast_ASCII.ply"
Torso_ascii = "Manequin/Mannequin_Torso_ASCII.ply"

def preprocessing(pcd, voxel_size = 0.01):
    # print(":: Downsampling with voxel %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    
    radius_normal = voxel_size * 2
    # print(":: Normal with search radius %.3f." % radius_normal)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Computation of FPFH feature with search radius %.3f." % radius_feature)

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_pcd, target_pcd, voxel_size=0.01):
    target = o3d.io.read_point_cloud(target_pcd)
    source = o3d.io.read_point_cloud(source_pcd)

    # print(":: Prepare Source Dataset")
    source_down, source_fpfh = preprocessing(source, voxel_size)
    # print(":: Prepare Target Dataset")
    target_down, target_fpfh = preprocessing(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh

