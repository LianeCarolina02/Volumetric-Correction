import open3d as o3d
import visualization as vis
import prepare_dataset as prd
import numpy as np
import time
import evaluation as ev

start_time = time.time()
sigma = 0.001

Breast = "Manequin/Mannequin_Breast_ASCII.ply"
Breast_noise = f"Noise_ply/Breast_Noise_{sigma}.ply"

Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"

Torso = "Manequin/Mannequin_Torso_ASCII.ply"

def vanilla_icp(source, target, threshold):
    print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))

    trans_init = np.identity(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return reg_p2p


if __name__ == '__main__':
    voxel_size = 0.01
    threshold = 0.02

    source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(Breast, Torso, voxel_size = voxel_size)

    vanilla_icp = vanilla_icp(source_down, target_down, threshold)

    # vis.draw_registration_result(source, target, vanilla_icp.transformation)

    ev = ev.evaluation(source, target, vanilla_icp.transformation)

    end_time = time.time()

    duration = end_time - start_time
    print(f"Duration of {duration}")

