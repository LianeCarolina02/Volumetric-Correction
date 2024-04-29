import open3d as o3d
import visualization as vis
import prepare_dataset as prd
import numpy as np
import time
import RANSAC
import evaluation as ev

start_time_0 = time.time()

def vanilla_icp(source, target, threshold):
    # print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))

    trans_init = np.identity(4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return reg_p2p

def robust_icp(source, target, threshold, sigma):
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)
    trans_init = np.identity(4)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                        threshold, trans_init,
                                                        p2l)
    
    return reg_p2l