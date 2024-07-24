import open3d as o3d
import numpy as np
import time


def vanilla_icp(source, target, threshold, trans_init):
    # print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return reg_p2p

def robust_icp(source, target, threshold):
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))
    loss = o3d.pipelines.registration.TukeyLoss(k=0.02)
    print("Using robust loss:", loss)
    trans_init = np.identity(4)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                        threshold, trans_init,
                                                        p2l)
    

    
    return reg_p2l