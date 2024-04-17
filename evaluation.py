
import open3d as o3d
import numpy as np
import sys

def evaluation(source, target, trans_init, threshold = 0.02,):

    source_point = np.asarray(source)
    print(f"\n No. Points of the Source: {source_point} \n")

    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)

    ev_points = np.asarray(evaluation.correspondence_set)
    print(f":: Evaluation")
    print(f"    Fitness:{evaluation.fitness} \n")
    print(f"    Inlier RMSE: {evaluation.inlier_rmse} \n")
    # print(f"    Correspondence Set: {ev_points} \n")
    print(f"    Transformation: \n {evaluation.transformation}")

