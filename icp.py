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


if __name__ == '__main__':
    voxel_size = 0.01
    threshold = 0.2
    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    Breast = "Manequin/Mannequin_Breast_ASCII.ply"
    Fascia = "Manequin/Mannequin_Fascia_ASCII.ply"
    Torso = "Manequin/Mannequin_Torso_ASCII.ply"

    sources_Breast = [Breast]
    sources_Fascia = [Fascia]

    for sigma in sigmas:
        sources_Breast.append(f"Noise_ply/Breast_Noise_{sigma}.ply")
        sources_Fascia.append(f"Noise_ply/Fascia_Noise_{sigma}.ply")

    n_sources = len(sources_Fascia)

    fitness_1st = []
    rmse_1st = []
    fitness = []
    rmse = []
    durations = []

    end_time_0 = time.time()

    duration_0 = end_time_0 - start_time_0
    types = [sources_Breast, sources_Fascia]

    for type in types:
        for idx in type:
            start_time_1st = time.time()
            print(f"\n \n \n {idx}")

            print("::Preparing Dataset")
            source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(idx, Torso, voxel_size = voxel_size)
            
            print("RANSAC trasnformation")
            first_transformation = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size, distance=5)
            
            end_time_1st = time.time()
            duration_1st = end_time_1st - start_time_1st

            original_source = o3d.io.read_point_cloud(type[0])

            print("Metrics appending")

            fitness_1st.append(ev.evaluation(original_source, target, trans_init=first_transformation.transformation, threshold=threshold).fitness)
            rmse_1st.append(ev.evaluation(original_source, target, trans_init=first_transformation.transformation, threshold=threshold).inlier_rmse)

            print("Vanilla ICP Transformation")
            start_time_2nd = time.time()

            source_down.transform(first_transformation.transformation)
            icp = vanilla_icp(source_down, target_down, threshold)

            transformation = icp.transformation @ first_transformation.transformation

            end_time_2nd = time.time()
            duration_2nd = end_time_2nd - start_time_2nd

            fitness.append(ev.evaluation(original_source, target, trans_init=transformation, threshold=threshold).fitness)
            rmse.append(ev.evaluation(original_source, target, trans_init=transformation, threshold=threshold).inlier_rmse)
            durations.append(duration_0 + duration_1st + duration_2nd)

print(f"\n RMSE 1st {rmse_1st} \n")
print(f"\n Fitness 1st: {fitness_1st} \n")
print(f"\n RMSE: {rmse} \n")
print(f"\n Fitness: {fitness} \n")
print(f"\n Durations {durations} \n")



    # source, target, source_down, target_down, source_fpfh, target_fpfh = prd.prepare_dataset(idx, Torso, voxel_size = voxel_size)
    # first_transformation = RANSAC.global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh, voxel_size, distance=5)
        
    # source = o3d.io.read_point_cloud(Breast)

    # end_time = time.time()
    # save_time = end_time - start_time

    # vis.draw_registration_result(source, target, first_transformation.transformation)

    # start_time_1 = time.time()
    # source_down.transform(first_transformation.transformation)

    # # robust_icp = robust_icp(source_down, target_down, threshold, sigma)
    # vanilla_icp = vanilla_icp(source_down, target_down, threshold)

    # # transformation = robust_icp.transformation @ first_transformation.transformation
    # transformation = vanilla_icp.transformation @ first_transformation.transformation

    # end_time_1 = time.time()
    # duration = (end_time_1 - start_time_1) + save_time

    # vis.draw_registration_result(source, target, transformation)

    # print(f"Duration of {duration}")

