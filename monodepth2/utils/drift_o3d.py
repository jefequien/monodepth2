import open3d as o3d
import numpy as np

def compute_drift_transform(img0, depth0, img1, depth1, intrinsic):
    source_color = o3d.geometry.Image(np.array(img0))
    source_depth = o3d.geometry.Image(depth0)
    target_color = o3d.geometry.Image(np.array(img1))
    target_depth = o3d.geometry.Image(depth1)

    print(np.min(depth0), np.max(depth0))
    print(np.min(depth1), np.max(depth1))
    

    w, h = img0.size
    fx = intrinsic[0,0] * w
    fy = intrinsic[1,1] * h
    cx = intrinsic[0,2] * w
    cy = intrinsic[1,2] * h
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    # target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     target_rgbd_image, pinhole_camera_intrinsic)

    odo_init = np.identity(4)
    iteration_number_per_pyramid_level = o3d.utility.IntVector([ 10, 5, 2,])
    max_depth_diff = 1.
    min_depth = 0.
    max_depth = 100.
    option = o3d.odometry.OdometryOption(
        iteration_number_per_pyramid_level,
        max_depth_diff, 
        min_depth,
        max_depth
    )
    print(option)


    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
        odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_hybrid_term:
        return trans_hybrid_term
        # print("Using Hybrid RGB-D Odometry")
        # print(trans_hybrid_term)
        # source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     source_rgbd_image, pinhole_camera_intrinsic)
        # source_pcd_hybrid_term.transform(trans_hybrid_term)
        # o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
    else:
        return None