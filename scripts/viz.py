import numpy as np
import cv2 as cv

from map_reader import MapReader
from utils import scale_cam_intrinsic, get_rotation_translation_mat

class LocalizationVisualizer:

    def __init__(self, bag_info):
        bag_name, map_name, _, _ = bag_info

        self.map_reader = MapReader(map_name)

        # Cameras
        self.cam_ids = [1]
        self.cameras = [Camera(cam_id, camera_calibs[cam_id]) for cam_id in self.cam_ids]

        # Visualizers
        self.camera_visualizers = {
            camera.cam_id: CameraPoseVisualizer(camera)
            for camera in self.cameras
        }
    

    def visualize(self, data, prediction):
        vis_images = {}

        # Get landmarks
        localization = prediction['localization']
        landmarks = self.map_reader.get_landmarks(localization)

        # Visualize each camera fitting
        for cam_id in self.cam_ids:
            img = data[cam_id]

            pose_info = {}
            pose_info['localization'] = localization
            pose_info['tcw'] = prediction['cam{}_tcw'.format(cam_id)]

            cam_visualizer = self.camera_visualizers[cam_id]
            vis_images[cam_id] = cam_visualizer.visualize(img, pose_info, landmarks)
        return vis_images
    
    def show(self, vis_images):
        for name in vis_images:
            cv.imshow(name, vis_images[name])
        cv.waitKey(1)

class CameraPoseVisualizer:
    """ Visualizes camera pose fitting by projecting 3D landmarks onto
    an image.
    """

    def __init__(self, camera):
        self.camera = camera

    def visualize(self, img, pose_info, landmarks):
        """ 
        """
        proj = self.project_landmarks(img, pose_info, landmarks)
        img = draw_points(img, proj, color=(0, 255, 0))

        pose_info['tcw'] = np.eye(4)
        # img = draw_points(img, proj_without_tcw, color=(0, 0, 255))

        return img

    def project_landmarks(self, img, pose_info, landmarks):
        t = pose_info['localization'][:3]
        r = pose_info['localization'][3:]
        tcw = pose_info['tcw']

        # Transform matrices
        imu2enu = get_rotation_translation_mat(r, t)
        enu2imu = np.linalg.inv(imu2enu)
        cam2imu = self.camera.extrinsic
        imu2cam = np.linalg.inv(cam2imu)

        # Apply transforms
        landmarks = enu2imu[:3, :3].dot(landmarks.T).T + enu2imu[:3, 3]
        landmarks = imu2cam[:3, :3].dot(landmarks.T).T + imu2cam[:3, 3]

        # Filter for landmarks in front of camera (Z > 0)
        landmarks = np.array([l for l in landmarks if l[2] > 0])

        # Project landmarks onto image plane
        h, w = img.shape[:2]
        intrinsic = scale_cam_intrinsic(self.camera.intrinsic, self.camera.img_shape, (w,h))
        distortion = self.camera.distortion
        img_pts, _ = cv.projectPoints(landmarks, tcw[:3, :3], tcw[:3, 3], intrinsic, distortion)
        img_pts = img_pts[:, 0, :]
        img_pts[:, 0] = np.clip(img_pts[:, 0], 0, w - 1)
        img_pts[:, 1] = np.clip(img_pts[:, 1], 0, h - 1)
        return img_pts
    
def draw_points(img, points, color=(0, 255, 0)):
    for pt in points:
        pt = np.round(pt)
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img, pt, color=color, radius=2, thickness=-1)
    return img


class Camera:

    def __init__(self, cam_id, calib):
        self.cam_id = cam_id
        self.intrinsic = calib['intrinsic']
        self.extrinsic = calib['extrinsic']['imu-0']
        self.distortion = calib['distortion'].squeeze()
        self.img_shape = calib['img_shape']
    