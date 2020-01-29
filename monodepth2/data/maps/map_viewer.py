import numpy as np
import cv2 as cv
from PIL import Image

from .map_reader import MapReader
from .map_utils import scale_cam_intrinsic, get_rotation_translation_mat

class MapViewer:

    def __init__(self, map_name):
        self.map_reader = MapReader(map_name)

    def get_view(self, camera):
        cam2enu, enu2cam = camera.get_transforms()

        t = cam2enu[:3, 3]
        landmarks = self.map_reader.get_landmarks(t)
        landmarks = enu2cam[:3, :3].dot(landmarks.T).T + enu2cam[:3, 3]

        # Filter for landmarks in front of camera (Z > 0)
        landmarks = np.array([l for l in landmarks if l[2] > 0])

        # Project landmarks onto image plane
        w, h = camera.out_shape
        intrinsic = scale_cam_intrinsic(camera.intrinsic, camera.img_shape, camera.out_shape)
        distortion = camera.distortion
        tcw = np.eye(4)
        img_pts, _ = cv.projectPoints(landmarks, tcw[:3, :3], tcw[:3, 3], intrinsic, distortion)
        img_pts = img_pts[:, 0, :]
        img_pts[:, 0] = np.clip(img_pts[:, 0], 0, w - 1)
        img_pts[:, 1] = np.clip(img_pts[:, 1], 0, h - 1)

        view_img = np.zeros((h,w,3), dtype='uint8')
        view_img = draw_points(view_img, img_pts, color=(0, 255, 0))
        view_img = Image.fromarray(view_img, 'RGB')
        return view_img
    
def draw_points(img, points, color=(0, 255, 0)):
    for pt in points:
        pt = np.round(pt)
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img, pt, color=color, radius=2, thickness=-1)
    return img


class MapCamera:
    def __init__(self, calib):
        self.intrinsic = calib['intrinsic']
        self.extrinsic = calib['extrinsic']['imu-0']
        self.distortion = calib['distortion'].squeeze()
        self.img_shape = calib['img_shape']
        self.out_shape = calib['img_shape']

        self.T = np.eye(4)

    def set_position(self, position):
        r, t = position[3:], position[:3]
        imu2emu = get_rotation_translation_mat(r, t)
        cam2imu = self.extrinsic
        self.T = imu2emu.dot(cam2imu)

    def apply_T(self, cam_T):
        self.T = self.T.dot(cam_T)
    
    def get_transforms(self):
        cam2enu = self.T
        enu2cam = np.linalg.inv(cam2enu)
        return cam2enu, enu2cam
