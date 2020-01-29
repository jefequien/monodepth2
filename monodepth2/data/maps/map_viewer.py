import numpy as np
import cv2 as cv
from PIL import Image

from .map_reader import MapReader
from .map_utils import scale_cam_intrinsic, get_rotation_translation_mat

class MapViewer:

    def __init__(self, map_name):
        self.map_reader = MapReader(map_name)

    def get_view(self, camera):

        landmarks = self.map_reader.get_landmarks(camera.rt[1])

        cam2enu, enu2cam = camera.get_transforms()
        landmarks = enu2cam[:3, :3].dot(landmarks.T).T + enu2cam[:3, 3]

        # imu2emu = get_rotation_translation_mat(*camera.rt)
        # enu2imu = np.linalg.inv(imu2emu)
        # cam2imu = camera.extrinsic
        # imu2cam = np.linalg.inv(cam2imu)
        # # Apply transforms
        # landmarks = enu2imu[:3, :3].dot(landmarks.T).T + enu2imu[:3, 3]
        # landmarks = imu2cam[:3, :3].dot(landmarks.T).T + imu2cam[:3, 3]

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

        self.rt = None


    def set_position(self, position):
        self.rt = position[3:], position[:3]

        imu2emu = get_rotation_translation_mat(*self.rt)
        enu2imu = np.linalg.inv(imu2emu)
        cam2imu = self.extrinsic
        imu2cam = np.linalg.inv(cam2imu)

        r0, t0 = get_rt_vecs(imu2cam)
        r1, t1 = get_rt_vecs(enu2imu)

        # r1, t1 = position[3:], position[:3]
        self.rt = compose_rt_vecs(r0, t0, r1, t1)
        # self.rt = r1, t1

    def apply_T(self, cam_T):
        r0, t0 = get_rt_vecs(cam_T)
        r1, t1 = self.rt
        self.rt = compose_rt_vecs(r0, t0, r1, t1)
    
    def get_transforms(self):
        cam2enu = get_rotation_translation_mat(*self.rt)
        enu2cam = np.linalg.inv(cam2enu)
        return cam2enu, enu2cam

def get_rt_vecs(mat):
    r = cv.Rodrigues(mat[:3, :3])[0]
    t = mat[:3, 3]
    return r.reshape(3), t.reshape(3)

def compose_rt_vecs(r0, t0, r1, t1):
    r, t = cv.composeRT(r0, t0, r1, t1)[:2]
    return r.reshape(3), t.reshape(3)