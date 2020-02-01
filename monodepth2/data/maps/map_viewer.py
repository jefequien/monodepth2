import numpy as np
import cv2 as cv
from PIL import Image
import time

from .map_reader import MapReader
from .map_utils import scale_cam_intrinsic, get_rotation_translation_mat

class MapViewer:

    def __init__(self, map_name):
        self.map_reader = MapReader(map_name)

    def get_view(self, camera):
        w, h = camera.out_shape
        view_img = np.zeros((h,w,3), dtype='uint8')
        cam2enu, enu2cam = camera.get_transforms()

        pos = cam2enu[:3, 3]
        solid_lines = self.map_reader.get_solid_lines(pos)
        for line in solid_lines:
            line_pts = self.project_landmarks(line, camera)
            view_img = draw_line(view_img, line_pts, color=(255,0,0))

        dash_lines = self.map_reader.get_dash_lines(pos)
        for dash in dash_lines:
            dash_pts = self.project_landmarks(dash, camera)
            if len(dash_pts) == 4:
                view_img = draw_poly(view_img, dash_pts, color=(0,255,0))

        view_img = Image.fromarray(view_img, 'RGB')
        return view_img
    
    def project_landmarks(self, landmarks, camera):
        if len(landmarks) == 0:
            return []

        w, h = camera.out_shape
        cam2enu, enu2cam = camera.get_transforms()
        landmarks = enu2cam[:3, :3].dot(landmarks.T).T + enu2cam[:3, 3]

        # Filter for landmarks in front of camera (Z > 0)
        landmarks = np.array([l for l in landmarks if l[2] > 0])
        if len(landmarks) == 0:
            return []

        # Project landmarks onto image plane
        intrinsic = scale_cam_intrinsic(camera.intrinsic, camera.img_shape, camera.out_shape)
        distortion = camera.distortion
        tcw = np.eye(4)
        img_pts, _ = cv.projectPoints(landmarks, tcw[:3, :3], tcw[:3, 3], intrinsic, distortion)
        img_pts = img_pts[:, 0, :]
        img_pts = np.array([p for p in img_pts if p[0] > 0 and p[0] < w and p[1] > 0 and p[1] < h])

        return img_pts

def draw_poly(img, pts, color=(0,255,0)):
    if len(pts) == 0:
        return img

    pts = np.round(pts).astype(int)
    img = cv.fillConvexPoly(img, pts, color=color)
    return img

def draw_line(img, pts, color=(0,255,0)):
    if len(pts) == 0:
        return img

    pts = np.round(pts).astype(int)
    pts = pts.reshape((-1,1,2))
    img = cv.polylines(img, [pts], False, color=color, thickness=5)
    return img

def draw_points(img, points, color=(0, 255, 0)):
    for pt in points:
        pt = np.round(pt)
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img, pt, color=color, radius=4, thickness=-1)
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
