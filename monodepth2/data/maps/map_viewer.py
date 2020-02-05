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
        """ Returns a RGB image and depth image of virtual camera view.
        """
        w, h = camera.out_shape
        color_img = np.zeros((h,w,3), dtype='uint8')
        cam2enu, enu2cam = camera.get_transforms()

        pos = cam2enu[:3, 3]
        roadmarkers = self.map_reader.get_road_markers(pos)

        # Draw solid lines
        for line in self.map_reader.parse_solid_lines(roadmarkers):
            img_pts, depths = self.project_landmarks(line, camera)
            color_img = draw_poly(color_img, img_pts, color=(0,255,0))

        # Draw curb
        for line in self.map_reader.parse_curb(roadmarkers):
            img_pts, depths = self.project_landmarks(line, camera)
            color_img = draw_poly(color_img, img_pts, color=(255,0,0))

        # Draw dashed lines
        for line in self.map_reader.parse_dash_lines(roadmarkers):
            img_pts, depths = self.project_landmarks(line, camera)
            color_img = draw_poly(color_img, img_pts, color=(0,0,255))

        # Draw dash blobs
        for blob in self.map_reader.parse_dash_blobs(roadmarkers):
            img_pts, depths = self.project_landmarks(blob, camera)
            if len(img_pts) == len(blob):
                color_img = draw_poly(color_img, img_pts, color=(0,255,0))

        # Draw surface markings
        for blob in self.map_reader.parse_surface_markings(roadmarkers):
            img_pts, depths = self.project_landmarks(blob, camera)
            if len(img_pts) == len(blob):
                color_img = draw_poly(color_img, img_pts, color=(255,255,0))

        color_img = Image.fromarray(color_img, 'RGB')
        return color_img
    
    def project_landmarks(self, landmarks, camera):
        if len(landmarks) == 0:
            return [], []

        w, h = camera.out_shape
        cam2enu, enu2cam = camera.get_transforms()
        landmarks = enu2cam[:3, :3].dot(landmarks.T).T + enu2cam[:3, 3]

        # Filter for landmarks in front of camera (Z > 0)
        landmarks = np.array([l for l in landmarks if l[2] > 0])
        if len(landmarks) == 0:
            return [], []

        # Project landmarks onto image plane
        intrinsic = scale_cam_intrinsic(camera.intrinsic, camera.img_shape, camera.out_shape)
        distortion = camera.distortion
        tcw = np.eye(4)
        img_pts, _ = cv.projectPoints(landmarks, tcw[:3, :3], tcw[:3, 3], intrinsic, distortion)
        
        img_pts = img_pts[:, 0, :]
        depths = landmarks[:,2]
        
        filtered = [(p,d) for p,d in zip(img_pts, depths) if p[0] > 0 and p[0] < w and p[1] > 0 and p[1] < h]
        if len(filtered) == 0:
            return [],[]
        else:
            return zip(*filtered)

def draw_poly(img, pts, color=(0,255,0)):
    if len(pts) == 0:
        return img

    pts = np.round(pts).astype(int)
    img = cv.fillConvexPoly(img, pts, color=color)
    return img

# def draw_line(img, pts, color=(0,255,0)):
#     if len(pts) == 0:
#         return img

#     pts = np.round(pts).astype(int)
#     pts = pts.reshape((-1,1,2))
#     img = cv.polylines(img, [pts], False, color=color, thickness=2)
#     return img

def draw_line(img, pts, depths, color=(0,255,0)):
    for p0, p1, d0, d1 in zip(pts[:-1], pts[1:], depths[:-1], depths[1:]):
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        cv.line(img, p0, p1, color=color, thickness=2)
    return img

def draw_points(img, points, values):
    for pt, v in zip(points, values):
        pt = np.round(pt)
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img, pt, color=v, radius=10, thickness=-1)
    return img


class MapCamera:
    def __init__(self, calib):
        self.intrinsic = calib['intrinsic']
        self.extrinsic = calib['extrinsic']['imu-0']
        self.distortion = calib['distortion'].squeeze()
        self.img_shape = calib['img_shape']
        self.out_shape = calib['img_shape']

        self.T = np.eye(4)
    
    def set_out_shape(self, shape):
        self.out_shape = shape

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
