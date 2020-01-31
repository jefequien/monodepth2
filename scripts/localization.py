import os
import time
import numpy as np
import cv2 as cv
from PIL import Image

from monodepth2.model import MonodepthModel
from monodepth2.data.maps.map_viewer import MapViewer, MapCamera
from monodepth2.utils.calibration_manager import CalibrationManager
from monodepth2.utils.visualize import vis_depth


class LocalizationModel:

    def __init__(self, cfg):
        self.output_dir = cfg.OUTPUT_DIR
        self.vis_images = {}

        # Model
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
        self.model = MonodepthModel(cfg)
        self.model.load_model(save_folder)
        self.model.set_eval()

        # Localization params
        self.position = None
        self.cam_Ts = {}
        self.observation = {}
        self.initialized = False

        self.timers = {}
        for i in range(5):
            self.timers[i] = time.time()
    
    def setup(self, bag_info):
        bag_name, map_name, begin, end = bag_info
        self.calib_manager = CalibrationManager(dataset=bag_name)
        self.camera_calibs = self.calib_manager.get_cameras()

        self.map_viewer = MapViewer(map_name)
        self.map_cameras = {
            'cam{}'.format(cam_id): MapCamera(self.camera_calibs[cam_id]) 
            for cam_id in self.camera_calibs.keys()
        }

    def initialize(self, observation):
        self.cam_names = [k for k in observation.keys() if 'cam' in k]
        self.position = observation['gps_data']
        for cam_name in self.cam_names:
            self.map_cameras[cam_name].set_position(self.position)

        self.last_observation = observation
        self.initialized = True

    def step(self, observation):
        if not self.initialized:
            self.initialize(observation)
        
        all_data = self.prepare_data(observation)
        all_preds = self.model.predict(all_data)

        # Update from predictions
        for cam_name, data, cam_T, depth in zip(self.cam_names, all_data, all_preds['cam_T'], all_preds['depth']):

            # self.map_cameras[cam_name].set_position(observation['gps_data'])
            self.map_cameras[cam_name].apply_T(cam_T)

            # Visualize
            color_img = data[0, 'color']
            depth_img = vis_depth(depth)

            pose_img = self.map_viewer.get_view(self.map_cameras[cam_name])
            pose_img = pose_img.resize(color_img.size, Image.ANTIALIAS)
            color_img.paste(pose_img, (0,0), pose_img.convert('L'))

            self.vis_images['{} color'.format(cam_name)] = color_img
            self.vis_images['{} depth'.format(cam_name)] = depth_img
        self.show()

        self.last_observation = observation
        return None

    
    def prepare_data(self, observation):
        """ Create dataset-like object from training.
        Arguments:
            observation: cam_id -> image
        Return:
            all_data: [] for batching
        """
        all_data = []
        for cam_name in self.cam_names:
            img0 = observation[cam_name]
            img1 = self.last_observation[cam_name]
            calib = self.get_calibration(cam_name)

            data = {}
            data[0, 'color'] = img0
            data[0, 'calib'] = calib
            data[-1, 'color'] = img1
            data[-1, 'calib'] = calib

            all_data.append(data)
        return all_data
    
    def get_calibration(self, cam_name):
        cam_id = int(cam_name.replace('cam', ''))
        intrinsic = self.camera_calibs[cam_id]['intrinsic']
        extrinsic = self.camera_calibs[cam_id]['extrinsic']['imu-0']
        distortion = self.camera_calibs[cam_id]['distortion'].squeeze()
        img_shape = self.camera_calibs[cam_id]['img_shape']
        
        # Scale intrinsic to be per unit size
        K = np.eye(4)
        K[:3,:3] = intrinsic
        K[0, :] /= img_shape[0] # width
        K[1, :] /= img_shape[1] # height
        
        calib = {}
        calib['K'] = np.array(K, dtype=np.float32)
        calib['ext_T'] = np.array(extrinsic, dtype=np.float32)
        return calib

    def show(self):
        for name, img in self.vis_images.items():
            if img.size[0] > 1000:
                img = img.resize((img.size[0] // 4, img.size[1] // 4), Image.ANTIALIAS)
            img = np.array(img)[:,:,::-1]
            cv.imshow('{}'.format(name), img)
        cv.waitKey(1)