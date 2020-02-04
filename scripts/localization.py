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
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.vis_images = {}

        # Model
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
        self.model = MonodepthModel(cfg)
        self.model.load_model(save_folder)
        self.model.to(self.device)
        self.model.set_eval()

        # Localization params
        self.position = None
        self.cam_Ts = {}
        self.observation = {}
        self.initialized = False
        self.num_steps = 0

        self.timers = {}
        for i in range(5):
            self.timers[i] = time.time()
    
    def setup(self, bag_info):
        bag_name, map_name = bag_info[:2]
        self.calib_manager = CalibrationManager(dataset=bag_name)
        self.camera_calibs = self.calib_manager.get_cameras()

        self.map_viewer = MapViewer(map_name)
        self.map_cameras = {
            'cam{}'.format(cam_id): MapCamera(self.camera_calibs[cam_id]) 
            for cam_id in self.camera_calibs.keys()
        }

    def initialize(self, observation):
        # self.cam_names = [k for k in observation.keys() if 'cam' in k]
        self.cam_names = ['cam1']
        self.position = observation['gps_data']
        for cam_name in self.cam_names:
            self.map_cameras[cam_name].set_position(self.position)
            self.map_cameras[cam_name].set_out_shape(observation['cam1'].size)
            self.map_view, _ = self.map_viewer.get_view(self.map_cameras[cam_name])

        self.last_observation = observation
        self.initialized = True

    def step(self, observation):
        if not self.initialized:
            self.initialize(observation)
        
        all_data = self.prepare_data(observation)
        all_preds = self.model.predict(all_data)

        # Update from predictions
        for cam_name, data, cam_T, drift_T, depth in zip(self.cam_names, all_data, all_preds['cam_T'], all_preds['drift_T'], all_preds['depth']):
            img = self.last_observation['cam1']
            map_pred = self.last_observation['map_pred/{}'.format(cam_name)]

            # Scale predictions by 10
            cam_T[:3, 3] *= 10
            drift_T[:3, 3] *= 10
            depth *= 10

            # Correct last step drift
            self.map_cameras[cam_name].apply_T(drift_T)
            aligned_view, _ = self.map_viewer.get_view(self.map_cameras[cam_name])

            # self.map_cameras[cam_name].apply_T(cam_T)
            if self.num_steps % 1 == 0:
                self.map_cameras[cam_name].set_position(observation['gps_data'])

            self.map_view, _ = self.map_viewer.get_view(self.map_cameras[cam_name])

            # Visualize
            depth_img = vis_depth(depth)
            # map_depth_img = vis_depth(map_depth, vmax=0.1)

            map_vis = img.copy()
            map_vis.paste(aligned_view, (0,0), aligned_view.convert('L'))
            pred_vis = img.copy()
            pred_vis.paste(map_pred, (0,0), map_pred.convert('L'))

            self.vis_images['{} color'.format(cam_name)] = img
            self.vis_images['{} depth'.format(cam_name)] = depth_img
            self.vis_images['{} map_view'.format(cam_name)] = map_vis
            self.vis_images['{} map_pred'.format(cam_name)] = pred_vis
            # self.vis_images['{} map_depth_img'.format(cam_name)] = map_depth_img
        self.show()

        self.last_observation = observation
        self.num_steps += 1
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
            map_view = self.map_view
            map_pred = self.last_observation['map_pred/{}'.format(cam_name)]

            data = {}
            data[0, 'color'] = img0
            data[0, 'calib'] = calib
            data[-1, 'color'] = img1
            data[-1, 'calib'] = calib
            data[0, 'map_view'] = map_view
            data[0, 'map_pred'] = map_pred

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
            img = np.array(img)
            if np.ndim(img) == 3:
                img = img[:,:,::-1]
            cv.imshow('{}'.format(name), img)
        cv.waitKey(1)