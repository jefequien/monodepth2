import os
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

        self.position = observation['gps_data']
        for cam_name in self.cam_names:
            self.map_cameras[cam_name].set_position(self.position)

        # Update from predictions
        for cam_name, data, cam_T, depth in zip(self.cam_names, all_data, all_preds['cam_T'], all_preds['depth']):

            # self.map_cameras[cam_name].apply_T(cam_T)

            # Visualize
            color_img = data[0]
            depth_img = vis_depth(depth)
            pose_img = self.map_viewer.get_view(self.map_cameras[cam_name])

            self.vis_images['{} color'.format(cam_name)] = color_img
            self.vis_images['{} depth'.format(cam_name)] = depth_img
            self.vis_images['{} pose'.format(cam_name)] = pose_img
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
            cam_id = int(cam_name.replace('cam', ''))
            calibs = {}
            calibs['K'] = self.camera_calibs[cam_id]['intrinsic']
            calibs['ext_T'] = self.camera_calibs[cam_id]['extrinsic']['imu-0']

            data = {}
            data[0] = observation[cam_name]
            data[-1] = self.last_observation[cam_name]
            data[0, 'calib'] = calibs
            data[-1, 'calib'] = calibs

            all_data.append(data)
        return all_data

    def show(self):
        for name, img in self.vis_images.items():
            if img.size[0] > 1000:
                img = img.resize((img.size[0] // 4, img.size[1] // 4), Image.ANTIALIAS)
            img = np.array(img)[:,:,::-1]
            cv.imshow('{}'.format(name), img)
        cv.waitKey(1)