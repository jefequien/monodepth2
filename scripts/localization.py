import os
import time
import numpy as np
import cv2 as cv
from PIL import Image
import time

from monodepth2.model import MonodepthModel
from monodepth2.data.maps.map_viewer import MapViewer, MapCamera
from monodepth2.utils.calibration_manager import CalibrationManager
from monodepth2.utils.visualize import vis_depth


class LocalizationModel:

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.vis_images = {}
        self.recorders = {}

        # Model
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
        self.model = MonodepthModel(cfg)
        self.model.load_model(save_folder)
        self.model.to(self.device)
        self.model.set_eval()

        # Localization params
        self.position = None
        self.cam_Ts = {}
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
            self.map_view = self.map_viewer.get_view(self.map_cameras[cam_name])

        self.last_observation = observation
        self.initialized = True

    def step(self, observation):
        if not self.initialized:
            self.initialize(observation)

        t0 = time.time()
        all_data = self.prepare_data(observation)
        t1 = time.time()
        all_preds = self.model.predict(all_data)
        t2 = time.time()
        print("pre: {}, inf: {}".format(t1-t0, t2-t1))
        # Scale predictions by 10
        for cam_name, cam_T, drift_T, depth in zip(self.cam_names, all_preds['cam_T'], all_preds['drift_T'], all_preds['depth']):
            cam_T[:3, 3] *= 10
            drift_T[:3, 3] *= 10
            depth *= 10

        # Move cameras
        for cam_name, cam_T, drift_T, depth in zip(self.cam_names, all_preds['cam_T'], all_preds['drift_T'], all_preds['depth']):
            img = observation['cam1']
            map_pred = observation['map_pred/{}'.format(cam_name)]

            self.map_cameras[cam_name].apply_T(cam_T)
            if self.num_steps % 1 == 0:
                self.map_cameras[cam_name].set_position(observation['gps_data'])

            unaligned_view = self.map_viewer.get_view(self.map_cameras[cam_name])

            # Visualize
            depth_img = vis_depth(depth)
            depth_vis = Image.new('RGB', (512*2, 288))
            depth_vis.paste(img, (0,0))
            depth_vis.paste(depth_img, (512,0))
            self.vis_images['{} depth'.format(cam_name)] = depth_vis

            pred_vis = img.copy()
            pred_vis.paste(map_pred, (0,0), map_pred.convert('L'))
            unaligned_vis = img.copy()
            unaligned_vis.paste(unaligned_view, (0,0), unaligned_view.convert('L'))
            self.vis_images['{} color'.format(cam_name)] = img
            self.vis_images['{} map_pred'.format(cam_name)] = pred_vis
            self.vis_images['{} unaligned'.format(cam_name)] = unaligned_vis

            # self.vis_images['{} aligned'.format(cam_name)] = aligned_vis
            # self.vis_images['{} map_depth_img'.format(cam_name)] = map_depth_img

        # Drift
        self.map_view = self.map_viewer.get_view(self.map_cameras[cam_name])

        t0 = time.time()
        all_data = self.prepare_data(observation)
        t1 = time.time()
        all_preds = self.model.predict(all_data)
        t2 = time.time()
        print("pre: {}, inf: {}".format(t1-t0, t2-t1))
        # Scale predictions by 10
        for cam_name, cam_T, drift_T, depth in zip(self.cam_names, all_preds['cam_T'], all_preds['drift_T'], all_preds['depth']):
            cam_T[:3, 3] *= 10
            drift_T[:3, 3] *= 10
            depth *= 10

        for cam_name, cam_T, drift_T, depth in zip(self.cam_names, all_preds['cam_T'], all_preds['drift_T'], all_preds['depth']):
            img = observation['cam1']
            
            self.map_cameras[cam_name].apply_T(drift_T)
            aligned_view = self.map_viewer.get_view(self.map_cameras[cam_name])

            # Visualize
            aligned_vis = img.copy()
            aligned_vis.paste(aligned_view, (0,0), aligned_view.convert('L'))
            self.vis_images['{} aligned'.format(cam_name)] = aligned_vis
        

        # new_im = Image.new('RGB', (512*3, 288))
        # new_im.paste(unaligned_vis, (0,0))
        # new_im.paste(aligned_vis, (512,0))
        # new_im.paste(pred_vis, (512*2,0))
        # self.vis_images['{} merge'.format(cam_name)] = new_im
        
        self.show()
        self.record_video()
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
            map_pred = observation['map_pred/{}'.format(cam_name)]

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
    
    def record_video(self):
        video_dir = os.path.join(self.output_dir, 'videos')
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)

        for name, img in self.vis_images.items():
            if name not in self.recorders:
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                fname = os.path.join(video_dir, '{}.avi'.format(name))
                self.recorders[name] = cv.VideoWriter(fname, fourcc, 20.0, (img.size[0], img.size[1]))
            recorder = self.recorders[name]

            img = np.array(img)
            if np.ndim(img) == 3:
                img = img[:,:,::-1]
            
            recorder.write(img)
        
    def close(self):
        for recorder in self.recorders.values():
            recorder.release()

    # def save_video(self):

    #     cap = cv2.VideoCapture(0)

    #     # Define the codec and create VideoWriter object
    #     #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    #     #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    #     out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

    #     while(cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret==True:
    #             frame = cv2.flip(frame,0)

    #             # write the flipped frame
    #             out.write(frame)

    #             cv2.imshow('frame',frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #         else:
    #             break

    #     # Release everything if job is finished
    #     cap.release()
    #     out.release()
    #     cv2.destroyAllWindows()