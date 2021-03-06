import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from .synced_dataset import SyncedDataset
from .bag_reader import CameraBagReader
from monodepth2.utils.calibration_manager import CalibrationManager


class TSDataset(SyncedDataset):

    def __init__(self, bag_info, data_ids, transform=None, gps_noise=None):
        super(TSDataset, self).__init__(data_ids=data_ids, transform=transform)
        bag_name = bag_info[0]

        self.bag_reader = CameraBagReader(bag_info, gps_noise)
        self.load_dir = load_bag_to_disk(self.bag_reader, '{} {}'.format(str(bag_info), gps_noise))

        self.calib_manager = CalibrationManager(dataset=bag_name)
        self.camera_calibs = self.calib_manager.get_cameras()

    def __len__(self):
        """Do not count first and last frames from bag reader.
        """
        return self.bag_reader.__len__() - 2
    
    def get_image(self, cam_name, idx, shift=0):
        """Shift by one."""
        bag_idx = idx + 1 + shift
        fname = os.path.join(self.load_dir, '{}/{}.jpg'.format(bag_idx, cam_name))
        return Image.open(fname)
    
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

    def get_gps(self, idx, shift=0):
        """Shift by one."""
        bag_idx = idx + 1 + shift
        fname = os.path.join(self.load_dir, '{}/{}.pkl'.format(bag_idx, 'gps_data'))
        with open(fname, 'rb') as f:
            gps_data = pickle.load(f)
            return np.array(gps_data, dtype=np.float32)
    
    def get_map_view(self, idx, shift=0):
        cam_name = 'cam1'
        bag_idx = idx + 1 + shift
        fname = os.path.join(self.load_dir, '{}/map_view/{}.jpg'.format(bag_idx, cam_name))
        return Image.open(fname)
    
    def get_map_pred(self, idx, shift=0):
        cam_name = 'cam1'
        bag_idx = idx + 1 + shift
        fname = os.path.join(self.load_dir, '{}/map_pred/{}.jpg'.format(bag_idx, cam_name))
        return Image.open(fname)


def load_bag_to_disk(bag_reader, name, reload=False):
    load_dir = "/home/jeffrey.hu/tmp/tsdataset/{}".format(name)
    if os.path.isdir(load_dir) and not reload:
        return load_dir

    print('Loading bag to disk... ', bag_reader.bag_info)
    for idx, data in enumerate(tqdm(bag_reader)):
        for k, v in data.items():
            if isinstance(v, Image.Image):
                fname = os.path.join(load_dir, '{}/{}.jpg'.format(idx, k))
                if not os.path.isdir(os.path.dirname(fname)):
                    os.makedirs(os.path.dirname(fname))
                if not os.path.isfile(fname) or reload:
                    v.save(fname, 'JPEG')
            
            else:
                fname = os.path.join(load_dir, '{}/{}.pkl'.format(idx, k))
                if not os.path.isdir(os.path.dirname(fname)):
                    os.makedirs(os.path.dirname(fname))

                with open(fname, 'wb') as f:
                    pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
    return load_dir
