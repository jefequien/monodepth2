import os
import pickle
import numpy as np
from tqdm import tqdm

from .synced_dataset import SyncedDataset
from .bag_reader import CameraBagReader
from monodepth2.utils.calibration_manager import CalibrationManager


class TSDataset(SyncedDataset):

    def __init__(self, bag_info, data_ids, transform=None):
        super(TSDataset, self).__init__(data_ids=data_ids, transform=transform)
        bag_name, map_name, begin, end = bag_info

        self.bag_reader = CameraBagReader(bag_info)
        self.load_dir = load_bag_to_disk(self.bag_reader)

        self.calib_manager = CalibrationManager(dataset=bag_name)
        self.camera_calibs = self.calib_manager.get_cameras()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def __len__(self):
        return self.bag_reader.__len__()
    
    def get_image(self, cam_name, idx, shift=0):
        fname = os.path.join(self.load_dir, '{}/{}.pkl'.format(idx+shift, cam_name))
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    def get_calibration(self, cam_name):
        cam_id = int(cam_name.split('cam')[1])
        intrinsic = self.camera_calibs[cam_id]['intrinsic']
        extrinsic = self.camera_calibs[cam_id]['extrinsic']['imu-0']
        distortion = self.camera_calibs[cam_id]['distortion'].squeeze()
        img_shape = self.camera_calibs[cam_id]['img_shape']

        calib = {}
        # calib['K'] = np.array(intrinsic, dtype=np.float32)
        calib['ext_T'] = np.array(extrinsic, dtype=np.float32)
        calib['K'] = self.K
        return calib


def load_bag_to_disk(bag_reader, reload=False):
    load_dir = "/tmp/tsdatasets/{}".format(str(bag_reader.bag_info))
    if not reload and os.path.isdir(load_dir):
        return load_dir

    print('Loading bag to disk... ', bag_reader.bag_info)
    for idx, data in enumerate(tqdm(bag_reader)):
        for k, v in data.items():
            fname = os.path.join(load_dir, '{}/{}.pkl'.format(idx, k))

            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
            
            with open(fname, 'wb') as f:
                pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
    return load_dir


