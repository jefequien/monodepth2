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

    def __len__(self):
        """Do not count first and last frames from bag reader.
        """
        return self.bag_reader.__len__() - 2
    
    def get_image(self, cam_name, idx, shift=0):
        """Shift by one."""
        bag_idx = idx + 1 + shift
        fname = os.path.join(self.load_dir, '{}/{}.pkl'.format(bag_idx, cam_name))
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    def get_calibration(self, cam_name):
        cam_id = int(cam_name.replace('cam', ''))
        intrinsic = self.camera_calibs[cam_id]['intrinsic']
        extrinsic = self.camera_calibs[cam_id]['extrinsic']['imu-0']
        distortion = self.camera_calibs[cam_id]['distortion'].squeeze()
        img_shape = self.camera_calibs[cam_id]['img_shape']

        calib = {}
        calib['K'] = np.array(intrinsic, dtype=np.float32)
        calib['ext_T'] = np.array(extrinsic, dtype=np.float32)
        return calib


def load_bag_to_disk(bag_reader, reload=False):
    load_dir = "/tmp/tsdatasets/{}".format(str(bag_reader.bag_info))

    print('Loading bag to disk... ', bag_reader.bag_info)
    for idx, data in enumerate(tqdm(bag_reader)):
        for k, v in data.items():
            fname = os.path.join(load_dir, '{}/{}.pkl'.format(idx, k))
            if os.path.isfile(fname) and not reload:
                continue

            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
            with open(fname, 'wb') as f:
                pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
    return load_dir
