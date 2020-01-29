import os
import numpy as np
from PIL import Image

from .synced_dataset import SyncedDataset

class KITTIDataset(SyncedDataset):

    def __init__(self, root, fpath, data_ids, transform=None):
        super(KITTIDataset, self).__init__(data_ids=data_ids, transform=transform)
        self.root = root
        self.filenames = readlines(fpath)
        self.img_ext = '.jpg'
        self.cam_map = {
            'cam1': '2', # l
            'cam2': '3', # r
        }
        self.T = None

    def __len__(self):
        return len(self.filenames)

    def get_image(self, cam_name, index, shift=0):
        """Returns image, intrinsic, extrinsic
        """
        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1]) + shift
        # side = int(line[2])

        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.root, 
            folder,
            "image_0{}/data".format(self.cam_map[cam_name]), 
            f_str
        )

        img = pil_loader(image_path)
        return img
    
    def get_calibration(self, cam_name):
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        calib = {}
        calib['K'] = K
        calib['ext_T'] = None
        calib['img_shape'] = np.array([192, 640], dtype=np.float32)
        return calib

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines