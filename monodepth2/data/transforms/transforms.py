
import random
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

from ..maps.map_utils import scale_cam_intrinsic

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        inputs = {}
        for t in self.transforms:
            data, inputs = t(data, inputs)
        return inputs

class PrepareImageInputs(object):
    def __init__(self, scales, height, width):
        self.scales = scales
        self.height = height
        self.width = width
    
    def __call__(self, data, inputs):
        for k, v in data.items():
            f_id, data_type = k
            if data_type == 'color' or data_type == 'color_aug':
                for s in self.scales:
                    r = 2 ** s
                    if s == 0:
                        inputs[data_type, f_id, s] = v
                    else:
                        img = inputs[data_type, f_id, s-1]
                        size = (self.height // r, self.width // r)
                        inputs[data_type, f_id, s] = F.resize(img, size, interpolation=Image.ANTIALIAS)
        return data, inputs

class PrepareCalibInputs(object):
    def __init__(self, scales, height, width):
        self.scales = scales
        self.height = height
        self.width = width

    def __call__(self, data, inputs):
        for k, v in data.items():
            f_id, data_type = k
            if data_type == 'calib':
                # Scale intrinsic to input size
                src_size = v['img_shape']
                dst_size = self.width, self.height
                K = scale_cam_intrinsic(v['K'], src_size, dst_size)

                for s in self.scales:
                    K_s = K.copy()
                    K_s[0, :] *= self.width // (2 ** s)
                    K_s[1, :] *= self.height // (2 ** s)

                    inputs["K", f_id, s] = torch.from_numpy(K_s)
                    inputs[("inv_K", f_id, s)] = torch.from_numpy(np.linalg.pinv(K_s))
        return data, inputs

class ToTensorInputs(object):
    def __call__(self, data, inputs):
        for k,v in inputs.items():
            data_type, f_id, s = k
            if data_type == 'color' or data_type == 'color_aug':
                inputs[k] = F.to_tensor(v)
        return data, inputs

class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, data, inputs):
        for k, v in data.items():
            f_id, data_type = k
            if data_type == 'color':
                data[f_id, 'color'] = F.resize(v, (self.height, self.width))
        return data, inputs

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data, inputs):
        do_flip = random.random() < self.prob
        if do_flip:
            for k,v in data.items():
                f_id, data_type = k
                if data_type == 'color':
                    data[f_id, 'color'] = F.hflip(v)
        return data, inputs

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, data, inputs):
        jittered = {}
        for k,v in data.items():
            f_id, data_type = k
            if data_type == 'color':
                jittered[f_id, 'color_aug'] = self.color_jitter(v)
        data.update(jittered)
        return data, inputs