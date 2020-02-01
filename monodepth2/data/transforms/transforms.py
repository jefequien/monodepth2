
import random
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        inputs = {}
        for t in self.transforms:
            data, inputs = t(data, inputs)
        return inputs

class PrepareAuxInputs(object):
    def __init__(self, aux_ids, scales, height, width):
        self.aux_ids = aux_ids
        self.scales = scales
        self.height = height
        self.width = width
    
    def __call__(self, data, inputs):
        for aux_id in self.aux_ids:
            if aux_id == 'gps':
                gps0 = data[0, 'gps'][:3]
                gps_1 = data[-1, 'gps'][:3]
                gps1 = data[1, 'gps'][:3]
                # Make gps delta smaller
                inputs['gps_delta', 1] = (gps1 - gps0) * 0.1
                inputs['gps_delta', -1] = (gps0 - gps_1) * 0.1

            elif aux_id == 'map_view':
                mv0 = data[0, aux_id]
                mv1 = data[1, aux_id]
                mv_1 = data[-1, aux_id]
                mv0 = F.resize(mv0, (self.height, self.width), interpolation=Image.ANTIALIAS)
                mv1 = F.resize(mv1, (self.height, self.width), interpolation=Image.ANTIALIAS)
                mv_1 = F.resize(mv_1, (self.height, self.width), interpolation=Image.ANTIALIAS)

                for s in self.scales:
                    r = 2 ** s
                    if s == 0:
                        inputs[aux_id, 0, s] = mv0
                        inputs[aux_id, 1, s] = mv1
                        inputs[aux_id, -1, s] = mv_1
                    else:
                        size = (self.height // r, self.width // r)
                        inputs[aux_id, 0, s] = F.resize(mv0, size, interpolation=Image.ANTIALIAS)
                        inputs[aux_id, 1 , s] = F.resize(mv1, size, interpolation=Image.ANTIALIAS)
                        inputs[aux_id, -1 , s] = F.resize(mv_1, size, interpolation=Image.ANTIALIAS)
            else:
                raise Exception('Auxilliary data id not recognized for transform: {}'.format(aux_id))
        return data, inputs

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
                K = v['K']
                for s in self.scales:
                    # Scale intrinsic
                    K_s = K.copy()
                    K_s[0, :] *= self.width // (2 ** s)
                    K_s[1, :] *= self.height // (2 ** s)

                    inputs["K", f_id, s] = torch.from_numpy(K_s)
                    inputs[("inv_K", f_id, s)] = torch.from_numpy(np.linalg.pinv(K_s))
        return data, inputs

class ToTensorInputs(object):
    def __call__(self, data, inputs):
        for k,v in inputs.items():
            if isinstance(v, Image.Image):
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
                data[f_id, 'color'] = F.resize(v, (self.height, self.width), interpolation=Image.ANTIALIAS)
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