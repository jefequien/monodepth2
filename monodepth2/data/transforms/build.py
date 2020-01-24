import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


class DataTransform(object):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

        self.height = cfg.INPUT.HEIGHT
        self.width = cfg.INPUT.WIDTH
        self.frame_ids = cfg.INPUT.FRAME_IDS
        self.scales = cfg.MODEL.SCALES

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.interp = Image.ANTIALIAS
        self.to_tensor = transforms.ToTensor()
        self.resize = {}
        for i in self.scales:
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def __call__(self, data):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Raw images
        for i in self.frame_ids:
            inputs[("color", i, -1)] = data[i]

        # Intrinsics
        for scale in self.scales:
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        
        self.preprocess(inputs, color_aug)

        # Remove raw images
        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in self.scales:
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

def build_transforms(cfg, is_train=True):
    transform = DataTransform(cfg, is_train=is_train)
    return transform
