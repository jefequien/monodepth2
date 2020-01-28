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
        self.scales = cfg.MODEL.SCALES

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = {}
        for s in self.scales:
            r = 2 ** s
            self.resize[s] = transforms.Resize((self.height // r, self.width // r),
                                               interpolation=Image.ANTIALIAS)

    def __call__(self, data):
        """Returns a single training item from data.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color",     <frame_id>, <scale>)      for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K",         <frame_id>, <scale>)      for camera intrinsics,
            ("inv_K",     <frame_id>, <scale>)      for camera intrinsics inverted,
            ("ext_T"      <frame_id>, <scale>)      for camera extrinsics,

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        imgs, intrs, extrs = {}, {}, {}
        for frame_id, v in data.items():
            imgs[frame_id] = v[0]
            intrs[frame_id] = v[1]
            extrs[frame_id] = v[2]

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # Resize images
        for f, img in imgs.items():
            inputs[("color", f, -1)] = img
            inputs[("color_aug", f, -1)] = color_aug(img)
            for s in self.scales:
                inputs[('color', f, s)] = self.resize[s](inputs[('color', f, s-1)])
                inputs[('color_aug', f, s)] = self.resize[s](inputs[('color_aug', f, s-1)])

            del inputs[("color", f, -1)]
            del inputs[("color_aug", f, -1)]
        
        inputs = {k: self.to_tensor(v) for k,v in inputs.items()}

        # Intrinsics
        for f, K in intrs.items():
            for s in self.scales:
                K_s = K.copy()
                K_s[0, :] *= self.width // (2 ** s)
                K_s[1, :] *= self.height // (2 ** s)
                inputs[("K", f, s)] = torch.from_numpy(K_s)
                inputs[("inv_K", f, s)] = torch.from_numpy(np.linalg.pinv(K_s))

        return inputs

def build_transforms(cfg, is_train=True):
    transform = DataTransform(cfg, is_train=is_train)
    return transform
