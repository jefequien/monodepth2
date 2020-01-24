import os

import torch

from monodepth2.networks import build_models
from monodepth2.networks.layers import *
from monodepth2.data.transforms import build_transforms

class LocalizationModel:

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE

        self.frame_ids = cfg.INPUT.FRAME_IDS


        self.transform = build_transforms(cfg, is_train=False)

        # Models
        self.models = build_models(cfg)
        self.to(self.device)
        self.set_eval()
    

    def step(self, data):
        inputs = self.transform(data)
        
        depth_features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
        depth_outputs = self.models["depth_decoder"](depth_features)
        pose_outputs = self.predict_poses(inputs)
        print(pose_outputs)
    
    def predict_poses(self, inputs):
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}
        for f_i in self.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                axisangle, translation = self.models["pose_decoder"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def to(self, device):
        for m in self.models.values():
            m.to(device)

    def load_models(self):
        pass