import os
import logging
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from monodepth2.config import cfg
from monodepth2.networks import build_models
from monodepth2.networks.layers import *
from monodepth2.data.transforms import build_transforms

from monodepth2.utils import normalize_image

logger = logging.getLogger('localization_model')


class LocalizationModel:

    def __init__(self, config_file):
        cfg.merge_from_file(config_file)

        self.height = cfg.INPUT.HEIGHT
        self.width = cfg.INPUT.WIDTH
        self.frame_ids = cfg.INPUT.FRAME_IDS

        self.device = cfg.MODEL.DEVICE
        self.transform = build_transforms(cfg, is_train=False)

        # Models
        self.models = build_models(cfg)
        self.to(self.device)
        self.set_eval()

        # Localization params
        self.tsmap = None
        self.initialized = False
    
    def setup(self, tsmap, bag_name):
        self.tsmap = tsmap
    
    def initialize(self, obs):
        self.initialized = True

    def step(self, obs):
        data = {}
        data[0] = obs['cam1']
        data[1] = obs['cam1']
        data[-1] = obs['cam1']

        inputs = self.transform(data)
        outputs = self.predict(inputs)
    
        # Get largest scale
        disp = outputs[("disp", 0)]
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        disp = scaled_disp.squeeze().cpu().numpy()
        depth_img = self.vis_depth(disp)
        depth_img.show()
        raise

        # for k in pose_outputs:
        #     pose_output = pose_outputs[k][0].to('cpu')
        #     print(k, pose_output)
        return None
    
    def predict(self, inputs):
        outputs = {}
        with torch.no_grad():
            inputs = {k: v.unsqueeze(0) for k,v in inputs.items()} 
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            depth_features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
            depth_outputs = self.models["depth_decoder"](depth_features)

            pose_outputs = self.predict_poses(inputs)

            outputs.update(depth_outputs)
            outputs.update(pose_outputs)
        return outputs
    
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
    
    def vis_depth(self, depth):
        # Saving colormapped depth image
        vmax = np.percentile(depth, 95)
        normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        return im


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def to(self, device):
        for m in self.models.values():
            m.to(device)

    def load_models(self, save_folder):
        assert os.path.isdir(save_folder), "Cannot find folder {}".format(save_folder)

        logger.info("Loading from {}".format(save_folder))
        for model_name, model in self.models.items():
            logger.info("Loading {} weights...".format(model_name))
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            pretrained_dict = torch.load(save_path)

            # Filter layers
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        