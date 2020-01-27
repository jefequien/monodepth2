import os
import logging

from monodepth2.data.transforms import build_transforms

from monodepth2.networks import build_models
from monodepth2.networks.layers import *

logger = logging.getLogger('monodepth2.model')

class MonodepthModel(object):

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.scales = cfg.MODEL.SCALES

        self.frame_ids = cfg.INPUT.FRAME_IDS
        
        self.output_dir = cfg.OUTPUT_DIR

        self.models = build_models(cfg)
        self.to(self.device)

        self.transform = build_transforms(cfg, is_train=False)
    
    def predict(self, data):
        inputs = self.transform(data)
        outputs = {}

        # Inference
        with torch.no_grad():
            inputs = {k: v.unsqueeze(0) for k,v in inputs.items()} # Create a batch size of 1
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            outputs = self.process_batch(inputs)

        depth = outputs[('disp', 0)][0,0,:,:]

        preds = {}
        preds['depth'] = depth.cpu().detach().numpy()
        preds['T'] = outputs[("cam_T_cam", 0, -1)][0].cpu().detach().numpy()
        return preds

    def process_batch(self, inputs):

        depth_features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
        depth_outputs = self.models["depth_decoder"](depth_features)
        pose_outputs = self.predict_poses(inputs)

        outputs = {}
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

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def to(self, device):
        for m in self.models.values():
            m.to(device)
    
    def parameters_to_train(self):
        parameters_to_train = []
        for m in self.models.values():
            parameters_to_train += list(m.parameters())
        return parameters_to_train
    
    def load_model(self, save_folder):
        for model_name, model in self.models.items():
            logger.info("Loading {} weights...".format(model_name))
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            pretrained_dict = torch.load(save_path)

            # Filter layers
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    
    def save_model(self, save_folder):
        """Save model weights to disk
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        logger.info("Saving to {}".format(save_folder))

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)
