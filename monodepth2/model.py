import os
import time
import logging

from monodepth2.data.transforms import build_transforms

from monodepth2.networks import build_models
from monodepth2.networks.layers import *

logger = logging.getLogger('monodepth2.model')

class MonodepthModel(object):

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR

        self.min_depth = cfg.SOLVER.MIN_DEPTH
        self.max_depth = cfg.SOLVER.MAX_DEPTH

        self.models = build_models(cfg)
        self.transform = build_transforms(cfg, is_train=False)
    
    def predict(self, all_data):
        """Predict a list of data as a batch.
        """
        # Tranform and batch a list of data
        batch_inputs = {}
        for data in all_data:
            inputs = self.transform(data)
            for k,v in inputs.items():
                if k not in batch_inputs:
                    batch_inputs[k] = []
                batch_inputs[k].append(v)
        batch_inputs = {k: torch.stack(v) for k,v in batch_inputs.items()}

        outputs = {}
        with torch.no_grad():
            _, outputs = self.process_batch(batch_inputs)

        # Get depth
        scale = 0
        disp = outputs[("disp", scale)]
        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        depth = depth[:,0,:,:]

        # Get camera transform
        cam_T = outputs[('cam_T_cam', 0, -1)]
        drift_T = outputs[('map_cam_T_cam', 0, 0)]

        preds = {}
        preds['disp'] = disp.cpu().detach().numpy()
        preds['depth'] = depth.cpu().detach().numpy()
        preds['cam_T'] = cam_T.cpu().detach().numpy()
        preds['drift_T'] = drift_T.cpu().detach().numpy()
        return preds

    def process_batch(self, inputs):
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        depth_features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
        depth_outputs = self.models["depth_decoder"](depth_features)
        pose_outputs = self.predict_poses(inputs)
        map_pose_outputs = self.predict_map_poses(inputs)

        outputs = {}
        outputs.update(depth_outputs)
        outputs.update(pose_outputs)
        outputs.update(map_pose_outputs)
        return inputs, outputs
    
    def predict_poses(self, inputs):
        frame_ids = set([k[1] for k in inputs])
        
        pose_inputs = {f_i: inputs["color_aug", f_i, 0] for f_i in frame_ids}
        pose_outputs = {}
        for f_i in frame_ids:
            if f_i == 0:
                continue

            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                x = [pose_inputs[f_i], pose_inputs[0]]
            else:
                x = [pose_inputs[0], pose_inputs[f_i]]

            pose_features = [self.models["pose_encoder"](torch.cat(x, 1))]
            axisangle, translation = self.models["pose_decoder"](pose_features)
            pose_outputs[("axisangle", 0, f_i)] = axisangle
            pose_outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            pose_outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return pose_outputs

    def predict_map_poses(self, inputs):
        pose_outputs = {}

        # Map view then map pred
        x = [inputs["map_view", 0, 0], inputs["map_pred", 0, 0]]

        pose_features = [self.models["map_pose_encoder"](torch.cat(x, 1))]
        axisangle, translation = self.models["map_pose_decoder"](pose_features)
        pose_outputs[("axisangle", 0, 0)] = axisangle
        pose_outputs[("translation", 0, 0)] = translation

        # Invert the matrix if the frame id is negative
        pose_outputs[("map_cam_T_cam", 0, 0)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)

        return pose_outputs

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
            if os.path.isfile(save_path):
                model_dict = torch.load(save_path)
                model.load_state_dict(model_dict, strict=False)
            else:
                logger.info("Could not load {} weights".format(model_name))
    
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
