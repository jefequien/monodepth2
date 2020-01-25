
import os
import datetime
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from monodepth2.data import make_data_loader
from monodepth2.networks import build_models
from monodepth2.networks.layers import *

from monodepth2.utils import normalize_image


logger = logging.getLogger('monodepth2.trainer')

class Trainer:

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.scales = cfg.MODEL.SCALES

        self.num_epochs = cfg.SOLVER.NUM_EPOCHS
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.disparity_smoothness = cfg.SOLVER.DISPARITY_SMOOTHNESS
        self.min_depth = cfg.SOLVER.MIN_DEPTH
        self.max_depth = cfg.SOLVER.MAX_DEPTH

        self.height = cfg.INPUT.HEIGHT
        self.width = cfg.INPUT.WIDTH
        self.frame_ids = cfg.INPUT.FRAME_IDS
        self.epoch = 1
        self.step = 0

        self.output_dir = cfg.OUTPUT_DIR
        # Tensorboard writers
        now = datetime.datetime.now()
        self.writers = {}
        for mode in ["train", "valid"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.output_dir, "{} {}".format(mode, now)))

        # Models
        self.models = build_models(cfg)
        self.to(self.device)

        # Optimizer
        self.parameters_to_train = []
        for m in self.models.values():
            self.parameters_to_train += list(m.parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, cfg.SOLVER.BASE_LR)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, cfg.SOLVER.SCHEDULER_STEP_SIZE, 0.1)

        # Data
        self.train_loader = make_data_loader(cfg, is_train=True)
        self.val_loader = make_data_loader(cfg, is_train=False)
        logger.info("Train dataset size: {}".format(len(self.train_loader.dataset)))
        logger.info("Valid dataset size: {}".format(len(self.val_loader.dataset)))

        # Loss
        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)
    
    def train(self):
        while self.epoch <= self.num_epochs:
            logger.info("Epoch {}/{}".format(self.epoch, self.num_epochs))

            self.run_epoch()
            self.valid()
            self.save_model()
            self.epoch += 1
    
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        self.set_train()

        for _, inputs in enumerate(tqdm(self.train_loader)):
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            self.step += 1
            self.log_losses(losses, is_train=True)

    def valid(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        with torch.no_grad():
            for idx, inputs in enumerate(tqdm(self.val_loader)):
                outputs, losses = self.process_batch(inputs)

                self.log_losses(losses, is_train=False)
                if idx % 10 == 0:
                    self.log_images(inputs, outputs, is_train=False)

                del inputs, outputs, losses
    
    def process_batch(self, inputs):
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        depth_features = self.models["depth_encoder"](inputs["color_aug", 0, 0])
        depth_outputs = self.models["depth_decoder"](depth_features)
        pose_outputs = self.predict_poses(inputs)

        outputs = {}
        outputs.update(depth_outputs)
        outputs.update(pose_outputs)
        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses
    
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
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_loss = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= len(self.scales)
        losses["loss"] = total_loss
        return losses

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def log_losses(self, losses, is_train=True):
        """Write an event to the tensorboard events file
        """
        mode = "train" if is_train else "valid"
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
    
    def log_images(self, inputs, outputs, is_train=True):
        mode = "train" if is_train else "valid"
        writer = self.writers[mode]

        for j in range(min(4, self.batch_size)):  # write a maxmimum of four images
            for s in self.scales:
                for frame_id in self.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

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
    
    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.output_dir, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        logger.info("Saving to {}".format(save_folder))

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        # Save trainer state
        trainer_state = {
            'optimizer': self.model_optimizer.state_dict(),
            'epoch': self.epoch, 
            'step': self.step,
        }
        save_path = os.path.join(save_folder, "{}.pth".format('trainer'))
        torch.save(trainer_state, save_path)

        # Symlink latest model
        latest_model_path = os.path.join(self.output_dir, "models", "latest_weights")
        if os.path.islink(latest_model_path):
            os.unlink(latest_model_path)
        os.symlink(os.path.basename(save_folder), latest_model_path)
    
    def load_model(self):
        """Load model(s) from disk
        """
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
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

        # Load trainer state
        save_path = os.path.join(save_folder, "{}.pth".format("trainer"))
        if os.path.isfile(save_path):
            logger.info("Loading trainer...")
            trainer_state = torch.load(save_path)
            self.epoch = trainer_state['epoch'] + 1
            self.step = trainer_state['step'] + 1
            self.model_optimizer.load_state_dict(trainer_state['optimizer'])
        else:
            logger.info("Could not load trainer")
            
