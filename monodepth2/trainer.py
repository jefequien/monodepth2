
import os
import datetime
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import MSELoss
from tensorboardX import SummaryWriter

from monodepth2.model import MonodepthModel
from monodepth2.data import make_data_loader
from monodepth2.networks.layers import *

from monodepth2.utils import normalize_image

logger = logging.getLogger('monodepth2.trainer')

class Trainer(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE

        self.height = cfg.INPUT.HEIGHT
        self.width = cfg.INPUT.WIDTH
        self.scales = cfg.INPUT.SCALES
        self.frame_ids = cfg.INPUT.FRAME_IDS
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"

        self.num_epochs = cfg.SOLVER.NUM_EPOCHS
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.disparity_smoothness = cfg.SOLVER.DISPARITY_SMOOTHNESS
        self.min_depth = cfg.SOLVER.MIN_DEPTH
        self.max_depth = cfg.SOLVER.MAX_DEPTH

        self.epoch = 0
        self.step = 0

        self.output_dir = cfg.OUTPUT_DIR
        self.log_freq = cfg.SOLVER.LOG_FREQ
        self.val_freq = cfg.SOLVER.VAL_FREQ
        # Tensorboard writers
        now = datetime.datetime.now()
        self.writers = {}
        for mode in ["train", "valid"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.output_dir, "{} {}".format(mode, now)))

        # Model
        self.model = MonodepthModel(cfg)
        self.model.to(self.device)

        # Optimizer
        self.model_optimizer = optim.Adam(self.model.parameters_to_train(), cfg.SOLVER.BASE_LR)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, cfg.SOLVER.SCHEDULER_STEP_SIZE, cfg.SOLVER.SCHEDULER_GAMMA)

        # Data
        self.train_loader = make_data_loader(cfg, is_train=True)
        self.val_loader = make_data_loader(cfg, is_train=False)
        self.val_iter = iter(self.val_loader)
        logger.info("Train dataset size: {}".format(len(self.train_loader.dataset)))
        logger.info("Valid dataset size: {}".format(len(self.val_loader.dataset)))

        # Loss
        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.gps_loss = MSELoss()
        self.gps_loss.to(self.device)

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
        for p in self.model.parameters_to_train():
            p.requires_grad = False
        for p in self.model.parameters(['map_pose_encoder', 'map_pose_decoder']):
            p.requires_grad = True

        while self.epoch < self.num_epochs:
            logger.info("Epoch {}/{}  LR {}".format(self.epoch + 1, self.num_epochs, self.get_lr()))
            
            self.run_epoch()
            self.epoch += 1
            self.model_lr_scheduler.step()
            self.checkpoint()
    
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model.set_train()

        for _, inputs in enumerate(tqdm(self.train_loader)):
            inputs, outputs = self.model.process_batch(inputs)
            losses = self.compute_losses(inputs, outputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            self.step += 1
            if self.step % self.log_freq == 0:
                self.log_losses(losses, is_train=True)
            if self.step % self.val_freq == 0:
                self.validate()

    def validate(self):
        """Validating the model on a single minibatch and log progress
        """
        self.model.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            inputs, outputs = self.model.process_batch(inputs)
            losses = self.compute_losses(inputs, outputs)

            self.log_losses(losses, is_train=False)
            self.log_images(inputs, outputs, is_train=False)
            del inputs, outputs, losses
        
        self.model.set_train()
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        # Create warped images
        self.generate_images_pred(inputs, outputs)
        self.generate_map_pred(inputs, outputs)

        total_loss = 0
        for scale in self.scales:
            img_loss = self.compute_image_loss(inputs, outputs, scale)
            map_loss = self.compute_map_loss(inputs, outputs, scale)
            total_loss += img_loss + map_loss
            losses["loss/{}".format(scale)] = img_loss
            losses["loss/map{}".format(scale)] = map_loss
        total_loss /= len(self.scales)

        gps_loss = self.compute_gps_loss(inputs, outputs)
        losses["loss/gps"] = gps_loss

        losses["loss"] = total_loss + gps_loss
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
                    depth, inputs[("inv_K", frame_id, source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", frame_id, source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]
    

    def generate_map_pred(self, inputs, outputs):
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

            frame_id = 0
            T = outputs[("map_cam_T_cam", 0, frame_id)]

            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", frame_id, source_scale)])
            pix_coords = self.project_3d[source_scale](
                cam_points, inputs[("K", frame_id, source_scale)], T)

            outputs[("sample", frame_id, scale)] = pix_coords

            outputs[("map_view", frame_id, scale)] = F.grid_sample(
                inputs[("map_view", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            outputs[("map_view_identity", frame_id, scale)] = \
                inputs[("map_view", frame_id, source_scale)]

    def compute_image_loss(self, inputs, outputs, scale):
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
        return loss

    def compute_map_loss(self, inputs, outputs, scale):
        pred = outputs[("map_view", 0, scale)]
        target = inputs[("map_pred", 0, 0)]
        mask, idxs = torch.max(target, dim=1)

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1)

        loss = l1_loss * mask
        loss = loss.mean()
        return loss

    def compute_gps_loss(self, inputs, outputs):
        gps_loss = 0
        for frame_id in self.frame_ids[1:]:
            pred_trans = outputs[("translation", 0, frame_id)][:,0]
            targ_trans = inputs['gps_delta', frame_id]
            pred_norm = torch.norm(pred_trans, dim=2)
            targ_norm = torch.norm(targ_trans, dim=1, keepdim=True)
            gps_loss += self.gps_loss(pred_norm, targ_norm)
        return gps_loss

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def get_lr(self):
        for param_group in self.model_optimizer.param_groups:
            return param_group['lr']
    
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

        num_images = min(4, self.batch_size) # write a maxmimum of four images
        for j in range(num_images): 

            for s in self.scales:
                for frame_id in self.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data,
                        self.step
                    )
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, 
                            self.step
                        )

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), 
                    self.step
                )

                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], 
                    self.step
                )

                writer.add_image(
                    "map_view_{}/{}".format(s, j),
                    inputs[("map_view", 0, s)][j].data,
                    self.step
                )
                writer.add_image(
                    "map_pred_{}/{}".format(s, j),
                    inputs[("map_pred", 0, s)][j].data,
                    self.step
                )
                if s == 0:
                    writer.add_image(
                        "map_warp_{}/{}".format(s, j),
                        outputs[("map_view", 0, s)][j].data,
                        self.step
                    )
    
    def checkpoint(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.output_dir, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        logger.info("Saving to {}".format(save_folder))
        self.model.save_model(save_folder)

        # Save trainer state
        trainer_state = {
            'epoch': self.epoch,
            'step': self.step,
            'optimizer': self.model_optimizer.state_dict(),
            'scheduler': self.model_lr_scheduler.state_dict(),
        }
        save_path = os.path.join(save_folder, "{}.pth".format('trainer'))
        torch.save(trainer_state, save_path)

        # Symlink latest model for resuming
        latest_model_path = os.path.join(self.output_dir, "models", "latest_weights")
        if os.path.islink(latest_model_path):
            os.unlink(latest_model_path)
        os.symlink(os.path.basename(save_folder), latest_model_path)
    
    def load_checkpoint(self, load_optimizer=True):
        """Load model(s) from disk
        """
        save_folder = os.path.join(self.output_dir, "models", "latest_weights")
        assert os.path.isdir(save_folder), "Cannot find folder {}".format(save_folder)

        logger.info("Loading from {}".format(save_folder))
        self.model.load_model(save_folder)
        self.model.to(self.device)

        if load_optimizer:
            # Load trainer state
            save_path = os.path.join(save_folder, "{}.pth".format("trainer"))
            if os.path.isfile(save_path):
                logger.info("Loading trainer...")
                trainer_state = torch.load(save_path)
                self.epoch = trainer_state['epoch']
                self.step = trainer_state['step']
                self.model_lr_scheduler.load_state_dict(trainer_state['scheduler'])
                logger.info("Unresolved issue: Unable to load saved optimizer weights.")
                # self.model_optimizer.load_state_dict(trainer_state['optimizer'])
            else:
                logger.info("Could not load trainer")
            
