import numpy as np

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from monodepth2.data.transforms import build_transforms
from monodepth2.networks.layers import *

class DriftComputer():

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE

        self.height = cfg.INPUT.HEIGHT
        self.width = cfg.INPUT.WIDTH
        self.scales = [0]

        self.batch_size = 1
        self.disparity_smoothness = cfg.SOLVER.DISPARITY_SMOOTHNESS
        self.min_depth = cfg.SOLVER.MIN_DEPTH
        self.max_depth = cfg.SOLVER.MAX_DEPTH

        self.frame_ids = [0,-1]

        self.to_tensor = ToTensor()
        self.dx = torch.eye(4, requires_grad=True)

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

    def compute_drift(self, source, target, inputs, outputs):
        x = np.eye(4).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)
        x = torch.matmul(self.dx, x)

        inputs[("color", -1, 0)] = self.to_tensor(source).unsqueeze(0)
        inputs[("color", 0, 0)] = self.to_tensor(target).unsqueeze(0)
        # inputs[('inv_K', -1, 0)] = torch.from_numpy(np.linalg.pinv(K_s)).unsqueeze(1)
        # outputs[("disp", 0)] = torch.from_numpy(disp).unsqueeze(1)
        outputs[("cam_T_cam", 0, -1)] = x
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        outputs = {k:v.to(self.device) for k,v in outputs.items()}

        self.generate_images_pred(inputs, outputs)

        total_loss = 0
        for scale in self.scales:
            img_loss = self.compute_image_loss(inputs, outputs, scale)
            total_loss += img_loss
        total_loss /= len(self.scales)

        total_loss.backward()

        return self.dx.grad.cpu().detach().numpy()

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

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
