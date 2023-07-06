import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PoseLoss(nn.Module):
    def __init__(self, image_size=(800,1200), sigma=5, downscale=16):
        super(PoseLoss, self).__init__()
        self.image_size = image_size

        self.heatmap_height = self.image_size[0] // downscale
        self.heatmap_width = self.image_size[1] // downscale

        self.sigma = sigma
        self.downscale = downscale

    def _generate_maps(self, target):
        """
        Generates a confidence map for a given target.

        Args:
            target (torch.Tensor): The ground-truth target tensor of shape
                                   (batch size, num frames, num joints, num dimensions).

        Returns:
            torch.Tensor: A tensor of shape (batch size, num frames, num joints, height, width)
                          representing the confidence maps for each joint.
        """
        batch_size, num_frames, num_joints, num_dims = target.shape

        x = target[..., 0]  # x-coordinates
        y = target[..., 1]  # y-coordinates

        # Generate a 2D Gaussian point at each location (x,y)
        xx = torch.arange(self.heatmap_width, device=target.device).unsqueeze(0).unsqueeze(0)
        yy = torch.arange(self.heatmap_height, device=target.device).unsqueeze(0).unsqueeze(-1)
        xx = xx.expand(batch_size, num_frames, num_joints, -1, -1)
        yy = yy.expand(batch_size, num_frames, num_joints, -1, -1)
        mu_x = (x / self.image_size[1] * self.heatmap_width).unsqueeze(-1).unsqueeze(-1)
        mu_y = (y / self.image_size[0] * self.heatmap_height).unsqueeze(-1).unsqueeze(-1)
        sigma = self.sigma
        tmp_size = sigma * 3
        g = torch.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * sigma ** 2))
        g = g / (2 * math.pi * sigma ** 2)

        # Convert the generated Gaussians to heatmaps
        maps = g.reshape(batch_size, num_frames, num_joints, -1).reshape(batch_size, num_frames, num_joints,
                                                                         self.heatmap_height, self.heatmap_width)

        return maps

    def forward(self, pred, target, mask=None):
        """
        Compute the mean squared error (MSE) loss between predicted and target human pose joint coordinates.

        :param pred: A tensor of predicted joint coordinates with dimensions (batch size, number of frames, number of joints, number of dimensions).
        :param target: A tensor of target joint coordinates with dimensions (batch size, number of frames, number of joints, number of dimensions).
        :param mask: A tensor of binary values to indicate which joints should be included in the loss calculation.
        :return: The mean squared error (MSE) loss between predicted and target joint coordinates.
        """
        target_heatmaps = self._generate_maps(target)
        pred_heatmaps = self._generate_maps(pred)

        if mask is not None:
            # Expand mask tensor to match shape of target_heatmaps
            mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(target_heatmaps).type_as(target_heatmaps)
            target_heatmaps = target_heatmaps * mask
            pred_heatmaps = pred_heatmaps * mask

        mse_loss = F.mse_loss(pred_heatmaps, target_heatmaps)

        return mse_loss


    def _generate_single_map(self, x, y):
        """
           Generates a 2D Gaussian point at location x,y in tensor t.

           x should be in range (0, image width)
           y should be in range (0, image height)

           sigma is the standard deviation of the generated 2D Gaussian.
           """
        t = torch.zeros(self.heatmap_height, self.heatmap_width)
        h, w = self.image_size
        sigma = self.sigma

        # Heatmap pixel per output pixel
        mu_x = int(x/w * self.heatmap_width)
        mu_y = int(y/h * self.heatmap_height)

        tmp_size = sigma * 3

        # Top-left
        x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

        # Bottom right
        x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
        if x1 >= self.heatmap_width or y1 >= self.heatmap_height or x2 < 0 or y2 < 0:
            return t

        size = 2 * tmp_size + 1
        tx = np.arange(0, size, 1, np.float32)
        ty = tx[:, np.newaxis]
        x0 = y0 = size // 2

        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))

        # Determine the bounds of the source gaussian
        g_x_min, g_x_max = max(0, -x1), min(x2, self.heatmap_width) - x1
        g_y_min, g_y_max = max(0, -y1), min(y2, self.heatmap_height) - y1

        # Image range
        img_x_min, img_x_max = max(0, x1), min(x2, self.heatmap_width)
        img_y_min, img_y_max = max(0, y1), min(y2, self.heatmap_height)

        t[img_y_min:img_y_max, img_x_min:img_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]

        return t




