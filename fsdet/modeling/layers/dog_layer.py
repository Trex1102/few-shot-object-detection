import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

import torch
from torch import nn
import torch.nn.functional as F

class DoGLayer2(nn.Module):
    """
    Difference-of-Gaussian layer. Computes DoG maps over a grayscale version of the input.

    Args:
        sigmas (list of float): increasing list of Gaussian sigma values.
    Returns:
        Tensor: Bx1xHxW DoG response (sum of DoG across scales).
    """
    def __init__(self, sigmas):
        super().__init__()
        assert isinstance(sigmas, (list, tuple)) and len(sigmas) >= 2, \
            "sigmas must be a list of at least two values"
        self.sigmas = sigmas
        # Precompute Gaussian kernels for each sigma
        self.kernels = nn.ModuleList([self._make_kernel(s) for s in sigmas])

    def _make_kernel(self, sigma):
        # Kernel radius = 3 * sigma, ensure at least 1
        radius = max(int(3 * sigma), 1)
        size = 2 * radius + 1
        # Create 1D Gaussian
        coords = torch.arange(size, dtype=torch.float32) - radius
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        # Outer product to get 2D kernel
        kernel2d = g[:, None] @ g[None, :]
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        conv = nn.Conv2d(
            1, 1, kernel_size=size, padding=radius, bias=False,
            groups=1
        )
        conv.weight.data.copy_(kernel2d)
        conv.weight.requires_grad = False
        return conv

    def forward(self, x):  # x: BxCxHxW
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            # RGB -> Luma
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            x_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            x_gray = x[:, :1]
        # Compute Gaussian-blurred images at each sigma
        blurred = [kernel(x_gray) for kernel in self.kernels]
        # Difference-of-Gaussian: sum of differences between successive scales
        dog = torch.zeros_like(blurred[0])
        for prev, curr in zip(blurred, blurred[1:]):
            dog = dog + (curr - prev)
        return dog




class DoGLayer(nn.Module):
    """
    Applies Difference-of-Gaussians filtering to a 3-channel image tensor.
    """

    def __init__(self, channels=3, kernel_size=5, sigma1=1.0, sigma2=2.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # Build Gaussian kernels
        self.register_buffer("gauss1", self._create_gaussian_kernel(sigma1))
        self.register_buffer("gauss2", self._create_gaussian_kernel(sigma2))

        # Depthwise convolution for Gaussian blur
        self.blur1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                               groups=channels, bias=False)
        self.blur2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                               groups=channels, bias=False)

        # Set kernels
        self._set_weights()

    def _create_gaussian_kernel(self, sigma):
        """
        Creates a 2D Gaussian kernel using the given sigma.
        """
        k = self.kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel

    def _set_weights(self):
        """
        Sets the depthwise conv weights to fixed Gaussian kernels.
        """
        weight1 = self.gauss1.expand(self.channels, 1, -1, -1).clone()
        weight2 = self.gauss2.expand(self.channels, 1, -1, -1).clone()
        self.blur1.weight.data.copy_(weight1)
        self.blur2.weight.data.copy_(weight2)
        self.blur1.weight.requires_grad = False
        self.blur2.weight.requires_grad = False

    def forward(self, x):
        """
        x: (B, C, H, W) image tensor in 3-channel RGB format.
        Returns: DoG image (B, C, H, W) with edge-enhanced features.
        """
        blur1 = self.blur1(x)
        blur2 = self.blur2(x)
        dog = blur1 - blur2
        return dog
