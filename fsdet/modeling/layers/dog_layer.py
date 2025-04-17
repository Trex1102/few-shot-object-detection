import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

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
