# fsdet/modeling/fusion/msdog_cbam_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAM(nn.Module):
    
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        # After concatenation, channels = in_channels * 2
        self.channel_attn = ChannelAttention(in_channels * 2, reduction)  # :contentReference[oaicite:8]{index=8}
        self.spatial_attn = SpatialAttention(spatial_kernel)             # :contentReference[oaicite:9]{index=9}
        # reduce back to in_channels
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    def forward(self, feat_raw, feat_dog):
        x = torch.cat([feat_raw, feat_dog], dim=1)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return self.conv1x1(x)
