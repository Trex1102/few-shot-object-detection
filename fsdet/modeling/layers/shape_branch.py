
import torch
from torch import nn
import torch.nn.functional as F



# ------------------ Shape Branch Modules ------------------
class ShapeFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1)
            for in_c in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, inputs):
        # inputs: list of FPN feature maps [P2...PN]
        laters = [l(x) for l, x in zip(self.lateral_convs, inputs)]
        # top-down merge with exact size matching
        for i in range(len(laters)-1, 0, -1):
            target_size = laters[i-1].shape[-2:]
            laters[i-1] = laters[i-1] + F.interpolate(
                laters[i], size=target_size, mode="nearest"
            )
        outs = [out(l) for out, l in zip(self.output_convs, laters)]
        return outs