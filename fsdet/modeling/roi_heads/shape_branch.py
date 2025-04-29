import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler



class ShapeBranch(nn.Module):
    """
    Bounding Shape Mask Branch (BshapeNet):
    Predicts a boundary mask and returns pooled shape features.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.shape_pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, roi_feats_list):
        pooled_feats = []
        for feats in roi_feats_list:
            # feats: Ni x C x H x W
            mask_logits = self.mask_conv(feats)
            attn = torch.sigmoid(mask_logits)
            shaped = feats * attn
            vec = self.shape_pooler(shaped).flatten(1)  # Ni x 256
            pooled_feats.append(vec)
        return pooled_feats