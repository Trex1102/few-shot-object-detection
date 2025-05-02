import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

class ShapeBranch(nn.Module):
    """
    Shape mask -> vector.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,256,3,padding=1), nn.ReLU(inplace=True), nn.Conv2d(256,1,1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, roi_pooled):
        # list of Ni x C x H x W
        outputs = []
        for feats in roi_pooled:
            mask = torch.sigmoid(self.conv(feats))
            attended = feats * mask
            vec = self.avgpool(attended).flatten(1)
            outputs.append(vec)
        return outputs  # list of Ni x 256

class ShapeBranch2(nn.Module):
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

class ShapeBranchWithLoss(nn.Module):
    """
    Shape auto-encoder branch with reconstruction loss.
    """
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        # encoder conv
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        # embedding
        self.fc_embed = nn.Linear(256, embedding_dim)
        # decoder
        self.fc_decode = nn.Linear(embedding_dim, 256)
        self.dec_conv = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(256,1,1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, roi_pooled, gt_masks=None):
        outputs, aux_losses = [], {}
        for idx, feats in enumerate(roi_pooled):
            x = self.enc_conv(feats)
            vec = self.avgpool(x).flatten(1)  # Ni x 256
            embed = self.fc_embed(vec)        # Ni x emb
            # decoder reconstruct mask logits
            decoded = self.fc_decode(embed).view(-1,256,1,1)
            mask_logits = self.dec_conv(decoded)  # Ni x 1 x 1 x 1
            outputs.append(embed)
            if self.training:
                # upsample gt_masks[idx] to 1x1 and compute BCE
                m = F.adaptive_avg_pool2d(gt_masks[idx].unsqueeze(1).float(), (1,1)).flatten(1)
                loss = F.binary_cross_entropy_with_logits(mask_logits.flatten(1), m)
                aux_losses[f"shape_loss_{idx}"] = loss
        return outputs, aux_losses
