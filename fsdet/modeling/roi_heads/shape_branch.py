import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

class ShapeBranchBottleneck(nn.Module):
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

class ShapeBranchWithLoss(nn.Module):
    """
    Shape auto-encoder branch with reconstruction loss.
    Safely skips mask loss if no gt_masks provided.
    """
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_embed = nn.Linear(256, embedding_dim)
        self.fc_decode = nn.Linear(embedding_dim, 256)
        self.dec_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, roi_pooled, gt_masks=None):
        """
        Args:
            roi_pooled (list[Tensor]): per-image lists of Ni×C×H×W
            gt_masks (list[Tensor] or None): per-image Ni×H×W masks, or None
        Returns:
            outputs: list of Ni×embeddings
            aux_losses: dict of shape losses (empty if no masks)
        """
        outputs = []
        aux_losses = {}
        for idx, feats in enumerate(roi_pooled):
            feats = feats.contiguous()
            x = self.enc_conv(feats)                    # Ni×256×H×W
            vec = self.avgpool(x).flatten(1)            # Ni×256
            embed = self.fc_embed(vec)                  # Ni×emb_dim
            outputs.append(embed)

            # only compute mask loss if training AND a mask tensor exists
            if self.training and gt_masks is not None:
                # guard against missing or empty masks
                if idx < len(gt_masks) and gt_masks[idx] is not None:
                    mask = gt_masks[idx].unsqueeze(1).float()  # Ni×1×H×W
                    # downsample mask to 1×1
                    down = F.adaptive_avg_pool2d(mask, (1, 1)).flatten(1)  # Ni×1
                    decoded = self.fc_decode(embed).view(-1, 256, 1, 1)
                    mask_logits = self.dec_conv(decoded).flatten(1)        # Ni×1
                    loss = F.binary_cross_entropy_with_logits(mask_logits, down)
                    aux_losses[f"shape_loss_{idx}"] = loss
                else:
                    # no gt_masks for this image → zero loss
                    aux_losses[f"shape_loss_{idx}"] = vec.new_tensor(0.0, requires_grad=True)

        return outputs, aux_losses



class ShapeBranchSE(nn.Module):
    """
    Shape auto-encoder branch with reconstruction loss.
    """
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.fc_embed = nn.Linear(256, embedding_dim)
        self.fc_decode = nn.Linear(embedding_dim, 256)
        self.dec = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(256,1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, roi_pooled, gt_masks=None):
        outputs, aux_losses = [], {}
        for idx, feats in enumerate(roi_pooled):
            x = self.enc(feats)
            vec = self.avgpool(x).flatten(1)
            emb = self.fc_embed(vec)
            decoded = self.dec(self.fc_decode(emb).view(-1,256,1,1))
            outputs.append(emb)
            if self.training and gt_masks is not None:
                m = F.adaptive_avg_pool2d(gt_masks[idx].unsqueeze(1).float(), (1,1)).flatten(1)
                aux_losses[f"shape_loss_{idx}"] = F.binary_cross_entropy_with_logits(decoded.flatten(1), m)
        return outputs, aux_losses
