import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler




class ContextBranch(nn.Module):
    """
    RoI Context Mining (Auto-Context R-CNN):
    Mines the 8 surrounding cells in a 3x3 grid around each RoI.
    """
    def __init__(self, feature_strides, output_size=7, sampling_ratio=2):
        super().__init__()
        # Pooler over feature maps to extract context crops
        self.context_pooler = ROIPooler(
            output_size=(output_size, output_size),
            scales=[1.0 / s for s in feature_strides],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )
        # simple transform to fuse context features
        self.context_fusion = nn.Sequential(
            nn.Conv2d(256 * 8, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, proposals):
        # proposals: list of Instances per image
        # build 8 offset boxes per proposal (3x3 grid minus center)
        all_boxes = []
        for props in proposals:
            boxes = props.proposal_boxes.tensor  # N x 4
            w = (boxes[:, 2] - boxes[:, 0]) / 3.0
            h = (boxes[:, 3] - boxes[:, 1]) / 3.0
            centers = boxes.reshape(-1, 1, 4)
            # generate grid offsets
            offsets = []
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    dx = j * w
                    dy = i * h
                    offsets.append(
                        torch.stack([
                            boxes[:, 0] + dx,
                            boxes[:, 1] + dy,
                            boxes[:, 0] + dx + w,
                            boxes[:, 1] + dy + h,
                        ], dim=1)
                    )
            # stack 8 x N x 4 -> (8N) x 4
            context_boxes = torch.cat(offsets, dim=0)
            # record image index
            ids = torch.arange(len(boxes), device=boxes.device)
            ids = ids.repeat(8)
            all_boxes.append({"boxes": context_boxes, "ids": ids})
        # pool context features
        pool_inputs = []
        for i, feats in enumerate(features):
            pool_inputs.append(feats)
        context_feats = self.context_pooler(
            pool_inputs,
            [b['boxes'] for b in all_boxes],
            [b['ids'] for b in all_boxes]
        )  # (sum 8N) x C x S x S
        # reshape and fuse per proposal
        fused = []
        ptr = 0
        for props in proposals:
            N = len(props)
            chunk = context_feats[ptr:ptr + 8 * N]
            ptr += 8 * N
            # reshape to N x (8C) x S x S
            chunk = chunk.view(N, -1, chunk.size(-2), chunk.size(-1))
            fused.append(self.context_fusion(chunk))
        return fused