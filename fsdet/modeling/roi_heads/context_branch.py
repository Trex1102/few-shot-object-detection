import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
import torch.nn.functional as F


class ContextBranch(nn.Module):
    """
    RoI Context Mining (Auto-Context R-CNN):
    Mines the 8 surrounding cells in a 3x3 grid around each RoI.
    """
    def __init__(self, feature_strides, output_size=7, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.context_pooler = ROIPooler(
            output_size=(output_size, output_size),
            scales=[1.0 / s for s in feature_strides],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )
        self.context_fusion = nn.Sequential(
            nn.Conv2d(256 * 8, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, proposals):
        all_boxes = []
        for props in proposals:
            boxes = props.proposal_boxes.tensor  # N x 4
            w = (boxes[:, 2] - boxes[:, 0]) / 3.0
            h = (boxes[:, 3] - boxes[:, 1]) / 3.0
            offsets = []
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    dx = j * w
                    dy = i * h
                    offsets.append(torch.stack([
                        boxes[:, 0] + dx,
                        boxes[:, 1] + dy,
                        boxes[:, 0] + dx + w,
                        boxes[:, 1] + dy + h,
                    ], dim=1))
            context_boxes = torch.cat(offsets, dim=0)  # (8N) x 4
            all_boxes.append(Boxes(context_boxes))
        context_feats = self.context_pooler(list(features), all_boxes)
        fused = []
        ptr = 0
        for props in proposals:
            N = len(props)
            chunk = context_feats[ptr:ptr + 8 * N]
            ptr += 8 * N
            chunk = chunk.view(N, -1, self.output_size, self.output_size)
            fused.append(self.context_fusion(chunk))
        return fused
    
class ContextBranchBottleneck(nn.Module):
    """
    RoI Context Mining -> pooled vector.
    """
    def __init__(self, feature_strides, output_size=7, sampling_ratio=2):
        super().__init__()
        self.pooler = ROIPooler(
            output_size=(output_size, output_size),
            scales=[1.0/s for s in feature_strides],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2"
        )
        # fuse 8 context crops -> map of 256x output_size
        self.conv = nn.Conv2d(256*8, 256, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, features, proposals):
        # extract context crops
        all_boxes = []
        for props in proposals:
            boxes = props.proposal_boxes.tensor
            w = (boxes[:,2]-boxes[:,0])/3.0; h = (boxes[:,3]-boxes[:,1])/3.0
            crops = []
            for i in range(3):
                for j in range(3):
                    if i==1 and j==1: continue
                    crops.append(torch.stack([
                        boxes[:,0]+j*w, boxes[:,1]+i*h,
                        boxes[:,0]+j*w+w, boxes[:,1]+i*h+h
                    ], dim=1))
            all_boxes.append(Boxes(torch.cat(crops, dim=0)))
        ctx_feats = self.pooler(list(features), all_boxes)  # R_total x (256*8) x S x S
        # fuse and pool per proposal
        fused, ptr = [], 0
        for props in proposals:
            N = len(props)
            chunk = ctx_feats[ptr:ptr+8*N]
            ptr += 8*N
            merged = self.act(self.conv(chunk.view(N, -1, chunk.size(-2), chunk.size(-1))))
            vec = self.avgpool(merged).flatten(1)
            fused.append(vec)
        return fused  # list of Ni x 256
    

class ContextBranchWithLoss(nn.Module):
    """
    RoI Context Mining with auxiliary classification loss,
    masking out invalid labels.
    """
    def __init__(self, feature_strides, output_size=7, num_classes=80, sampling_ratio=2):
        super().__init__()
        self.num_classes = num_classes
        self.pooler = ROIPooler(
            output_size=(output_size, output_size),
            scales=[1.0 / s for s in feature_strides],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )
        self.conv = nn.Conv2d(256 * 8, 256, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, features, proposals, gt_classes=None):
        all_boxes = []
        for props in proposals:
            boxes = props.proposal_boxes.tensor
            w = (boxes[:, 2] - boxes[:, 0]) / 3.0
            h = (boxes[:, 3] - boxes[:, 1]) / 3.0
            crops = []
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    crops.append(torch.stack([
                        boxes[:, 0] + j * w,
                        boxes[:, 1] + i * h,
                        boxes[:, 0] + j * w + w,
                        boxes[:, 1] + i * h + h
                    ], dim=1))
            all_boxes.append(Boxes(torch.cat(crops, dim=0)))

        # Fixed: features is a list of tensors
        ctx_feats = self.pooler(features, all_boxes)

        outputs, aux_losses = [], {}
        ptr = 0
        for idx, props in enumerate(proposals):
            N = len(props)
            chunk = ctx_feats[ptr:ptr + 8 * N]
            ptr += 8 * N

            merged = chunk.view(N, -1, chunk.size(-2), chunk.size(-1))
            merged = self.act(self.conv(merged))
            vec = self.avgpool(merged).flatten(1)
            outputs.append(vec)

            if self.training and gt_classes is not None:
                labels = gt_classes[idx]
                valid = (labels >= 0) & (labels < self.num_classes)
                if valid.any():
                    logits = self.cls_head(vec[valid])
                    aux_losses[f"ctx_loss_{idx}"] = F.cross_entropy(logits, labels[valid])
                else:
                    aux_losses[f"ctx_loss_{idx}"] = vec.new_tensor(0.0, requires_grad=True)

        return outputs, aux_losses



class ContextBranchSE(nn.Module):
    """
    RoI Context Mining with auxiliary classification loss.
    """
    def __init__(self, feature_strides, output_size=7, num_classes=80, sampling_ratio=2):
        super().__init__()
        self.pooler = ROIPooler(
            output_size=(output_size, output_size),
            scales=[1.0/s for s in feature_strides],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2"
        )
        self.conv = nn.Conv2d(256*8, 256, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_head = nn.Linear(256, num_classes)

    def forward(self, features, proposals, gt_classes=None):
        all_boxes = []
        for props in proposals:
            boxes = props.proposal_boxes.tensor
            w = (boxes[:,2]-boxes[:,0])/3.0; h = (boxes[:,3]-boxes[:,1])/3.0
            crops = []
            for i in range(3):
                for j in range(3):
                    if i==1 and j==1: continue
                    crops.append(torch.stack([
                        boxes[:,0]+j*w, boxes[:,1]+i*h,
                        boxes[:,0]+j*w+w, boxes[:,1]+i*h+h
                    ], dim=1))
            all_boxes.append(Boxes(torch.cat(crops, dim=0)))
        ctx_feats = self.pooler(list(features), all_boxes)
        outputs, aux_losses, ptr = [], {}, 0
        for idx, props in enumerate(proposals):
            N = len(props)
            chunk = ctx_feats[ptr:ptr+8*N]; ptr += 8*N
            merged = self.act(self.conv(chunk.view(N, -1, chunk.size(-2), chunk.size(-1))))
            vec = self.avgpool(merged).flatten(1)
            outputs.append(vec)
            if self.training:
                labels = gt_classes[idx]
                logits = self.cls_head(vec)
                aux_losses[f"ctx_loss_{idx}"] = F.cross_entropy(logits, labels)
        return outputs, aux_losses