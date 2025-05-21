import logging

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from torch import nn
import torch.nn.functional as F

from fsdet.modeling.roi_heads import build_roi_heads
from fsdet.modeling.fusion.cbam import CBAM
from fsdet.modeling.layers.dog_layer import DoGLayer
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models._utils import IntermediateLayerGetter

from .build import META_ARCH_REGISTRY

__all__ = ["ShapeRCNN"
]


@META_ARCH_REGISTRY.register()
class ShapeRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        # Appearance backbone + FPN
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        # Shape branch: use same backbone channels as standard FPN
        in_channels = [256, 512, 1024, 2048]
        self.shape_fpn = ShapeFPN(in_channels, out_channels=256)
        self.shape_roi_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        # Fusion conv
        self.fusion_conv = nn.Conv2d(512, 256, kernel_size=1)

        # Normalization
        mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1,1,1).to(self.device)
        std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1,1,1).to(self.device)
        self.normalizer = lambda x: (x - mean) / std
        self.to(self.device)

        # Freeze options
        if cfg.MODEL.BACKBONE.get("FREEZE", False):
            for p in self.backbone.parameters(): p.requires_grad = False
        if cfg.MODEL.PROPOSAL_GENERATOR.get("FREEZE", False):
            for p in self.proposal_generator.parameters(): p.requires_grad = False
        if cfg.MODEL.ROI_HEADS.get("FREEZE_FEAT", False):
            for p in self.roi_heads.box_head.parameters(): p.requires_grad = False

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        edge_maps = [compute_edge_map(x["image"].to(self.device)) for x in batched_inputs]
        image_list = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return image_list, edge_maps

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images, edge_maps = self.preprocess_image(batched_inputs)
        gt_instances = [x.get("instances", x.get("targets")).to(self.device)
                        for x in batched_inputs]

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        # Shape FPN features: use same pyramid inputs as backbone FPN levels
        shape_feats = self.shape_fpn([features[f] for f in sorted(features.keys())[:4]])

        # RoIAlign both streams
        boxes = [p.proposal_boxes for p in proposals]
        app_pooled = self.roi_heads.box_pooler(features, boxes)
        app_feat = self.roi_heads.box_head(app_pooled)
        shape_pooled = self.roi_heads.box_pooler({f:shape_feats[i] for i,f in enumerate(sorted(features.keys())[:4])}, boxes)
        shape_feat = self.shape_roi_conv(shape_pooled)

        # Fusion
        fused = self.fusion_conv(torch.cat([app_feat, shape_feat], dim=1))

        # Head predictions
        pred_instances, detector_losses = self.roi_heads.box_predictor(fused, proposals, gt_instances)

        # Auxiliary shape loss
        loss_shape = 0
        for i, edges in enumerate(edge_maps):
            crop = edges.unsqueeze(0).repeat(len(proposals[i]),1,1,1)
            pred_mask = F.interpolate(shape_feat, size=crop.shape[-2:], mode='bilinear', align_corners=False)
            loss_shape += F.l1_loss(pred_mask, crop)
        detector_losses['loss_shape'] = loss_shape * 0.1

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs):
        images, _ = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        processed = []
        for r, inp, size in zip(results, batched_inputs, images.image_sizes):
            h = inp.get('height', size[0]); w = inp.get('width', size[1])
            processed.append({'instances': detector_postprocess(r, h, w)})
        return processed