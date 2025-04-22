import logging

import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from fsdet.modeling.roi_heads import build_roi_heads
from fsdet.modeling.layers.dog_layer import DoGLayer2

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["DoGRCNNV2"]


@META_ARCH_REGISTRY.register()
class DoGRCNNV2(nn.Module):
    """
    Generalized R-CNN with FPN-level DoG fusion.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Backbone and FPN
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        # Normalizer
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # DoG layer and FPN-level fusion modules
        # cfg.MODEL.DOG.SIGMAS is a list of sigma values e.g. [1.0, 2.0, 4.0]
        self.dog_layer = DoGLayer2([1.0,2.0,4.0])
        fpn_shapes = self.backbone.output_shape()
        self.dog_encoders = nn.ModuleDict({
            level: nn.Conv2d(1, spec.channels, kernel_size=3, padding=1)
            for level, spec in fpn_shapes.items()
        })

        self.to(self.device)

        # Freezing stages per TFA
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")
        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")
        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
        # Freeze DoG modules during base training if specified
        if True:
            for p in self.dog_layer.parameters():
                p.requires_grad = False
            for p in self.dog_encoders.parameters():
                p.requires_grad = False
            print("froze DoG fusion modules in base stage")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        # extract GT instances
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # Backbone features
        features = self.backbone(images.tensor)

        # DoG map & normalization
        dog_map = self.dog_layer(images.tensor)  # Bx1xHxW
        # normalize per-image
        mean = dog_map.mean(dim=[2,3], keepdim=True)
        std = dog_map.std(dim=[2,3], keepdim=True) + 1e-6
        dog_map = (dog_map - mean) / std

        # FPN-level fusion: upsample, project, and add
        fused = {}
        for level, feat in features.items():
            dog_feat = F.interpolate(
                dog_map, size=feat.shape[-2:], mode='bilinear', align_corners=False
            )
            dog_proj = self.dog_encoders[level](dog_feat)
            fused[level] = feat + dog_proj
        features = fused

        # Proposals
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # ROI heads
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # DoG fusion in inference as well
        dog_map = self.dog_layer(images.tensor)
        mean = dog_map.mean(dim=[2,3], keepdim=True)
        std = dog_map.std(dim=[2,3], keepdim=True) + 1e-6
        dog_map = (dog_map - mean) / std
        fused = {}
        for level, feat in features.items():
            dog_feat = F.interpolate(
                dog_map, size=feat.shape[-2:], mode='bilinear', align_corners=False
            )
            fused[level] = feat + self.dog_encoders[level](dog_feat)
        features = fused

        # RPN / proposals
        if detected_instances is None:
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if not do_postprocess:
            return results

        processed_results = []
        for res, inp, size in zip(results, batched_inputs, images.image_sizes):
            height = inp.get("height", size[0])
            width = inp.get("width", size[1])
            res = detector_postprocess(res, height, width)
            processed_results.append({"instances": res})
        return processed_results

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        return ImageList.from_tensors(images, self.backbone.size_divisibility)
