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

# CBAM fusion module
from fsdet.modeling.fusion.cbam import CBAM
# DoG filter layer (computes Difference-of-Gaussians on raw images)
from fsdet.modeling.layers.dog_layer import DoGLayer
# Pretrained segmentation model
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models._utils import IntermediateLayerGetter


# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["SegmentationRCNN_final_logits_all_levels", 
            "SegmentationRCNN_level_wise_features"
]


@META_ARCH_REGISTRY.register()
class SegmentationRCNN_level_wise_features(nn.Module):
    
    """
    SegmentationRCNN: uses a pretrained DeepLabV3+ResNet101 segmentation stream
    to provide semantic features, fused with raw features via CBAM, for detection.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)

        self.seg_model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.seg_model.to(self.device).eval()
        
        # Extract intermediate features from segmentation backbone layers 1-4
        return_layers = {
            'layer1': 'res2',  # conv2_x -> res2
            'layer2': 'res3',  # conv3_x -> res3
            'layer3': 'res4',  # conv4_x -> res4
            'layer4': 'res5',  # conv5_x -> res5
        }
        self.seg_backbone = IntermediateLayerGetter(
            self.seg_model.backbone, return_layers=return_layers
        )

        # Adapter conv1x1 to map segmentation features to detection feature dims
        self.seg_adapters = nn.ModuleDict({
            lvl: nn.Conv2d(
                self.seg_model.classifier[4].in_channels,  # segmentation backbone out channels
                shape.channels,
                kernel_size=1,
                bias=False
            ) for lvl, shape in self.backbone.output_shape().items()
        })

        self.dog_layer = DoGLayer(
            channels=3,
            kernel_size=5,
            sigma1=1,
            sigma2=2
        )
        
        # CBAM fusion modules for each FPN level
        self.fusion_layers = nn.ModuleDict({
            lvl: CBAM(shape.channels, reduction=16)
            for lvl, shape in self.backbone.output_shape().items()
        })

        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

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
        self.to(self.device)

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

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        # 1. Preprocess raw images into ImageList
        images = self.preprocess_image(batched_inputs)
        feats_raw = self.backbone(images.tensor)
        

        # 3) Extract segmentation backbone features (multi-level)
        with torch.no_grad():
            seg_feats = self.seg_backbone(images.tensor)

        # 4) Adapt and fuse features per level
        feats_fused = {}
        for lvl, raw_feat in feats_raw.items():
            seg_feat = seg_feats[lvl]
            seg_adapted = self.seg_adapters[lvl](seg_feat)
            feats_fused[lvl] = self.fusion_layers[lvl](raw_feat, seg_adapted)



        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, feats_fused, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, feats_fused, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images

    def extract_seg_features(self, images_tensor, image_sizes):
        """
        Run DeepLabV3 on raw images to get segmentation logits,
        then adapt and upsample per FPN level.
        """
        with torch.no_grad():
            seg_out = self.seg_model(images_tensor)['out']  # [B,21,H,W]
        # Create ImageList for consistent sizes
        seg_list = ImageList(seg_out, image_sizes)
        feats_seg = {}
        for lvl, adapter in self.seg_adapters.items():
            # Upsample seg output to match raw FPN feature size
            size = self.backbone.output_shape()[lvl].stride  # pixel stride
            # Compute spatial size: assume images_tensor size//stride
            target_h = images_tensor.shape[2] // size
            target_w = images_tensor.shape[3] // size
            seg_feat = F.interpolate(seg_list.tensor,
                                     size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            feats_seg[lvl] = adapter(seg_feat)
        return feats_seg


@META_ARCH_REGISTRY.register()
class SegmentationRCNN_final_logits_all_levels(nn.Module):
    
    """
    SegmentationRCNN: uses a pretrained DeepLabV3+ResNet101 segmentation stream
    to provide semantic features, fused with raw features via CBAM, for detection.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)

        self.seg_model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.seg_model.to(self.device).eval()
        # Adapt segmentation output channels to FPN feature channels
        seg_out_channels = 21  # PASCAL VOC has 21 classes
        # Create adapters per level to map seg features to backbone channels
        self.seg_adapters = nn.ModuleDict({
            lvl: nn.Conv2d(seg_out_channels,
                           shape.channels,
                           kernel_size=1,
                           bias=False)
            for lvl, shape in self.backbone.output_shape().items()
        })

        self.dog_layer = DoGLayer(
            channels=3,
            kernel_size=5,
            sigma1=1,
            sigma2=2
        )
        
        # CBAM fusion modules for each FPN level
        self.fusion_layers = nn.ModuleDict({
            lvl: CBAM(shape.channels, reduction=16)
            for lvl, shape in self.backbone.output_shape().items()
        })

        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

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
        self.to(self.device)

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

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        # 1. Preprocess raw images into ImageList
        images = self.preprocess_image(batched_inputs)

        # 2) Segmentation feature extraction
        feats_seg = self.extract_seg_features(images.tensor, images.image_sizes)

        # 3. Extract features from raw and DoG streams
        feats_raw = self.backbone(images.tensor)

        # 4. Fuse features per FPN level
        feats_fused = {}
        for lvl in feats_raw:
            feats_fused[lvl] = self.fusion_layers[lvl](
                feats_raw[lvl], feats_seg[lvl]
            )



        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, feats_fused, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, feats_fused, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images

    def extract_seg_features(self, images_tensor, image_sizes):
        """
        Run DeepLabV3 on raw images to get segmentation logits,
        then adapt and upsample per FPN level.
        """
        with torch.no_grad():
            seg_out = self.seg_model(images_tensor)['out']  # [B,21,H,W]
        # Create ImageList for consistent sizes
        seg_list = ImageList(seg_out, image_sizes)
        feats_seg = {}
        for lvl, adapter in self.seg_adapters.items():
            # Upsample seg output to match raw FPN feature size
            size = self.backbone.output_shape()[lvl].stride  # pixel stride
            # Compute spatial size: assume images_tensor size//stride
            target_h = images_tensor.shape[2] // size
            target_w = images_tensor.shape[3] // size
            seg_feat = F.interpolate(seg_list.tensor,
                                     size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
            feats_seg[lvl] = adapter(seg_feat)
        return feats_seg


# @META_ARCH_REGISTRY.register()
# class ProposalNetwork(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.device = torch.device(cfg.MODEL.DEVICE)

#         self.backbone = build_backbone(cfg)
#         self.proposal_generator = build_proposal_generator(
#             cfg, self.backbone.output_shape()
#         )

#         pixel_mean = (
#             torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
#         )
#         pixel_std = (
#             torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
#         )
#         self.normalizer = lambda x: (x - pixel_mean) / pixel_std
#         self.to(self.device)

#     def forward(self, batched_inputs):
#         """
#         Args:
#             Same as in :class:`GeneralizedRCNN.forward`

#         Returns:
#             list[dict]: Each dict is the output for one input image.
#                 The dict contains one key "proposals" whose value is a
#                 :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
#         """
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [self.normalizer(x) for x in images]
#         images = ImageList.from_tensors(
#             images, self.backbone.size_divisibility
#         )
#         features = self.backbone(images.tensor)

#         if "instances" in batched_inputs[0]:
#             gt_instances = [
#                 x["instances"].to(self.device) for x in batched_inputs
#             ]
#         elif "targets" in batched_inputs[0]:
#             log_first_n(
#                 logging.WARN,
#                 "'targets' in the model inputs is now renamed to 'instances'!",
#                 n=10,
#             )
#             gt_instances = [
#                 x["targets"].to(self.device) for x in batched_inputs
#             ]
#         else:
#             gt_instances = None
#         proposals, proposal_losses = self.proposal_generator(
#             images, features, gt_instances
#         )
#         # In training, the proposals are not useful at all but we generate them anyway.
#         # This makes RPN-only models about 5% slower.
#         if self.training:
#             return proposal_losses

#         processed_results = []
#         for results_per_image, input_per_image, image_size in zip(
#             proposals, batched_inputs, images.image_sizes
#         ):
#             height = input_per_image.get("height", image_size[0])
#             width = input_per_image.get("width", image_size[1])
#             r = detector_postprocess(results_per_image, height, width)
#             processed_results.append({"proposals": r})
#         return processed_results
