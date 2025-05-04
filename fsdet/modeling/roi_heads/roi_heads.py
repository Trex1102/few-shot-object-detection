"""Implement ROI_heads."""
# flake8: noqa
import logging
from typing import Dict

import numpy as np
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes


from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputs

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from .context_branch import ContextBranch
from .shape_branch import ShapeBranch
from .context_branch import ContextBranchWithLoss
from .shape_branch import ShapeBranchWithLoss
from .context_branch import ContextBranchBottleneck
from .shape_branch import ShapeBranchBottleneck
from .context_branch import ContextBranchSE
from .shape_branch import ShapeBranchSE 


from fsdet.modeling.fusion.se import SEFusion




ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances

@ROI_HEADS_REGISTRY.register()
class ParallelFusionROIHeadsBottleneck(StandardROIHeads):
    """
    Lightweight projection fusion: project each branch to 256-dim.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        strides = [input_shape[f].stride for f in self.in_features]
        in_ch = input_shape[self.in_features[0]].channels
        # branches
        self.context_branch = ContextBranchBottleneck(strides, output_size=res)
        self.shape_branch = ShapeBranchBottleneck(in_channels=in_ch)
        # projection modules
        head_dim = self.box_head.output_size
        self.box_proj = nn.Sequential(nn.Linear(head_dim,256), nn.ReLU(inplace=True))
        self.ctx_proj = nn.Sequential(nn.Linear(256,256), nn.ReLU(inplace=True))
        self.shp_proj = nn.Sequential(nn.Linear(256,256), nn.ReLU(inplace=True))
        # fusion norm
        self.fusion_bn = nn.BatchNorm1d(256*3)
        # predictor on 768-d
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=256*3, height=1, width=1)
        )

    def _forward_box(self, features, proposals):
        # 1) pooled and box_head: R x C x H x W -> R x D
        pooled = self.box_pooler(features, [p.proposal_boxes for p in proposals])
        head = self.box_head(pooled)
        D = head.view(head.size(0), -1)
        # split per image
        nums = [len(p) for p in proposals]
        head_list = list(D.split(nums, dim=0))  # Ni x head_dim
        # project box
        box_vecs = [self.box_proj(h) for h in head_list]
        # 2) context vectors
        ctx_vecs = [self.ctx_proj(v) for v in self.context_branch(features, proposals)]
        # 3) shape vectors: need roi_pooled list
        roi_list = list(pooled.split(nums, dim=0))
        shp_vecs = [self.shp_proj(v) for v in self.shape_branch(roi_list)]
        # fuse per image
        fused = [torch.cat([b,c,s], dim=1) for b,c,s in zip(box_vecs, ctx_vecs, shp_vecs)]
        F = torch.cat(fused, dim=0)
        F = self.fusion_bn(F)
        # predict
        logits, deltas = self.box_predictor(F)
        outputs = FastRCNNOutputs(
            self.box2box_transform, logits, deltas, proposals, self.smooth_l1_beta
        )
        if self.training:
            return outputs.losses()
        else:
            inst, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return inst



@ROI_HEADS_REGISTRY.register()
class ParallelFusionROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # parallel branches
        pooler_res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.context_branch = ContextBranch(
            feature_strides=[input_shape[f].stride for f in self.in_features],
            output_size=pooler_res
        )
        in_ch = input_shape[self.in_features[0]].channels
        self.shape_branch = ShapeBranch(in_channels=in_ch)
        # compute feature dims
        head_feat_dim = self.box_head.output_size
        context_feat_dim = 256 * pooler_res * pooler_res
        shape_feat_dim = 256
        fused_dim = head_feat_dim + context_feat_dim + shape_feat_dim
        # batch norm on fused vector
        self.fusion_bn = nn.BatchNorm1d(fused_dim)
        # override predictor
        self.box_predictor = FastRCNNOutputLayers(
            cfg,
            ShapeSpec(channels=fused_dim, height=1, width=1)
        )

    def _forward_box(self, features, proposals):
        # 1) pooled features
        pooled = self.box_pooler(
            features,
            [p.proposal_boxes for p in proposals]
        )  # R x C x H x W
        # 2) box head features
        head_feats = self.box_head(pooled)  # R x D
        num_rois = [len(p) for p in proposals]
        head_list = list(head_feats.split(num_rois, dim=0))
        # 3) context features
        ctx_feats = self.context_branch(features, proposals)
        ctx_flat = [cf.view(cf.size(0), -1) for cf in ctx_feats]
        # 4) shape features
        roi_list = list(pooled.split(num_rois, dim=0))
        shp_feats = self.shape_branch(roi_list)
        # 5) fuse and normalize
        fused_list = []
        for hf, cf, sf in zip(head_list, ctx_flat, shp_feats):
            vec = torch.cat([hf, cf, sf], dim=1)
            fused_list.append(vec)
        fused_feats = torch.cat(fused_list, dim=0)  # R x fused_dim
        fused_feats = self.fusion_bn(fused_feats)
        # 6) prediction
        pred_logits, pred_deltas = self.box_predictor(fused_feats)
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_logits,
            pred_deltas,
            proposals,
            self.smooth_l1_beta
        )
        if self.training:
            return outputs.losses()
        else:
            instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img
            )
            return instances


@ROI_HEADS_REGISTRY.register()
class ParallelFusionROIHeadsWithLoss(StandardROIHeads):
    """
    Parallel fusion ROI heads with decoupled auxiliary losses.
    Context and Shape branches are run on detached features to
    avoid backprop into the backbone/RPN.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # RoI pooler resolution and feature strides
        fsod_cfg = cfg.MODEL.get("FSOD", {})
        self.stage2     = fsod_cfg.get("STAGE2", False)
        self.ctx_weight = fsod_cfg.get("CTX_LOSS_WEIGHT", 1.0)
        self.shp_weight = fsod_cfg.get("SHP_LOSS_WEIGHT", 1.0)

        res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        strides = [input_shape[f].stride for f in self.in_features]
        in_ch = input_shape[self.in_features[0]].channels
        num_cls = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        # Auxiliary branches
        self.context_branch = ContextBranchWithLoss(
            feature_strides=strides,
            output_size=res,
            num_classes=num_cls
        )
        self.shape_branch = ShapeBranchWithLoss(
            in_channels=in_ch,
            embedding_dim=128
        )

        # Projection heads to 256-d
        head_dim = self.box_head.output_size
        self.box_proj = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.ReLU(inplace=True)
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.shp_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )

        # Fusion normalization and predictor
        self.fusion_bn = nn.BatchNorm1d(256 * 3)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=256 * 3, height=1, width=1)
        )

    def _forward_box(self, features, proposals, targets=None):
        """
        Unified train/inference for fused head.
        """
        # 1) RoI pooling + box head
        pooled    = self.box_pooler(features, [p.proposal_boxes for p in proposals])
        box_feats = self.box_head(pooled)                  # R×C×H×W
        flat      = box_feats.view(box_feats.size(0), -1)  # R×head_dim
        nums      = [len(p) for p in proposals]
        head_list = flat.split(nums, dim=0)

        # 2) Prepare GT only if training
        if self.training and targets is not None:
            gt_classes = [p.gt_classes for p in proposals]
            gt_masks   = [t.gt_masks.tensor for t in targets]
        else:
            gt_classes = None
            gt_masks   = None

        # 3) Run aux branches only in training Stage1
        if self.training and not self.stage2:
            detached_feats  = [f.detach() for f in features]
            ctx_vecs, ctx_losses = self.context_branch(detached_feats, proposals, gt_classes)

            roi_list        = pooled.split(nums, dim=0)
            detached_rois   = [r.detach() for r in roi_list]
            shp_vecs, shp_losses = self.shape_branch(detached_rois, gt_masks)
        else:
            # dummy empty losses and vectors for Stage2 or inference
            ctx_vecs = [head.new_zeros(head.size(0), 256) for head in head_list]
            shp_vecs = [head.new_zeros(head.size(0), 128) for head in head_list]
            ctx_losses = {}
            shp_losses = {}

        # 4) Project into 256-d
        box_vecs = [self.box_proj(h) for h in head_list]
        ctx_proj = [self.ctx_proj(v) for v in ctx_vecs]
        shp_proj = [self.shp_proj(v) for v in shp_vecs]

        # 5) Fuse & normalize
        fused  = [torch.cat([b, c, s], dim=1)
                for b, c, s in zip(box_vecs, ctx_proj, shp_proj)]
        Fcat   = torch.cat(fused, dim=0)
        Fnorm  = self.fusion_bn(Fcat)

        # 6) Predict
        logits, deltas = self.box_predictor(Fnorm)
        outputs = FastRCNNOutputs(
            self.box2box_transform, logits, deltas, proposals, self.smooth_l1_beta
        )

        if self.training:
            # detection loss
            losses = outputs.losses()
            # add aux if in Stage1
            if not self.stage2:
                avg_ctx = sum(ctx_losses.values()) / max(1, len(ctx_losses))
                avg_shp = sum(shp_losses.values()) / max(1, len(shp_losses))
                losses["ctx_loss"] = avg_ctx * self.ctx_weight
                losses["shp_loss"] = avg_shp * self.shp_weight
            return losses
        else:
            # fused inference
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class ParallelFusionROIHeadsSE(StandardROIHeads):
    """
    Channel-wise attention fusion via SE using learned branch embeddings.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        res = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        strides = [input_shape[f].stride for f in self.in_features]
        in_ch = input_shape[self.in_features[0]].channels
        num_cls = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # branches
        self.context_branch = ContextBranchSE(strides, output_size=res, num_classes=num_cls)
        self.shape_branch = ShapeBranchSE(in_channels=in_ch, embedding_dim=128)
        # SE fusion: input C, embeddings 256+128
        channel_dim = self.box_head.output_size // (res*res)
        self.se_fusion = SEFusion(channel_dim, emb_dim=256+128)
        # predictor
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=channel_dim, height=1, width=1)
        )

    def _forward_box(self, features, proposals, targets=None):
        pooled = self.box_pooler(features, [p.proposal_boxes for p in proposals])
        # split per image
        nums = [len(p) for p in proposals]
        roi_list = list(pooled.split(nums, dim=0))
        gt_classes = [p.gt_classes for p in proposals]
        gt_masks = [t.gt_masks.tensor for t in targets] if targets is not None else None
        # branch outputs
        ctx_vecs, ctx_losses = self.context_branch(features, proposals, gt_classes)
        shp_vecs, shp_losses = self.shape_branch(roi_list, gt_masks)
        # fuse channel-wise
        fused_maps = []
        for fmap, cv, sv in zip(roi_list, ctx_vecs, shp_vecs):
            fused_maps.append(self.se_fusion(fmap, cv, sv))
        fused = torch.cat(fused_maps, dim=0)
        # box head on fused maps
        head_feats = self.box_head(fused)
        flat = head_feats.flatten(start_dim=1)
        logits, deltas = self.box_predictor(flat)
        outputs = FastRCNNOutputs(self.box2box_transform, logits, deltas, proposals, self.smooth_l1_beta)
        if self.training:
            losses = outputs.losses()
            losses.update(ctx_losses)
            losses.update(shp_losses)
            return losses
        else:
            inst, _ = outputs.inference(self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img)
            return inst