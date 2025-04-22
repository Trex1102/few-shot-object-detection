from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .dog_rcnn import DoGRCNN, DoGRCNN_low_levels_only, DoGRCNN_normal_inference, DoGRCNN_low_levels_only_normal_inference
from .segmentation_rcnn import SegmentationRCNN_final_logits_all_levels, SegmentationRCNN_level_wise_features
from .dog_rcnn_v2 import DoGRCNNV2