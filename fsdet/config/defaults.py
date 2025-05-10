from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# adding additional default values built on top of the default values in detectron2

_CC = _C

# Few‐Shot extension subtree
_CC.MODEL.FSOD = CN()
_CC.MODEL.FSOD.STAGE2 = False
_CC.MODEL.FSOD.CTX_LOSS_WEIGHT = 0.1
_CC.MODEL.FSOD.SHP_LOSS_WEIGHT = 0.1

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True
