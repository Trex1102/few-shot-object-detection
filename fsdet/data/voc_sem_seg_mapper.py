# fsdet/data/voc_sem_seg_mapper.py

import numpy as np
from skimage.measure import label
import torch

from detectron2.data import DatasetMapper, detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances

class VOCSegmentationInstanceMapper(DatasetMapper):
    """
    A mapper that, *in addition* to the usual detection fields,
    reads the VOC sem-seg PNG and turns each connected region
    of each class into an instance mask.
    """

    def __init__(self, cfg, is_train: bool=True):
        super().__init__(cfg, is_train)
        # You might need to tweak the path logic below if your sem_seg lives elsewhere:
        self.sem_seg_root = cfg.INPUT.SEM_SEG_ROOT  # e.g. "VOCdevkit/VOC2012/SegmentationClass"

        # Build a lookup: semantic class name → class index in VOC
        # E.g. VOC uses 0=background, 15=aeroplane, etc.
        voc_metadata = utils.read_image(self.sem_seg_root + "/2012_001268.png")  # dummy read to get shape
        # Alternatively, hard-code your mapping here if you know it.

    def __call__(self, dataset_dict):
        """
        dataset_dict: from DatasetCatalog for VOC Segmentation,
        must contain "file_name" and "sem_seg_file_name"
        """
        # 1) Use the parent to load image + detection boxes
        data = super().__call__(dataset_dict)

        # 2) Load the semantic segmentation label (H x W, uint8 class IDs)
        sem_seg = utils.read_image(
            dataset_dict["sem_seg_file_name"], format="L"
        ).astype("int32")  # shape (H,W)

        H, W = sem_seg.shape
        instances = Instances((H, W))

        # 3) For each detection category in this image,
        #    extract its mask and find connected components
        masks = []
        classes = []
        boxes = []
        for obj in dataset_dict["annotations"]:
            cat_id = obj["category_id"]  # VOC index 0..20 minus background
            # Create binary mask for this class
            sem_mask = (sem_seg == cat_id).astype("uint8")

            # Label connected regions: each object instance separately
            labeled = label(sem_mask)
            for inst_id in range(1, labeled.max() + 1):
                single = (labeled == inst_id).astype("uint8")
                # skip tiny regions
                if single.sum() < 20:
                    continue
                masks.append(torch.from_numpy(single))
                classes.append(cat_id)
                # compute bbox from mask
                ys, xs = torch.nonzero(torch.from_numpy(single), as_tuple=True)
                y0, y1 = ys.min().item(), ys.max().item()
                x0, x1 = xs.min().item(), xs.max().item()
                boxes.append([x0, y0, x1, y1])

        if len(masks) > 0:
            instances.gt_masks = BitMasks(torch.stack(masks))
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
        else:
            # no instances—create empty fields
            instances.gt_masks = BitMasks(torch.empty((0, H, W), dtype=torch.uint8))
            instances.gt_classes = torch.empty((0,), dtype=torch.int64)
            instances.gt_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32))

        data["instances"] = instances
        return data
