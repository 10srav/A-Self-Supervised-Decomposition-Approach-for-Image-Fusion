"""
DeFusion Datasets
=================
Dataset classes and augmentation for DeFusion training.

Components:
- CUDAugmentation: Common and Unique Decomposition augmentation
- COCODataset: COCO dataset loader with CUD augmentation
- FusionDataset: Generic dataset for fusion evaluation
"""

from .cud_augmentation import CUDAugmentation, generate_non_overlapping_masks
from .coco_dataset import COCODataset, get_coco_dataloader

__all__ = [
    'CUDAugmentation',
    'generate_non_overlapping_masks',
    'COCODataset',
    'get_coco_dataloader'
]
