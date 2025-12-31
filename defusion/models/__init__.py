"""
DeFusion Model Components
=========================
Implementation of "Fusion from Decomposition: A Self-Supervised Decomposition
Approach for Image Fusion"

Components:
- ResNeSt: Split-attention residual blocks (Section 3.3)
- Encoder (E): Feature extraction with 3 MaxPool layers
- Ensembler (Ec): Single residual layer for feature ensembling
- DecoderU (Du): Unique feature decoder
- DecoderC (Dc): Common feature decoder
- ProjectionHeads (Pc, Pu, Pr): Upsample features to image space
- DeFusion: Complete model integrating all components
"""

from .resnest import SplitAttention, ResNeStBlock
from .denet import Encoder, Ensembler, DecoderU, DecoderC
from .projection_heads import ProjectionHead, ProjectionHeadC, ProjectionHeadU, ProjectionHeadR
from .defusion import DeFusion, build_defusion

__all__ = [
    'SplitAttention',
    'ResNeStBlock',
    'Encoder',
    'Ensembler',
    'DecoderU',
    'DecoderC',
    'ProjectionHead',
    'ProjectionHeadC',
    'ProjectionHeadU',
    'ProjectionHeadR',
    'DeFusion',
    'build_defusion'
]
