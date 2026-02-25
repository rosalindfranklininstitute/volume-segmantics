"""
Custom encoders for Volume Segmantics.

This package contains encoder implementations that extend the encoders
available in segmentation_models_pytorch.
"""

from volume_segmantics.model.encoders.dino_encoder import (
    DINOEncoder,
    register_dino_encoders,
)

__all__ = [
    "DINOEncoder",
    "register_dino_encoders",
]
