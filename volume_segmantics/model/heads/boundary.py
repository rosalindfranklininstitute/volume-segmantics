"""Boundary head — single-channel sigmoid logits.

Output: ``(B, 1, H, W)`` raw logits. The boundary loss applies
``sigmoid`` internally (BCE-with-logits style) so the head emits raw.
Inference paths apply ``sigmoid`` externally before writing to the
prediction zarr's ``boundary_map`` array.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from volume_segmantics.model.heads.base import TargetKind


BOUNDARY_HEAD_NAME = "boundary"
BOUNDARY_CHANNELS: int = 1  # binary boundary mask


class BoundaryHead(nn.Module):
    """Binary boundary classifier; raw logits."""

    name: str = BOUNDARY_HEAD_NAME
    target_kind: TargetKind = TargetKind.BOUNDARY

    def __init__(
        self,
        in_channels: int,
        out_channels: int = BOUNDARY_CHANNELS,
        kernel_size: int = 3,
        activation: Optional[str] = None,
        deep_supervision: bool = False,
        num_classes: Optional[int] = None,
        spatial_dims: int = 2,
        config: Optional[object] = None,
        **_extra,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(
                f"BoundaryHead currently only supports 2D (spatial_dims=2); "
                f"got {spatial_dims}."
            )
        if out_channels != BOUNDARY_CHANNELS:
            raise ValueError(
                f"BoundaryHead.out_channels must be {BOUNDARY_CHANNELS} "
                f"(binary boundary mask); got {out_channels}"
            )
        self.out_channels = BOUNDARY_CHANNELS
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        return self.conv(decoder_features)


__all__ = ["BOUNDARY_CHANNELS", "BOUNDARY_HEAD_NAME", "BoundaryHead"]
