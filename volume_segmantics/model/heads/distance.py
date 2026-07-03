"""Distance head — single-channel smooth distance field.

Output: ``(B, 1, H, W)`` raw distance values (no activation; identity
output). The distance loss is L1 / L2 on the raw field, so the head
emits raw. Inference paths read the head's output verbatim into the
prediction zarr's ``distance_map`` array.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from volume_segmantics.model.heads.base import TargetKind


DISTANCE_HEAD_NAME = "distance"
DISTANCE_CHANNELS: int = 1


class DistanceHead(nn.Module):
    """Smooth distance field; identity output."""

    name: str = DISTANCE_HEAD_NAME
    target_kind: TargetKind = TargetKind.DISTANCE

    def __init__(
        self,
        in_channels: int,
        out_channels: int = DISTANCE_CHANNELS,
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
                f"DistanceHead currently only supports 2D (spatial_dims=2); "
                f"got {spatial_dims}."
            )
        if out_channels != DISTANCE_CHANNELS:
            raise ValueError(
                f"DistanceHead.out_channels must be {DISTANCE_CHANNELS}; "
                f"got {out_channels}"
            )
        self.out_channels = DISTANCE_CHANNELS
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        return self.conv(decoder_features)


__all__ = ["DISTANCE_CHANNELS", "DISTANCE_HEAD_NAME", "DistanceHead"]
