"""Semantic head — the v0.4-equivalent classifier.

Output: ``(B, num_classes, H, W)`` raw logits. Activation is the loss's
job (cross-entropy / dice all consume logits directly).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from volume_segmantics.model.heads.base import TargetKind


SEMANTIC_HEAD_NAME = "semantic"


class SemanticHead(nn.Module):
    name: str = SEMANTIC_HEAD_NAME
    target_kind: TargetKind = TargetKind.SEMANTIC

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
                f"SemanticHead currently only supports 2D (spatial_dims=2); "
                f"got {spatial_dims}. The 3D path is deferred — see "
                f"docs/v0_4_b3_release_plan.md §0.3."
            )
        if out_channels < 1:
            raise ValueError(
                f"SemanticHead.out_channels must be >= 1; got {out_channels}"
            )
        self.out_channels = int(out_channels)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        return self.conv(decoder_features)


__all__ = ["SEMANTIC_HEAD_NAME", "SemanticHead"]
