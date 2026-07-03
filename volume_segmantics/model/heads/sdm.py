"""Signed distance map head — tanh-bounded ``[-1, 1]``.

Variants
--------
* ``binary`` — 1 channel; SDM of foreground vs background union.
* ``per_class`` — ``num_classes - 1`` channels; one SDM per non-background
  class.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from volume_segmantics.model.heads.base import TargetKind


SDM_HEAD_NAME = "sdm"


class SDMHead(nn.Module):
    """tanh-bounded signed distance map."""

    name: str = SDM_HEAD_NAME
    target_kind: TargetKind = TargetKind.SDM

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        activation: Optional[str] = "tanh",  # informational only
        deep_supervision: bool = False,
        num_classes: Optional[int] = None,
        spatial_dims: int = 2,
        config: Optional[object] = None,
        variant: str = "binary",
        d_clip: float = 10.0,
        **_extra,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(
                f"SDMHead currently only supports 2D (spatial_dims=2); "
                f"got {spatial_dims}."
            )
        if variant not in ("binary", "per_class"):
            raise ValueError(
                f"SDMHead.variant must be 'binary' or 'per_class'; "
                f"got {variant!r}"
            )
        if d_clip <= 0.0:
            raise ValueError(f"SDMHead.d_clip must be positive; got {d_clip}")

        # Resolve out_channels per variant if the caller didn't pin it.
        if out_channels is None:
            if variant == "binary":
                out_channels = 1
            else:
                if num_classes is None or num_classes < 2:
                    raise ValueError(
                        f"SDMHead variant='per_class' requires num_classes "
                        f">= 2 (so out_channels = num_classes - 1 >= 1); "
                        f"got num_classes={num_classes}"
                    )
                out_channels = num_classes - 1
        if out_channels < 1:
            raise ValueError(
                f"SDMHead.out_channels must be >= 1; got {out_channels}"
            )

        self.out_channels = int(out_channels)
        self.variant = variant
        self.d_clip = float(d_clip)

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        # tanh bounds output to [-1, 1] to match the SDM target's
        # range. Targets are positive-inside FG / negative-outside,
        # divided by ``d_clip`` and clipped — same range, same sign
        # convention.
        return torch.tanh(self.conv(decoder_features))


__all__ = ["SDM_HEAD_NAME", "SDMHead"]
