"""Protocol and enum for head registry.

"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


class TargetKind(Enum):
    """What kind of target tensor a head consumes.
    """

    SEMANTIC = "semantic"
    BOUNDARY = "boundary"
    DISTANCE = "distance"
    SDM = "sdm"


@runtime_checkable
class PredictionHead(Protocol):
    """Structural protocol every head module satisfies.

    Heads are :class:`torch.nn.Module` subclasses that take the
    decoder's last-block feature map and project it to the head's
    output space. The Protocol is :func:`runtime_checkable` so the
    multi-task calculator + the model wrapper can validate head lists
    without forcing inheritance from a common base class — heads stay
    plain ``nn.Module`` subclasses.

    Attributes
    ----------
    name
        Canonical name registered into ``_HEADS`` (e.g. ``"semantic"``).
    out_channels
        Number of output channels on the head's projection. Equal to
        ``num_classes`` for ``"semantic"``; ``1`` for ``"boundary"`` /
        ``"distance"``; ``1`` (binary variant) or ``num_classes - 1``
        (per-class variant) for ``"sdm"``.
    target_kind
        Enum tag the dataset uses to route this head's target tensor
        through the right generator.
    """

    name: str
    out_channels: int
    target_kind: TargetKind

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        ...


__all__ = ["PredictionHead", "TargetKind"]
