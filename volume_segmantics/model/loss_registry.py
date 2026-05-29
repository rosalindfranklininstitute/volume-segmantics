"""Loss registry 
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from volume_segmantics.data import pipeline_registry as _registry
from volume_segmantics.data.pytorch3dunet_losses import (
    BoundaryDoULoss,
    BoundaryDoUDiceLoss,
    BoundaryLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
)
from volume_segmantics.model.operations.trainer_losses import (
    ClassWeightedDiceLoss,
    CombinedCEDiceLoss,
)


#  Helpers 


def _semantic_target_to_one_hot(
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Coerce a semantic target into ``(B, C, H, W) float`` one-hot.

    Accepts either:
    * ``(B, H, W) long`` class indices — converted to one-hot.
    * ``(B, C, H, W) float / long`` already one-hot — returned as float.
    """
    if target.dim() == 3:
        target_long = target.to(dtype=torch.long)
        return F.one_hot(target_long, num_classes=num_classes).permute(
            0, 3, 1, 2,
        ).float()
    if target.dim() == 4:
        return target.float()
    raise ValueError(
        f"semantic target must have 3 or 4 dims; got {target.dim()}"
    )


#  Wrapper classes (uniform forward(pred, target) signature) 


class _DiceForSemantic(nn.Module):
    """Multi-class dice with target-shape coercion.

    Wraps pytorch3dunet :class:`DiceLoss` with softmax normalisation
    and target one-hot conversion. The semantic head emits raw logits
    of shape ``(B, C, H, W)``; the dataset's semantic target is
    typically ``(B, H, W) long`` class indices. This wrapper bridges
    the two.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.inner = DiceLoss(normalization="softmax")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        target_oh = _semantic_target_to_one_hot(target, self.num_classes)
        return self.inner(pred, target_oh)


class _GeneralizedDiceForSemantic(nn.Module):
    """Generalized dice with target-shape coercion (semantic-only)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.inner = GeneralizedDiceLoss(normalization="softmax")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        target_oh = _semantic_target_to_one_hot(target, self.num_classes)
        return self.inner(pred, target_oh)


class _DiceForBoundary(nn.Module):
    """Sigmoid-dice on a 1-channel boundary head."""

    def __init__(self):
        super().__init__()
        self.inner = DiceLoss(normalization="sigmoid")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner(pred, target.float())


class _BoundaryBCEDice(nn.Module):
    """``alpha * BCE(logits) + beta * Dice(sigmoid)`` on a 1-channel head."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight)),
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(normalization="sigmoid")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float()
        return self.alpha * self.bce(pred, target) + self.beta * self.dice(
            pred, target,
        )


class _CrossEntropyWrapper(nn.Module):
    """``nn.CrossEntropyLoss`` with optional target-dim coercion.

    Accepts target as either ``(B, H, W) long`` (preferred) or
    ``(B, 1, H, W)`` (gets squeezed) or ``(B, C, H, W) float`` (one-hot,
    gets argmaxed back to indices — useful for unit tests).
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.inner = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        if target.dim() == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)
            else:
                # one-hot → indices
                target = target.argmax(dim=1)
        return self.inner(pred, target.long())


class _BCEForBoundary(nn.Module):
    """``nn.BCEWithLogitsLoss`` with optional ``pos_weight``."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.inner = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight)),
            )
        else:
            self.inner = nn.BCEWithLogitsLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner(pred, target.float())


class _RegressionLoss(nn.Module):
    """L1 / L2 regression with shape sanity for distance + SDM heads.

    Tolerates either ``(B, K, H, W)`` predictions paired with same-shape
    target, or a target missing the channel dim (``(B, H, W)``) which
    gets unsqueeze'd to ``(B, 1, H, W)`` for the 1-channel ``distance``
    case.
    """

    def __init__(self, kind: str = "l1"):
        super().__init__()
        if kind not in ("l1", "mse"):
            raise ValueError(f"kind must be 'l1' or 'mse'; got {kind!r}")
        self.kind = kind
        self.inner = nn.L1Loss() if kind == "l1" else nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float()
        if target.dim() == 3 and pred.dim() == 4 and pred.size(1) == 1:
            target = target.unsqueeze(1)
        return self.inner(pred, target)


#  Loss factories 


def _make_combined_ce_dice(
    *,
    num_classes: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    **_extra: Any,
) -> nn.Module:
    return CombinedCEDiceLoss(
        num_classes=int(num_classes), alpha=float(alpha), beta=float(beta),
    )


def _make_class_weighted_dice(
    *,
    num_classes: int = 2,
    weight_mode: str = "inverse_sqrt_freq",
    exclude_background: bool = False,
    **_extra: Any,
) -> nn.Module:
    return ClassWeightedDiceLoss(
        num_classes=int(num_classes),
        weight_mode=weight_mode,
        exclude_background=bool(exclude_background),
    )


def _make_dice(
    *,
    head_name: str = "semantic",
    num_classes: int = 2,
    **_extra: Any,
) -> nn.Module:
    if head_name == "semantic":
        return _DiceForSemantic(num_classes=int(num_classes))
    return _DiceForBoundary()


def _make_generalized_dice(
    *,
    head_name: str = "semantic",
    num_classes: int = 2,
    **_extra: Any,
) -> nn.Module:
    if head_name != "semantic":
        raise ValueError(
            f"generalized_dice is only supported for the semantic head; "
            f"got head_name={head_name!r}"
        )
    return _GeneralizedDiceForSemantic(num_classes=int(num_classes))


def _make_cross_entropy(
    *,
    num_classes: int = 2,
    class_weights: Optional[list] = None,
    **_extra: Any,
) -> nn.Module:
    weight = (
        torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None else None
    )
    return _CrossEntropyWrapper(num_classes=int(num_classes), weight=weight)


class _TverskyForSemantic(nn.Module):
    """:class:`TverskyLoss` with semantic-target shape coercion.

    TverskyLoss does its own ``_one_hot_encoder`` from class
    indices, so we feed it ``(B, H, W) long`` directly. Targets that
    arrive as ``(B, C, H, W)`` one-hot get argmaxed back to indices.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        include_background: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.inner = TverskyLoss(
            classes=int(num_classes),
            alpha=float(alpha), beta=float(beta),
            include_background=include_background,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        if target.dim() == 4:
            if target.size(1) == 1:
                target = target.squeeze(1)
            else:
                target = target.argmax(dim=1)
        return self.inner(pred, target.long())


def _make_tversky(
    *,
    num_classes: int = 2,
    alpha: float = 0.3,
    beta: float = 0.7,
    include_background: bool = False,
    **_extra: Any,
) -> nn.Module:
    # TverskyLoss expects logits + one-hot target; the wrapper
    # coerces ``(B, H, W) long`` indices to one-hot for parity with the
    # other semantic losses.
    return _TverskyForSemantic(
        num_classes=int(num_classes),
        alpha=float(alpha), beta=float(beta),
        include_background=include_background,
    )


def _make_bce(
    *,
    pos_weight: Optional[float] = None,
    **_extra: Any,
) -> nn.Module:
    return _BCEForBoundary(pos_weight=pos_weight)


def _make_boundary_dice(**_extra: Any) -> nn.Module:
    return _DiceForBoundary()


def _make_boundary_bce_dice(
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    pos_weight: Optional[float] = None,
    **_extra: Any,
) -> nn.Module:
    return _BoundaryBCEDice(
        alpha=float(alpha), beta=float(beta), pos_weight=pos_weight,
    )


def _make_boundary_loss(**_extra: Any) -> nn.Module:
    return BoundaryLoss()


def _make_boundary_dou(**_extra: Any) -> nn.Module:
    return BoundaryDoULoss()


def _make_l1(**_extra: Any) -> nn.Module:
    return _RegressionLoss(kind="l1")


def _make_mse(**_extra: Any) -> nn.Module:
    return _RegressionLoss(kind="mse")


#  Import-time registration 


_REGISTRATIONS = (
    # Semantic / multi-class
    ("dice_ce",             _make_combined_ce_dice),  # alias
    ("combined_ce_dice",    _make_combined_ce_dice),
    ("class_weighted_dice", _make_class_weighted_dice),
    ("dice",                _make_dice),
    ("generalized_dice",    _make_generalized_dice),
    ("cross_entropy",       _make_cross_entropy),
    ("tversky",             _make_tversky),
    # Binary / boundary
    ("bce",                 _make_bce),
    ("boundary_bce",        _make_bce),  # alias for binary BCE on boundary
    ("boundary_dice",       _make_boundary_dice),
    ("boundary_bce_dice",   _make_boundary_bce_dice),
    ("boundary_loss",       _make_boundary_loss),
    ("boundary_dou",        _make_boundary_dou),
    # Distance / SDM regression
    ("distance_l1",         _make_l1),
    ("distance_mse",        _make_mse),
    ("sdm_l1",              _make_l1),
    ("sdm_mse",             _make_mse),
)

for _name, _factory in _REGISTRATIONS:
    try:
        _registry.register_loss(_name, _factory)
    except KeyError:
        # Already registered — module reimport in the same process.
        pass


__all__ = []  # nothing to export — registry is the surface
