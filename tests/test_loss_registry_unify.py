"""Tests for the unified loss-construction path.

After unification, ``pipeline_registry`` is the single source of truth for
building losses: the head-aware snake_case factories (pipeline) and the raw
CamelCase factories (legacy 2D trainer) both live in
``volume_segmantics.model.loss_registry``. These tests pin the legacy
CamelCase names to the exact modules / parameters the old hard-coded
``VolSeg2dTrainer.loss_map`` produced.
"""

import torch
import torch.nn as nn
import pytest

from volume_segmantics.data import pipeline_registry as reg
import volume_segmantics.model.loss_registry  # noqa: F401  (registers losses)
from volume_segmantics.data.pytorch3dunet_losses import (
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
    BoundaryDoULoss,
    BoundaryDoUDiceLoss,
    BoundaryDoULossV2,
    BoundaryLoss,
)
from volume_segmantics.model.operations.trainer_losses import (
    ClassWeightedDiceLoss,
    CombinedCEDiceLoss,
)


# The superset kwargs dict the trainer now passes for every loss name.
TRAINER_KWARGS = dict(
    num_classes=4,
    alpha=0.3,
    beta=0.7,
    ce_weight=0.4,
    dice_weight=0.6,
    weight_mode="inverse_sqrt_freq",
    dice_weight_mode="inverse_sqrt_freq",
    exclude_background=True,
)


LEGACY_TYPE_MAP = {
    "BCEDiceLoss": BCEDiceLoss,
    "DiceLoss": DiceLoss,
    "BCELoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "GeneralizedDiceLoss": GeneralizedDiceLoss,
    "TverskyLoss": TverskyLoss,
    "BoundaryDoULoss": BoundaryDoULoss,
    "BoundaryDoUDiceLoss": BoundaryDoUDiceLoss,
    "BoundaryDoULossV2": BoundaryDoULossV2,
    "BoundaryLoss": BoundaryLoss,
    "ClassWeightedDiceLoss": ClassWeightedDiceLoss,
    "CombinedCEDiceLoss": CombinedCEDiceLoss,
}


@pytest.mark.parametrize("name,cls", list(LEGACY_TYPE_MAP.items()))
def test_legacy_camelcase_names_build_expected_type(name, cls):
    obj = reg.build_loss(name, **TRAINER_KWARGS)
    assert isinstance(obj, cls)
    assert isinstance(obj, nn.Module)


def test_bce_dice_weights_plumbed_from_alpha_beta():
    obj = reg.build_loss("BCEDiceLoss", **TRAINER_KWARGS)
    assert obj.alpha == 0.3
    assert obj.beta == 0.7


def test_combined_ce_dice_weights_plumbed_from_ce_dice_weight():
    # CombinedCEDiceLoss takes ce/dice weights, NOT the shared alpha/beta.
    obj = reg.build_loss("CombinedCEDiceLoss", **TRAINER_KWARGS)
    assert obj.alpha == 0.4   # ce_weight
    assert obj.beta == 0.6    # dice_weight


def test_tversky_classes_plumbed_from_num_classes():
    obj = reg.build_loss("TverskyLoss", **TRAINER_KWARGS)
    assert obj.classes == 4


def test_raw_dice_uses_identity_normalization():
    # The legacy trainer built DiceLoss(normalization="none").
    obj = reg.build_loss("DiceLoss", **TRAINER_KWARGS)
    x = torch.tensor([1.5, -2.0, 0.0])
    assert torch.equal(obj.normalization(x), x)


def test_boundary_dou_dice_uses_fixed_half_weights():
    # The legacy trainer fixed alpha=beta=0.5 regardless of settings.
    obj = reg.build_loss("BoundaryDoUDiceLoss", **TRAINER_KWARGS)
    assert obj.alpha == 0.5
    assert obj.beta == 0.5


def test_class_weighted_dice_params_plumbed():
    obj = reg.build_loss("ClassWeightedDiceLoss", **TRAINER_KWARGS)
    assert obj.num_classes == 4
    assert obj.weight_mode == "inverse_sqrt_freq"
    assert obj.exclude_background is True


def test_unknown_loss_raises_keyerror():
    # The trainer relies on this to trigger its SystemExit path.
    with pytest.raises(KeyError):
        reg.build_loss("lossnessmonster", **TRAINER_KWARGS)


def test_raw_and_semantic_names_coexist():
    # Same underlying class, two distinct registry names with different
    # target-handling semantics (raw vs head-aware).
    raw = reg.build_loss("DiceLoss", **TRAINER_KWARGS)
    semantic = reg.build_loss("dice", head_name="semantic", num_classes=4)
    assert type(raw).__name__ == "DiceLoss"
    assert type(semantic).__name__ == "_DiceForSemantic"
