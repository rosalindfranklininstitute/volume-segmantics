"""Additional loss testing
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from volume_segmantics.data.pytorch3dunet_losses import (
    BoundaryDoULoss,
    BoundaryDoULossV2,
    BoundaryDoUDiceLoss,
    BoundaryLoss,
    TverskyLoss,
    FastSurfaceDiceLoss2D,
    WeightedCrossEntropyLoss,
)


def _one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W) long indices -> (B, C, H, W) float one-hot."""
    return F.one_hot(idx, num_classes).permute(0, 3, 1, 2).float()



# BoundaryDoULoss


def test_boundary_dou_interior_value_derived_from_kernel():
    # The magic "5" must come from the plus-kernel, not be hard-coded.
    loss = BoundaryDoULoss(n_classes=1)
    assert loss._interior_value == float(loss.boundary_kernel.sum())
    assert loss._interior_value == 5.0


def test_boundary_dou_forward_backward_multiclass():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 12, 12, requires_grad=True)
    target = _one_hot(torch.randint(0, 3, (2, 12, 12)), 3)

    loss = BoundaryDoULoss(n_classes=3)(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_boundary_dou_perfect_prediction_low_loss():
    target = _one_hot(torch.randint(0, 2, (2, 16, 16)), 2)
    # Logits that, after sigmoid, closely match the one-hot target.
    logits = (target * 2 - 1) * 10.0
    loss = BoundaryDoULoss(n_classes=2)(logits, target)
    assert loss.item() < 0.1


def test_boundary_dou_handles_empty_channel():
    # A channel with no foreground must not produce NaN/Inf.
    target = torch.zeros(1, 2, 8, 8)
    target[:, 0] = 1.0  # channel 1 entirely background
    logits = torch.randn(1, 2, 8, 8)
    loss = BoundaryDoULoss(n_classes=2)(logits, target)
    assert torch.isfinite(loss)



# BoundaryDoULossV2


def test_boundary_dou_v2_inherits_adaptive_size():
    # The refactor makes V2 a subclass; it must share the base machinery.
    assert issubclass(BoundaryDoULossV2, BoundaryDoULoss)
    v2 = BoundaryDoULossV2(n_classes=1)
    assert v2._adaptive_size.__func__ is BoundaryDoULoss._adaptive_size
    assert v2._interior_value == 5.0


def test_boundary_dou_v2_forward_backward():
    torch.manual_seed(1)
    logits = torch.randn(2, 2, 12, 12, requires_grad=True)
    target = _one_hot(torch.randint(0, 2, (2, 12, 12)), 2)

    loss = BoundaryDoULossV2(n_classes=2, allowed_outlier_fraction=0.25)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_boundary_dou_v2_outlier_fraction_changes_loss():
    torch.manual_seed(2)
    logits = torch.randn(2, 2, 12, 12)
    target = _one_hot(torch.randint(0, 2, (2, 12, 12)), 2)

    keep_all = BoundaryDoULossV2(n_classes=2, allowed_outlier_fraction=1.0)(logits, target)
    drop_some = BoundaryDoULossV2(n_classes=2, allowed_outlier_fraction=0.25)(logits, target)
    assert torch.isfinite(keep_all) and torch.isfinite(drop_some)
    # Discarding the hardest negatives should not increase the loss.
    assert drop_some.item() <= keep_all.item() + 1e-6



# BoundaryDoUDiceLoss


def test_boundary_dou_dice_forward_backward():
    torch.manual_seed(3)
    logits = torch.randn(2, 1, 12, 12, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 12, 12)).float()

    loss = BoundaryDoUDiceLoss(alpha=0.5, beta=0.5)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_boundary_dou_dice_is_weighted_sum():
    torch.manual_seed(4)
    logits = torch.randn(1, 1, 10, 10)
    target = torch.randint(0, 2, (1, 1, 10, 10)).float()

    combined = BoundaryDoUDiceLoss(alpha=0.5, beta=0.5)
    expected = (
        0.5 * combined.bdou(logits, target) + 0.5 * combined.dice(logits, target)
    )
    assert torch.allclose(combined(logits, target), expected)



# BoundaryLoss (signed distance map term)


def test_boundary_loss_forward_backward():
    torch.manual_seed(5)
    outputs = torch.randn(2, 1, 16, 16, requires_grad=True)
    gt = torch.zeros(2, 1, 16, 16)
    gt[:, :, 4:12, 4:12] = 1.0  # a square: both fg and bg present

    loss = BoundaryLoss(classes=1)(outputs, gt)
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs.grad is not None and torch.isfinite(outputs.grad).all()


def test_boundary_loss_degenerate_masks_are_finite():
    # Empty (all background) and full (all foreground) masks both leave a
    # distance transform constant; the guarded normalisation must not divide
    # by zero.
    bl = BoundaryLoss(classes=2)
    gt = torch.zeros(1, 2, 8, 8)
    gt[:, 1] = 1.0  # channel 0 empty, channel 1 full
    loss = bl(torch.randn(1, 2, 8, 8), gt)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0  # both channels contribute zero SDM



# TverskyLoss



def test_tversky_forward_in_unit_range_and_backward():
    torch.manual_seed(6)
    logits = torch.randn(2, 3, 12, 12, requires_grad=True)
    target = torch.randint(0, 3, (2, 12, 12))

    loss = TverskyLoss(classes=3)(logits, target)
    assert 0.0 <= loss.item() <= 1.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_tversky_perfect_prediction_low_loss():
    target = torch.randint(0, 3, (2, 12, 12))
    onehot = _one_hot(target, 3)
    logits = onehot * 20.0  # confident, correct
    loss = TverskyLoss(classes=3, include_background=True)(logits, target)
    assert loss.item() < 0.05


def test_tversky_alpha_beta_asymmetry():
    # beta penalises false negatives. Scoring only the foreground class
    # (background excluded), an under-segmenting prediction produces mostly
    # false negatives, so a higher beta must yield a higher loss than a
    # higher alpha. With background included the two penalties appear in
    # both channels and cancel, so the foreground-only view is the clean test.
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    # Under-segment: predict mostly background -> false negatives on fg.
    logits = torch.zeros(1, 2, 16, 16)
    logits[:, 0] = 3.0

    fn_heavy = TverskyLoss(classes=2, alpha=0.1, beta=0.9, include_background=False)(logits, target)
    fp_heavy = TverskyLoss(classes=2, alpha=0.9, beta=0.1, include_background=False)(logits, target)
    assert fn_heavy.item() > fp_heavy.item()


def test_tversky_raises_when_no_classes_to_evaluate():
    # classes=1 with background excluded leaves nothing to evaluate.
    logits = torch.randn(1, 1, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    with pytest.raises(ValueError):
        TverskyLoss(classes=1, include_background=False)(logits, target)



# FastSurfaceDiceLoss2D



def test_fast_surface_dice_forward_backward():
    torch.manual_seed(8)
    preds = torch.randn(2, 2, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 2, 16, 16)).float()

    loss = FastSurfaceDiceLoss2D()(preds, targets)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0 + 1e-4
    loss.backward()
    assert preds.grad is not None and torch.isfinite(preds.grad).all()


def test_fast_surface_dice_perfect_beats_bad():
    targets = torch.zeros(1, 1, 24, 24)
    targets[:, :, 6:18, 6:18] = 1.0
    good = (targets * 2 - 1) * 10.0          # matches target
    bad = (targets * 2 - 1) * -10.0          # inverted

    fn = FastSurfaceDiceLoss2D()
    assert fn(good, targets).item() < fn(bad, targets).item()


def test_fast_surface_dice_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        FastSurfaceDiceLoss2D()(torch.randn(1, 1, 8, 8), torch.randn(1, 2, 8, 8))



# WeightedCrossEntropyLoss (Variable removal regression)



def test_weighted_cross_entropy_forward_backward():
    torch.manual_seed(9)
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))

    loss = WeightedCrossEntropyLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_weighted_cross_entropy_class_weights_detached():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    weights = WeightedCrossEntropyLoss._class_weights(logits)
    # Weights must not carry gradient (previously enforced via Variable).
    assert not weights.requires_grad
