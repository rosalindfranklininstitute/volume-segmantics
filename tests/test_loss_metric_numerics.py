"""Numerical-stability tests for losses, metrics and entropy normalisation.

"""

import warnings

import numpy as np
import pytest
import torch

from volume_segmantics.data.pytorch3dunet_losses import (
    BoundaryDoULoss,
    GeneralizedDiceLoss,
    compute_per_channel_dice,
)
from volume_segmantics.model.operations.vol_seg_2d_predictor import (
    VolSeg2dPredictor,
)


# --------------------------------------------------------------------------- #
# Entropy normalisation guard (the real bug)                                   #
# --------------------------------------------------------------------------- #

def test_entropy_normalise_single_class_no_divide_by_zero():
    """One predicted class -> max entropy 0; must yield finite zeros, no warning."""
    entropy_matrix = np.zeros((3, 4, 4))
    labels = np.zeros((3, 4, 4), dtype=np.uint8)  # only label 0 present
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = VolSeg2dPredictor._normalise_entropy_map(entropy_matrix, labels)
    assert np.isfinite(out).all()
    assert np.all(out == 0.0)


def test_entropy_normalise_multiclass_divides_by_log_n():
    """With n>1 classes, normaliser is entropy(uniform over n) = log(n)."""
    from scipy.stats import entropy

    labels = np.array([0, 1, 2, 3], dtype=np.uint8)  # 4 classes
    raw = np.full((2, 2, 2), entropy(np.full(4, 0.25)))  # == max entropy
    out = VolSeg2dPredictor._normalise_entropy_map(raw.copy(), labels)
    # raw == max_entropy everywhere -> normalised to 1.0
    np.testing.assert_allclose(out, 1.0, rtol=1e-6)


def test_entropy_normalise_preserves_finiteness_for_two_classes():
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    raw = np.random.default_rng(0).random((2, 3, 3))
    out = VolSeg2dPredictor._normalise_entropy_map(raw.copy(), labels)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Loss finite-output / finite-gradient guards (already safe; pin it)           #
# --------------------------------------------------------------------------- #

def _forward_backward(loss_fn, inp, tgt):
    inp = inp.clone().requires_grad_(True)
    loss = loss_fn(inp, tgt)
    loss.backward()
    return loss.detach(), inp.grad


@pytest.mark.parametrize(
    "target",
    [
        torch.zeros(1, 1, 4, 4),                      # empty mask
        torch.ones(1, 1, 4, 4),                       # full mask
        torch.cat([torch.ones(1, 1, 1, 4),           # single-row mask
                   torch.zeros(1, 1, 3, 4)], dim=2),
    ],
)
def test_generalized_dice_loss_finite(target):
    loss, grad = _forward_backward(
        GeneralizedDiceLoss(normalization="sigmoid"),
        torch.zeros(1, 1, 4, 4),
        target,
    )
    assert torch.isfinite(loss).all()
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "target",
    [
        torch.zeros(1, 1, 4, 4),
        torch.ones(1, 1, 4, 4),
        torch.cat([torch.ones(1, 1, 1, 4),
                   torch.zeros(1, 1, 3, 4)], dim=2),
    ],
)
def test_boundary_dou_loss_finite(target):
    loss, grad = _forward_backward(
        BoundaryDoULoss(n_classes=1),
        torch.zeros(1, 1, 4, 4),
        target,
    )
    assert torch.isfinite(loss).all()
    assert torch.isfinite(grad).all()


def test_compute_per_channel_dice_empty_is_finite_zero():
    inp = torch.zeros(1, 2, 4, 4)
    out = compute_per_channel_dice(inp, torch.zeros(1, 2, 4, 4))
    assert torch.isfinite(out).all()
    assert torch.all(out == 0.0)
