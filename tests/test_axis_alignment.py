"""Axis/orientation alignment regression tests.

A silent-correctness risk in the predictor is that per-axis
predictions get rotated back misaligned with the labels. These tests simulate
the full flow for each prediction axis with an asymmetric, coordinate-coded
volume and assert the rotated-back probabilities argmax to the original labels
voxel-for-voxel.

`rotate_array_to_axis` rotates a (Z, Y, X) volume so the chosen axis is axis 0
(sliced for 2D prediction); `rotate_4d_array_to_axis` rotates the stacked
per-slice predictions (S, C, H, W) back to (Z, Y, X, C). The two must be exact
inverses on the spatial layout.
"""

import numpy as np
import pytest

from volume_segmantics.utilities.base_data_utils import (
    rotate_array_to_axis,
    rotate_4d_array_to_axis,
)
from volume_segmantics.utilities.base_data_utils import Axis


def _coordinate_coded_labels(shape, num_classes):
    """Asymmetric labels: each voxel's class depends on all three coordinates,
    so any axis swap/transpose error changes the argmax somewhere."""
    z, y, x = np.indices(shape)
    return ((z * 7 + y * 13 + x * 17) % num_classes).astype(np.int64)


@pytest.mark.parametrize("axis", [Axis.Z, Axis.Y, Axis.X])
def test_perfect_prediction_rotates_back_aligned(axis):
    shape = (3, 4, 5)  # deliberately asymmetric (Z != Y != X)
    num_classes = 4
    labels = _coordinate_coded_labels(shape, num_classes)

    # 1. Rotate the volume so `axis` is axis 0 (the slicing axis).
    rotated = rotate_array_to_axis(labels, axis)
    s, h, w = rotated.shape

    # 2. "Predict" each 2D slice perfectly: one-hot along the channel dim,
    #    producing the (S, C, H, W) stack the predictor builds.
    stacked = np.zeros((s, num_classes, h, w), dtype=np.float32)
    for si in range(s):
        for c in range(num_classes):
            stacked[si, c] = rotated[si] == c

    # 3. Rotate the 4D predictions back to (Z, Y, X, C).
    aligned = rotate_4d_array_to_axis(stacked, axis)

    assert aligned.shape == (*shape, num_classes)
    pred = np.argmax(aligned, axis=-1)
    np.testing.assert_array_equal(pred, labels)


@pytest.mark.parametrize("axis", [Axis.Z, Axis.Y, Axis.X])
def test_rotate_array_to_axis_is_shape_consistent(axis):
    """rotate_array_to_axis moves the chosen axis to position 0."""
    shape = (3, 4, 5)
    vol = np.arange(np.prod(shape)).reshape(shape)
    rotated = rotate_array_to_axis(vol, axis)
    expected_first_dim = {Axis.Z: 3, Axis.Y: 4, Axis.X: 5}[axis]
    assert rotated.shape[0] == expected_first_dim
    # swapaxes-based rotation is its own inverse -> applying twice is identity.
    np.testing.assert_array_equal(rotate_array_to_axis(rotated, axis), vol)


def test_rotate_4d_rejects_non_4d():
    with pytest.raises(ValueError, match="expects 4D"):
        rotate_4d_array_to_axis(np.zeros((3, 4, 5)), Axis.Z)
