"""Tests for target generators (boundary, distance, sdm).

"""

from __future__ import annotations

import numpy as np
import pytest

import volume_segmantics.data.targets  # triggers registration
from volume_segmantics.data import pipeline_registry as reg
from volume_segmantics.data.targets import (
    derive_boundary_target_2d,
    derive_distance_target_2d,
    derive_sdm_target_2d,
)


# Registration 


def test_three_target_generators_registered():
    assert set(reg.list_target_generators()) >= {
        "boundary", "distance", "sdm",
    }


def test_target_generator_factories_callable():
    for name in ("boundary", "distance", "sdm"):
        kwargs = (
            dict(num_classes=2, variant="binary", d_clip=5.0)
            if name == "sdm" else {}
        )
        gen = reg.build_target_generator(name, **kwargs)
        assert callable(gen)


# Boundary 


@pytest.fixture
def labels_square():
    """20x20 label slice with a single 10x10 foreground square."""
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[5:15, 5:15] = 1
    return arr


def test_boundary_shape_and_dtype(labels_square):
    out = derive_boundary_target_2d(labels_square, width=1)
    assert out.shape == (20, 20)
    assert out.dtype == np.float32
    # All values in {0, 1}.
    unique = np.unique(out)
    assert np.all(np.isin(unique, [0.0, 1.0]))


def test_boundary_only_at_edges(labels_square):
    out = derive_boundary_target_2d(labels_square, width=1)
    # Interior of the square (rows/cols 7..12) must be 0 — boundary
    # only at edge pixels.
    interior = out[7:13, 7:13]
    assert float(interior.sum()) == 0.0


def test_boundary_3d_input_raises():
    arr = np.zeros((4, 20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="2D slice"):
        derive_boundary_target_2d(arr)


def test_boundary_zero_width_raises(labels_square):
    with pytest.raises(ValueError, match=">= 1"):
        derive_boundary_target_2d(labels_square, width=0)


# Distance 


def test_distance_shape_and_dtype(labels_square):
    out = derive_distance_target_2d(labels_square)
    assert out.shape == (20, 20)
    assert out.dtype == np.float32


def test_distance_zero_on_foreground(labels_square):
    out = derive_distance_target_2d(labels_square)
    assert float(out[labels_square > 0].max()) == 0.0


def test_distance_positive_on_background(labels_square):
    out = derive_distance_target_2d(labels_square)
    assert float(out[labels_square == 0].min()) > 0.0


def test_distance_rejects_non_edt(labels_square):
    with pytest.raises(ValueError, match="must be 'edt'"):
        derive_distance_target_2d(labels_square, distance_transform="manhattan")


# SDM (test for the positive-inside convention) 


def test_sdm_binary_shape_and_range(labels_square):
    out = derive_sdm_target_2d(labels_square, variant="binary", d_clip=5.0)
    assert out.shape == (20, 20)
    assert out.dtype == np.float32
    assert float(out.min()) >= -1.0
    assert float(out.max()) <= 1.0


def test_sdm_positive_inside_foreground(labels_square):
    """**The convention test** — §1.2 of the release plan.

    v0.4.0b3 SDM is positive-inside / negative-outside (matching v0.5).
    The :class:`SDMHead`'s tanh output and the loss (L1/MSE) only
    line up under this sign convention.
    """
    out = derive_sdm_target_2d(labels_square, variant="binary", d_clip=5.0)
    # Foreground pixels: positive.
    fg_vals = out[labels_square > 0]
    assert float(fg_vals.mean()) > 0.0
    assert float(fg_vals.min()) >= 0.0  # boundary may be 0 but no negatives
    # Background pixels: negative.
    bg_vals = out[labels_square == 0]
    assert float(bg_vals.mean()) < 0.0
    assert float(bg_vals.max()) <= 0.0


def test_sdm_zero_on_boundary():
    """SDM should be ~0 on the boundary (where edt_inside == edt_outside)."""
    arr = np.zeros((30, 30), dtype=np.int32)
    arr[10:20, 10:20] = 1
    out = derive_sdm_target_2d(arr, variant="binary", d_clip=10.0)
    # The pixels just inside the FG edge have edt_inside ~= 1 and
    # edt_outside ~= 0 (background pixel just outside is 1 step away);
    # so SDM ~= 1 - 0 = 1 normalised by d_clip = 0.1. The literal
    # boundary line in pixelspace lives in BG, where SDM is slightly
    # negative. We just assert there's a sign flip across rows 9 -> 10.
    assert float(out[9, 15]) < 0.0   # just outside FG
    assert float(out[10, 15]) > 0.0  # just inside FG


def test_sdm_per_class_shape(labels_square):
    arr = labels_square.copy()
    arr[16:18, 16:18] = 2  # second foreground class
    out = derive_sdm_target_2d(
        arr, variant="per_class", num_classes=3, d_clip=5.0,
    )
    assert out.shape == (2, 20, 20)  # num_classes - 1 = 2
    assert out.dtype == np.float32


def test_sdm_per_class_class0_positive_inside_class0_only(labels_square):
    arr = labels_square.copy()
    arr[16:18, 16:18] = 2
    out = derive_sdm_target_2d(
        arr, variant="per_class", num_classes=3, d_clip=5.0,
    )
    # Channel 0 = class 1's SDM. Positive inside the 10x10 square
    # (where labels == 1), negative inside the 2x2 patch (labels == 2,
    # which is not class 1).
    assert float(out[0][arr == 1].mean()) > 0.0
    assert float(out[0][arr == 2].mean()) < 0.0


def test_sdm_per_class_requires_num_classes(labels_square):
    with pytest.raises(ValueError, match="num_classes >= 2"):
        derive_sdm_target_2d(
            labels_square, variant="per_class", num_classes=None,
        )


def test_sdm_unknown_variant_raises(labels_square):
    with pytest.raises(ValueError, match="binary.*per_class"):
        derive_sdm_target_2d(labels_square, variant="trinary")


def test_sdm_negative_d_clip_raises(labels_square):
    with pytest.raises(ValueError, match="d_clip"):
        derive_sdm_target_2d(labels_square, d_clip=0.0)


def test_sdm_clipping_at_d_clip(labels_square):
    """A small d_clip should clip distances to ±1 well inside / outside."""
    out = derive_sdm_target_2d(labels_square, variant="binary", d_clip=1.0)
    # Center of FG square — distance to nearest BG = 5; clipped to 1.
    assert float(out[10, 10]) == pytest.approx(1.0, abs=1e-6)
    # Far corner of BG — distance >> 1; clipped to -1.
    assert float(out[0, 0]) == pytest.approx(-1.0, abs=1e-6)


def test_sdm_rejects_non_edt(labels_square):
    with pytest.raises(ValueError, match="must be 'edt'"):
        derive_sdm_target_2d(labels_square, distance_transform="cityblock")


def test_sdm_3d_input_raises():
    arr = np.zeros((4, 20, 20), dtype=np.int32)
    with pytest.raises(ValueError, match="2D slice"):
        derive_sdm_target_2d(arr)
