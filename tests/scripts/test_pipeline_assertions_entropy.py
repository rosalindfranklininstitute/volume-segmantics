"""The ``zarr-array-range`` entropy ceiling follows the class count.
A per-voxel entropy over C classes is at most ln C. 
"""

import math
from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.scripts.pipeline_assertions import assert_zarr_array_range


def _zarr_with(path: Path, n_classes: int, entropy_max: float, *, probs: bool = True) -> Path:
    g = zarr.open(str(path), mode="w")
    if probs:
        g.create_array("teacher_probs", data=np.full((n_classes, 2, 2, 2), 1.0 / n_classes, dtype=np.float32))
    g.create_array("tta_entropy_map", data=np.full((2, 2, 2), entropy_max, dtype=np.float32))
    return path


def test_three_class_entropy_at_ln3_passes_the_derived_bound(tmp_path):
    z = _zarr_with(tmp_path / "p.zarr", 3, math.log(3))
    assert_zarr_array_range(z, key="tta_entropy_map", minimum=0.0, max_log_classes=True)


def test_two_class_entropy_above_ln2_fails_and_names_the_bound(tmp_path):
    z = _zarr_with(tmp_path / "p.zarr", 2, math.log(3))
    with pytest.raises(AssertionError, match=r"> 0\.693.*\(ln 2 = 0\.6931\)"):
        assert_zarr_array_range(z, key="tta_entropy_map", minimum=0.0, max_log_classes=True)


def test_a_fixed_max_still_works(tmp_path):
    z = _zarr_with(tmp_path / "p.zarr", 3, 0.5)
    assert_zarr_array_range(z, key="tta_entropy_map", maximum=0.7)
    with pytest.raises(AssertionError, match="max=0.5 > 0.4"):
        assert_zarr_array_range(z, key="tta_entropy_map", maximum=0.4)


def test_both_bounds_at_once_is_refused(tmp_path):
    z = _zarr_with(tmp_path / "p.zarr", 3, 0.5)
    with pytest.raises(ValueError, match="not both"):
        assert_zarr_array_range(z, key="tta_entropy_map", maximum=0.7, max_log_classes=True)


def test_missing_probability_stack_is_reported(tmp_path):
    z = _zarr_with(tmp_path / "p.zarr", 3, 0.5, probs=False)
    with pytest.raises(AssertionError, match="'teacher_probs' missing"):
        assert_zarr_array_range(z, key="tta_entropy_map", max_log_classes=True)
