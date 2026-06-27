"""Tests for volume_segmantics.utilities.base_data_utils.

"""

import warnings

import numpy as np
import pytest

import volume_segmantics.utilities.base_data_utils as utils


def _clip(data, mean, sdf=2.575):
    return utils.clip_to_uint8(data, data_mean=mean, st_dev_factor=sdf)


@pytest.mark.parametrize("fill", [0.0, 7.0, -3.0, 255.0])
def test_clip_to_uint8_constant_volume_is_finite_and_deterministic(fill):
    """A constant volume (std dev == 0) must map to a uniform finite uint8."""
    data = np.full((4, 4, 4), fill, dtype=np.float32)
    out = _clip(data.copy(), mean=float(fill))
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()
    assert len(np.unique(out)) == 1  # uniform input -> uniform output


def test_clip_to_uint8_constant_volume_emits_no_runtime_warning():
    """The zero-range path must not divide-by-zero or do an undefined cast."""
    data = np.full((4, 4, 4), 7.0, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _clip(data.copy(), mean=7.0)
    assert np.isfinite(out).all()


def test_clip_to_uint8_constant_integer_volume():
    """Integer constant volume (cast to float internally) must be finite."""
    data = np.full((3, 5, 5), 42, dtype=np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _clip(data.copy(), mean=42.0)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


def test_clip_to_uint8_normal_volume_still_rescales():
    """Non-degenerate data must still span the 0..255 range as before."""
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 20.0, size=(8, 16, 16)).astype(np.float32)
    out = _clip(data.copy(), mean=float(data.mean()))
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_get_numpy_from_path_unknown_extension_raises_valueerror(tmp_path):
    bogus = tmp_path / "volume.unknownext"
    bogus.write_bytes(b"not a real volume")
    with pytest.raises(ValueError, match="unsupported file extension"):
        utils.get_numpy_from_path(bogus)


def test_get_numpy_from_path_no_extension_raises_valueerror(tmp_path):
    bogus = tmp_path / "volume_without_suffix"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported file extension"):
        utils.get_numpy_from_path(bogus)
