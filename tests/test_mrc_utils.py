import numpy as np
import pytest
import mrcfile

from volume_segmantics.data.mrc_utils import load_mrc, save_mrc
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg


@pytest.fixture
def synthetic_mrc(tmp_path):
    """Create a minimal synthetic MRC file for testing."""
    vol = np.random.randint(0, 255, (32, 64, 64), dtype=np.int16)
    path = tmp_path / "test.mrc"
    with mrcfile.new(str(path)) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = 8.0  # 8 angstroms
    return path, vol


def test_load_mrc_shape(synthetic_mrc):
    path, original = synthetic_mrc
    data, meta = load_mrc(path)
    assert data.shape == original.shape


def test_load_mrc_dtype_preserved(synthetic_mrc):
    """load_mrc should preserve the native dtype (int16 here), not cast."""
    path, _ = synthetic_mrc
    data, meta = load_mrc(path)
    assert data.dtype == np.int16


def test_load_mrc_voxel_size(synthetic_mrc):
    path, _ = synthetic_mrc
    data, meta = load_mrc(path)
    assert meta["voxel_size_angstrom"] == pytest.approx(
        np.full(3, 8.0), rel=1e-4
    )
    assert meta["voxel_size_nm"] == pytest.approx(
        np.full(3, 0.8), rel=1e-4
    )


def test_load_mrc_preserves_values(synthetic_mrc):
    path, original = synthetic_mrc
    data, _ = load_mrc(path)
    np.testing.assert_array_equal(data, original)


def test_save_mrc_roundtrip(tmp_path):
    labels = np.array([[[0, 1, 2], [3, 0, 1]]], dtype=np.int8)
    path = tmp_path / "output.mrc"
    save_mrc(labels, path, voxel_size_angstrom=10.0)

    data, meta = load_mrc(path)
    np.testing.assert_array_equal(data, labels)


def test_mrc_suffixes_in_config():
    for ext in [".mrc", ".mrcs", ".rec", ".map", ".st"]:
        assert ext in cfg.MRC_SUFFIXES
        assert ext in cfg.TRAIN_DATA_EXT
        assert ext in cfg.PREDICT_DATA_EXT


def test_load_mrc_float_dtype_preserved(tmp_path):
    """A float32 MRC (e.g. a tomogram) should stay float32 after load."""
    vol = np.random.randn(16, 32, 32).astype(np.float32)
    path = tmp_path / "float_image.mrc"
    with mrcfile.new(str(path)) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = 5.0
    data, meta = load_mrc(path)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, vol)


def test_get_numpy_from_path_mrc(synthetic_mrc):
    path, original = synthetic_mrc
    data, chunking = utils.get_numpy_from_path(path)
    assert data.shape == original.shape
    assert data.dtype == np.int16  # native dtype preserved
    assert chunking is True
