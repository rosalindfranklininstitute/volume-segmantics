"""Format read/write round-trip identity matrix.

Writes a known asymmetric volume to each supported container and reads it back
via the public ``get_numpy_from_path`` dispatch, asserting the data survives.
Also covers the ``numpy_from_zarr`` plain-array vs group layouts (the previous
``zarr.open(path)[0]`` returned a single slice of a plain array and raised on a
group under zarr 3.x).
"""

import numpy as np
import pytest
import zarr

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.utilities.base_data_utils import numpy_from_zarr


def _asymmetric_volume(dtype):
    # Distinct value per voxel so a transposed/sliced/truncated read is caught.
    # Avoid dims of size 3/4 so tifffile does not guess an RGB photometric.
    vol = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    if np.issubdtype(np.dtype(dtype), np.floating):
        return (vol.astype(dtype) / 7.0).astype(dtype)
    return (vol % 200).astype(dtype)  # keep within uint8/int range


# --- HDF5 and TIFF: dtype-exact round trips ------------------------------- #

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_hdf5_roundtrip_exact(tmp_path, dtype):
    vol = _asymmetric_volume(dtype)
    path = tmp_path / "vol.h5"
    utils.save_data_to_hdf5(vol, path)
    data, _ = utils.get_numpy_from_path(path)
    assert data.shape == vol.shape
    assert data.dtype == vol.dtype
    np.testing.assert_array_equal(data, vol)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
def test_tiff_roundtrip_exact(tmp_path, dtype):
    vol = _asymmetric_volume(dtype)
    path = tmp_path / "vol.tiff"
    utils.save_data_to_tif(vol, path)
    data, _ = utils.get_numpy_from_path(path)
    assert data.shape == vol.shape
    assert data.dtype == vol.dtype
    np.testing.assert_array_equal(data, vol)


# --- MRC: value-preserving within supported modes ------------------------- #

@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_mrc_roundtrip_values(tmp_path, dtype):
    vol = _asymmetric_volume(dtype)
    path = tmp_path / "vol.mrc"
    utils.save_data_to_mrc(vol, path)
    data, _ = utils.get_numpy_from_path(path)
    assert data.shape == vol.shape
    if np.issubdtype(np.dtype(dtype), np.floating):
        np.testing.assert_allclose(data, vol, rtol=0, atol=1e-6)
    else:
        np.testing.assert_array_equal(data.astype(np.int64), vol.astype(np.int64))


# --- zarr: both store layouts --------------------------------------------- #

def test_zarr_plain_array_returns_full_volume(tmp_path):
    """Regression: a plain zarr array must return the whole volume, not slice 0."""
    vol = _asymmetric_volume(np.uint16)
    path = tmp_path / "plain.zarr"
    zarr.save_array(str(path), vol)
    data = numpy_from_zarr(path)
    assert data.shape == vol.shape  # previously returned vol[0] -> (4, 5)
    np.testing.assert_array_equal(data, vol)


def test_zarr_group_member_zero(tmp_path):
    """A multiscale-style group returns its full-resolution member '0'."""
    vol = _asymmetric_volume(np.uint16)
    path = tmp_path / "group.zarr"
    g = zarr.open_group(str(path), mode="w")
    g["0"] = vol
    data = numpy_from_zarr(path)
    assert data.shape == vol.shape
    np.testing.assert_array_equal(data, vol)


def test_zarr_via_get_numpy_from_path(tmp_path):
    vol = _asymmetric_volume(np.uint16)
    path = tmp_path / "plain.zarr"
    zarr.save_array(str(path), vol)
    data, chunking = utils.get_numpy_from_path(path)
    np.testing.assert_array_equal(data, vol)
    assert chunking is True


def test_zarr_empty_group_raises(tmp_path):
    path = tmp_path / "empty.zarr"
    zarr.open_group(str(path), mode="w")  # group with no arrays
    with pytest.raises(ValueError, match="no arrays"):
        numpy_from_zarr(path)


# --- HDF5 reader robustness (handle leak + sys.exit -> raise) -------------- #

def test_numpy_from_hdf5_bad_nexus_path_raises_not_exits(tmp_path):
    """A nexus file lacking the expected data paths must raise, not sys.exit."""
    import h5py

    path = tmp_path / "bad.nxs"
    with h5py.File(path, "w") as f:
        f.create_dataset("/somewhere/else", data=np.zeros((2, 2, 2)))
    # Previously this called sys.exit(1) inside the library; now it raises.
    with pytest.raises(KeyError):
        utils.numpy_from_hdf5(path, nexus=True)


def test_numpy_from_hdf5_releases_file_handle(tmp_path):
    """The HDF5 file must be closed after reading (no leaked handle).

    A leaked open handle blocks deletion on Windows; a successful unlink proves
    the handle was released. The returned array must remain valid afterwards
    (it is materialised before the file closes).
    """
    vol = _asymmetric_volume(np.uint16)
    path = tmp_path / "vol.h5"
    utils.save_data_to_hdf5(vol, path)
    data, _ = utils.numpy_from_hdf5(path, hdf5_path="/data")
    path.unlink()  # raises PermissionError if a handle is still open (Windows)
    np.testing.assert_array_equal(data, vol)  # array survives file deletion
