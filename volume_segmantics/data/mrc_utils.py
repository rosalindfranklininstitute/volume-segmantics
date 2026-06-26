"""Utilities for reading and writing MRC/MRCS/REC/MAP files.

MRC is the dominant container format for electron microscopy data, used by
IMOD, RELION, cryoSPARC, Dragonfly, and most cryo-ET/FIB-SEM acquisition
systems. All MRC variants (.mrc, .mrcs, .rec, .map, .st) share the same
binary layout and can be handled by a single reader.

Reference: https://www.ccpem.ac.uk/mrc_format/mrc2014.php
"""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np


def load_mrc(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an MRC/MRCS/REC/MAP file.

    Returns data in its native dtype (matching TIFF/HDF5 loader behaviour).
    The caller is responsible for casting to float32 for image data or
    keeping integer dtype for label volumes.

    Parameters
    ----------
    path : str or Path
        Path to the MRC file.

    Returns
    -------
    data : np.ndarray
        Volume array in (Z, Y, X) order, in its original dtype.
    meta : dict
        Keys: voxel_size_angstrom, voxel_size_nm, mrc_mode, source_dtype.
    """
    with mrcfile.open(str(path), mode="r", permissive=True) as mrc:
        data = mrc.data.copy()
        vox = mrc.voxel_size
        voxel_size = np.array([vox.z, vox.y, vox.x], dtype=np.float32)
        mrc_mode = int(mrc.header.mode)
        source_dtype = data.dtype
        mapc = int(mrc.header.mapc)
        mapr = int(mrc.header.mapr)
        maps = int(mrc.header.maps)

    # Reorder axes to canonical (Z, Y, X) if non-standard
    if (mapc, mapr, maps) != (1, 2, 3):
        col_ax = mapc - 1  # axis for columns  -> X
        row_ax = mapr - 1  # axis for rows     -> Y
        sec_ax = maps - 1  # axis for sections -> Z
        data = np.moveaxis(data, [sec_ax, row_ax, col_ax], [0, 1, 2])

    meta = {
        "voxel_size_angstrom": voxel_size,
        "voxel_size_nm": voxel_size / 10.0,
        "mrc_mode": mrc_mode,
        "source_dtype": source_dtype,
    }
    return data, meta


def _mrc_safe_dtype(data: np.ndarray) -> np.dtype:
    """Pick the smallest MRC-representable dtype that holds ``data`` losslessly.

    MRC supports integer modes int8/int16/uint16 and float modes float16/float32
    (int32 and float64 have no MRC mode). A blind ``astype(np.int8)`` corrupts any
    label >= 128 and truncates float maps, so we choose a dtype by value range
    instead and raise clearly when nothing fits.
    """
    if np.issubdtype(data.dtype, np.floating):
        # Preserve float maps (probabilities/logits/tomograms). float16 stays
        # float16; everything else (incl. float64) maps to MRC's float32.
        return np.dtype(np.float16) if data.dtype == np.float16 else np.dtype(np.float32)

    if data.dtype == np.bool_:
        return np.dtype(np.uint8)

    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(
            f"save_mrc cannot represent array of dtype {data.dtype!r} as MRC."
        )

    if data.size == 0:
        return np.dtype(np.uint8)

    lo = int(data.min())
    hi = int(data.max())
    if lo >= 0 and hi <= 255:
        return np.dtype(np.uint8)
    if -32768 <= lo and hi <= 32767:
        return np.dtype(np.int16)
    if lo >= 0 and hi <= 65535:
        return np.dtype(np.uint16)
    raise ValueError(
        f"save_mrc cannot represent integer values in range [{lo}, {hi}] as MRC "
        "(supported modes: int8/int16/uint16). A segmentation with this many "
        "labels is almost certainly an error."
    )


def save_mrc(
    data: np.ndarray,
    path: str | Path,
    voxel_size_angstrom: float | np.ndarray = 1.0,
    overwrite: bool = True,
) -> None:
    """Write a segmentation volume (or float map) as MRC.

    The output dtype is chosen to preserve ``data`` losslessly within the MRC
    format's supported modes -- integer label maps keep their values (uint8 for
    <=255 labels, int16/uint16 for larger ranges) and float probability/logit
    maps stay floating point. Values that no MRC mode can represent raise a
    ``ValueError`` rather than being silently corrupted.

    Parameters
    ----------
    data : np.ndarray
        Volume to write (label map or float map), in (Z, Y, X) order.
    path : str or Path
        Output file path.
    voxel_size_angstrom : float or array-like
        Voxel size in angstroms. Scalar sets isotropic spacing; a length-3
        array is interpreted as per-axis ``(z, y, x)`` spacing (matching the
        ``meta["voxel_size_angstrom"]`` returned by :func:`load_mrc`).
    overwrite : bool
        Whether to overwrite an existing file.
    """
    out_dtype = _mrc_safe_dtype(data)
    with mrcfile.new(str(path), overwrite=overwrite) as mrc:
        mrc.set_data(np.ascontiguousarray(data, dtype=out_dtype))

        vox = np.atleast_1d(np.asarray(voxel_size_angstrom, dtype=np.float32))
        if vox.size == 1:
            mrc.voxel_size = float(vox[0])
        else:
            # Our (z, y, x) convention -> mrcfile's (x, y, z) setter order.
            z, y, x = float(vox[0]), float(vox[1]), float(vox[2])
            mrc.voxel_size = (x, y, z)
