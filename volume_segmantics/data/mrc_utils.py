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


def save_mrc(
    data: np.ndarray,
    path: str | Path,
    voxel_size_angstrom: float | np.ndarray = 1.0,
    overwrite: bool = True,
) -> None:
    """Write a segmentation volume as MRC.

    Parameters
    ----------
    data : np.ndarray
        Volume to write (typically a label map).
    path : str or Path
        Output file path.
    voxel_size_angstrom : float or array-like
        Voxel size in angstroms. Scalar sets isotropic; array sets per-axis.
    overwrite : bool
        Whether to overwrite an existing file.
    """
    with mrcfile.new(str(path), overwrite=overwrite) as mrc:
        mrc.set_data(data.astype(np.int8))
        mrc.voxel_size = float(np.mean(np.atleast_1d(voxel_size_angstrom)))
