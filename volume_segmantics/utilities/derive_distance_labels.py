"""Derive distance map or signed distance field from a segmentation label volume.

Supports Euclidean Distance Transform (EDT) and Signed Distance Field (SDF)
in both 3D volumetric and 2D slice-wise modes.

Usage:
    python -m volume_segmantics.utilities.derive_distance_labels \\
        input.h5 output.h5 --type sdf --mode 3d --normalize --clip_max 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def derive_edt_volume(labels: np.ndarray, sampling=None) -> np.ndarray:
    """Euclidean Distance Transform: distance from background to nearest foreground.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X).
    sampling : tuple, optional
        Voxel spacing (dz, dy, dx) for anisotropic data.

    Returns
    -------
    np.ndarray
        Float32 distance map (0 at foreground, positive in background).
    """
    binary = (labels > 0).astype(np.uint8)
    edt = distance_transform_edt(1 - binary, sampling=sampling)
    return edt.astype(np.float32)


def derive_sdf_volume(labels: np.ndarray, sampling=None) -> np.ndarray:
    """Signed Distance Field: negative inside foreground, positive outside.

    SDF = EDT_outside - EDT_inside
    Zero at the boundary.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X).
    sampling : tuple, optional
        Voxel spacing (dz, dy, dx).

    Returns
    -------
    np.ndarray
        Float32 signed distance field.
    """
    binary = (labels > 0).astype(np.uint8)
    edt_outside = distance_transform_edt(1 - binary, sampling=sampling)
    edt_inside = distance_transform_edt(binary, sampling=sampling)
    sdf = edt_outside - edt_inside
    return sdf.astype(np.float32)


def derive_edt_2d_slicewise(labels: np.ndarray, sampling=None) -> np.ndarray:
    """2D EDT applied per z-slice for anisotropic data.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X).
    sampling : tuple, optional
        2D pixel spacing (dy, dx).

    Returns
    -------
    np.ndarray
        Float32 distance map.
    """
    result = np.zeros_like(labels, dtype=np.float32)
    for z in range(labels.shape[0]):
        binary = (labels[z] > 0).astype(np.uint8)
        result[z] = distance_transform_edt(1 - binary, sampling=sampling)
    return result


def derive_sdf_2d_slicewise(labels: np.ndarray, sampling=None) -> np.ndarray:
    """2D SDF applied per z-slice for anisotropic data.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X).
    sampling : tuple, optional
        2D pixel spacing (dy, dx).

    Returns
    -------
    np.ndarray
        Float32 signed distance field.
    """
    result = np.zeros_like(labels, dtype=np.float32)
    for z in range(labels.shape[0]):
        binary = (labels[z] > 0).astype(np.uint8)
        edt_outside = distance_transform_edt(1 - binary, sampling=sampling)
        edt_inside = distance_transform_edt(binary, sampling=sampling)
        result[z] = edt_outside - edt_inside
    return result


def normalize_distance_map(dist: np.ndarray, clip_max: float = None) -> np.ndarray:
    """Normalize distance map to [-1, 1] range.

    Parameters
    ----------
    dist : np.ndarray
        Distance map (EDT or SDF).
    clip_max : float, optional
        Clip absolute values before normalizing.

    Returns
    -------
    np.ndarray
        Normalized distance map in [-1, 1].
    """
    if clip_max is not None:
        dist = np.clip(dist, -clip_max, clip_max)
    max_abs = np.max(np.abs(dist))
    if max_abs > 0:
        dist = dist / max_abs
    return dist.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Derive distance map or SDF from a segmentation label volume."
    )
    parser.add_argument("input", type=str, help="Path to input label volume (HDF5 or TIFF)")
    parser.add_argument("output", type=str, help="Path to output distance volume")
    parser.add_argument(
        "--type", choices=["edt", "sdf"], default="edt",
        help="Distance type: edt (Euclidean Distance Transform) or sdf (Signed Distance Field)"
    )
    parser.add_argument(
        "--mode", choices=["3d", "2d"], default="3d",
        help="3D volumetric or 2D slice-by-slice computation (default: 3d)"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize output to [-1, 1]"
    )
    parser.add_argument(
        "--clip_max", type=float, default=None,
        help="Clip absolute distance values before normalizing"
    )
    parser.add_argument(
        "--hdf5_path", type=str, default="/data",
        help="Internal HDF5 dataset path (default: /data)"
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)

    from volume_segmantics.utilities.base_data_utils import get_numpy_from_path

    input_path = Path(args.input)
    labels, _ = get_numpy_from_path(input_path, internal_path=args.hdf5_path)
    logging.info(f"Loaded labels: shape={labels.shape}, dtype={labels.dtype}")

    if args.type == "edt":
        if args.mode == "3d":
            dist = derive_edt_volume(labels)
        else:
            dist = derive_edt_2d_slicewise(labels)
    else:
        if args.mode == "3d":
            dist = derive_sdf_volume(labels)
        else:
            dist = derive_sdf_2d_slicewise(labels)

    if args.normalize:
        dist = normalize_distance_map(dist, clip_max=args.clip_max)

    logging.info(f"Distance map: shape={dist.shape}, min={dist.min():.4f}, max={dist.max():.4f}")

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(str(output_path), "w") as f:
            f.create_dataset(args.hdf5_path, data=dist, compression="gzip")
    elif suffix in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(str(output_path), dist)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    logging.info(f"Saved distance labels to {output_path}")


if __name__ == "__main__":
    main()
