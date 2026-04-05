"""Derive binary boundary labels from a segmentation label volume.

Computes morphological gradient (dilation - erosion) to extract boundaries
of labelled regions. Supports both 3D volumetric and 2D slice-wise modes.

Usage:
    python -m volume_segmantics.utilities.derive_boundary_labels \\
        input.h5 output.h5 --width 3 --mode 3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage


def derive_boundary_volume(labels: np.ndarray, width: int = 3) -> np.ndarray:
    """Derive boundary mask from a 3D label volume using morphological gradient.

    For binary labels: gradient = dilation(fg) - erosion(fg).
    For multi-class: per-class gradient, then max across all classes.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X) with integer class labels.
    width : int
        Boundary width in voxels (structuring element iterations).

    Returns
    -------
    np.ndarray
        Binary boundary mask (uint8, 0=interior, 1=boundary).
    """
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    struct = ndimage.iterate_structure(struct, width)

    unique = np.unique(labels)
    if len(unique) <= 2:
        # Binary case
        binary = (labels > 0).astype(np.uint8)
        dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
        eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
        return (dilated - eroded).astype(np.uint8)

    # Multi-class: per-class gradient, then max
    boundary = np.zeros_like(labels, dtype=np.uint8)
    for cls in unique:
        if cls == 0:
            continue
        binary = (labels == cls).astype(np.uint8)
        dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
        eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
        boundary = np.maximum(boundary, (dilated - eroded).astype(np.uint8))
    return boundary


def derive_boundary_2d_slicewise(labels: np.ndarray, width: int = 3) -> np.ndarray:
    """Derive boundary mask slice-by-slice (2D) for anisotropic data.

    Parameters
    ----------
    labels : np.ndarray
        3D label volume (Z, Y, X).
    width : int
        Boundary width in pixels.

    Returns
    -------
    np.ndarray
        Binary boundary mask (uint8).
    """
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
    struct = ndimage.iterate_structure(struct, width)

    boundary = np.zeros_like(labels, dtype=np.uint8)
    unique = np.unique(labels)

    for z in range(labels.shape[0]):
        sl = labels[z]
        if len(unique) <= 2:
            binary = (sl > 0).astype(np.uint8)
            dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
            eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
            boundary[z] = (dilated - eroded).astype(np.uint8)
        else:
            for cls in unique:
                if cls == 0:
                    continue
                binary = (sl == cls).astype(np.uint8)
                dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
                eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
                boundary[z] = np.maximum(boundary[z], (dilated - eroded).astype(np.uint8))
    return boundary


def main():
    parser = argparse.ArgumentParser(
        description="Derive binary boundary labels from a segmentation label volume."
    )
    parser.add_argument("input", type=str, help="Path to input label volume (HDF5 or TIFF)")
    parser.add_argument("output", type=str, help="Path to output boundary volume")
    parser.add_argument("--width", type=int, default=3, help="Boundary width in voxels (default: 3)")
    parser.add_argument(
        "--mode", choices=["3d", "2d"], default="3d",
        help="3D volumetric or 2D slice-by-slice boundary (default: 3d)"
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
    logging.info(f"Loaded labels: shape={labels.shape}, dtype={labels.dtype}, unique={np.unique(labels)}")

    if args.mode == "3d":
        boundary = derive_boundary_volume(labels, width=args.width)
    else:
        boundary = derive_boundary_2d_slicewise(labels, width=args.width)

    logging.info(f"Boundary volume: shape={boundary.shape}, boundary pixels={boundary.sum()}")

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(str(output_path), "w") as f:
            f.create_dataset(args.hdf5_path, data=boundary, compression="gzip")
    elif suffix in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(str(output_path), boundary)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    logging.info(f"Saved boundary labels to {output_path}")


if __name__ == "__main__":
    main()
