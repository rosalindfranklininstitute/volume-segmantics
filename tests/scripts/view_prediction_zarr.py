#!/usr/bin/env python
"""Open a pipeline ``prediction.zarr`` in napari for debugging.

napari's built-in zarr reader fails on these stores because they mix arrays
(``teacher_argmax``, ``instance_labels``, ``*_map``) with a nested *group*
(``per_axis_instances``) at the top level -- it tries to treat the group as an
array and raises ``'Group' object has no attribute 'size'``.

This script walks the store explicitly, adds each array as the right kind of
layer (integer label volumes as ``labels``, float maps as ``image``), squeezes
leading singleton channel axes, and launches napari.

Run it from the env that has napari (e.g. ``napari_simple``)::

    python tests/scripts/view_prediction_zarr.py path/to/prediction.zarr
    python tests/scripts/view_prediction_zarr.py path/to/prediction.zarr --image 50-images.tif

``--image`` optionally overlays the source grayscale volume for context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

# Arrays whose values are integer ids/classes -> shown as napari "labels".
_LABEL_NAME_HINTS = ("instance", "argmax", "per_axis", "labels", "watershed", "_cc")

# The per-axis instance maps are stored in their own orientation (see
# volume_segmantics.prediction.per_axis_instances._orient_output_for_axis):
#   xy -> (Z, Y, X)   xz -> (Y, Z, X)   yz -> (X, Z, Y)
# Reorient them to the canonical (Z, Y, X) so every layer aligns at 50 slices
# instead of napari widening the slice slider to the 800-long per-axis stacks.
_PER_AXIS_TO_ZYX = {
    "per_axis_instances/xy": None,        # already (Z, Y, X)
    "per_axis_instances/xz": (1, 0, 2),   # (Y, Z, X) -> (Z, Y, X)
    "per_axis_instances/yz": (1, 2, 0),   # (X, Z, Y) -> (Z, Y, X)
}


def _to_canonical_zyx(name: str, data: np.ndarray) -> np.ndarray:
    """Reorient a per-axis instance map to (Z, Y, X); pass others through."""
    order = _PER_AXIS_TO_ZYX.get(name, "passthrough")
    if order not in (None, "passthrough") and data.ndim == 3:
        return np.transpose(data, order)
    return data


def _walk_arrays(group, prefix: str = ""):
    """Yield ``(name, zarr_array)`` for every array in the store, recursively."""
    # zarr v2/v3 compatible iteration.
    array_keys = list(group.array_keys()) if hasattr(group, "array_keys") else [
        k for k, v in group.items() if hasattr(v, "shape")
    ]
    group_keys = list(group.group_keys()) if hasattr(group, "group_keys") else [
        k for k, v in group.items() if not hasattr(v, "shape")
    ]
    for key in array_keys:
        yield f"{prefix}{key}", group[key]
    for key in group_keys:
        yield from _walk_arrays(group[key], prefix=f"{prefix}{key}/")


def _is_label_layer(name: str, arr: np.ndarray) -> bool:
    lowered = name.lower()
    if any(hint in lowered for hint in _LABEL_NAME_HINTS):
        return True
    # Integer dtype with few distinct small values is label-like.
    return np.issubdtype(arr.dtype, np.integer)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("zarr_path", type=Path, help="Path to a prediction.zarr store.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional source grayscale volume (TIFF/H5) to overlay for context.",
    )
    args = parser.parse_args()

    if not args.zarr_path.exists():
        print(f"ERROR: {args.zarr_path} does not exist.", file=sys.stderr)
        return 1

    import napari

    viewer = napari.Viewer()

    if args.image is not None and args.image.exists():
        suffix = args.image.suffix.lower()
        if suffix in (".tif", ".tiff"):
            import tifffile
            src = tifffile.imread(str(args.image))
        else:
            import h5py
            with h5py.File(args.image, "r") as f:
                key = "data" if "data" in f else list(f.keys())[0]
                src = np.asarray(f[key][:])
        viewer.add_image(np.squeeze(src), name="source-image", colormap="gray", blending="additive")

    root = zarr.open(str(args.zarr_path), mode="r")
    added = 0
    for name, arr in _walk_arrays(root):
        data = _to_canonical_zyx(name, np.squeeze(np.asarray(arr[:])))
        if _is_label_layer(name, data):
            viewer.add_labels(data.astype(np.int64), name=name)
            kind = "labels"
        else:
            viewer.add_image(data, name=name, blending="additive")
            kind = "image"
        n_inst = (int(np.unique(data).size) - 1) if kind == "labels" else None
        extra = f", instances={n_inst}" if n_inst is not None else ""
        print(f"  + {kind:6s} {name}  shape={data.shape} dtype={data.dtype}{extra}")
        added += 1

    if not added:
        print("No arrays found in the store.", file=sys.stderr)
        return 1

    print(f"Loaded {added} layers from {args.zarr_path}. Launching napari...")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
