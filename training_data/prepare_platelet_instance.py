#!/usr/bin/env python
"""Prepare the platelet-EM 50-slice cell volume for the instance-seg pipeline.

The platelet-EM "rgba" archive stores labels as colour images: each cell is a
distinct RGBA colour and the background is ``(0, 0, 0, 0)``. The volume-segmantics
pipeline needs integer label volumes, so this script decodes
``labels-instance/50-instance-cell.tif`` into two single-channel volumes next to
the dataset:

    platelet_50_cell_SEMANTIC.tif   uint8   0 = background, 1 = cell foreground
    platelet_50_cell_INSTANCE.tif   uint16  0 = background, 1..N = per-cell ids

The grayscale image ``images/50-images.tif`` is already usable as-is and is not
copied.

* The SEMANTIC volume is the training label (the semantic head learns cell vs.
  background; instances are recovered at predict time by the distance-watershed
  assembler).
* The INSTANCE volume is the ground truth for the instance-count assertions.

Run it after downloading the dataset (see download_platelet_em.py)::

    python training_data/prepare_platelet_instance.py
    python training_data/prepare_platelet_instance.py --force   # rebuild outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

# Source (relative to this script's directory) and output names.
INSTANCE_RGBA = Path("images_and_labels_rgba/platelet-em/labels-instance/50-instance-cell.tif")
IMAGE_TIF = Path("images_and_labels_rgba/platelet-em/images/50-images.tif")
SEMANTIC_OUT = Path("platelet_50_cell_SEMANTIC.tif")
INSTANCE_OUT = Path("platelet_50_cell_INSTANCE.tif")

# Background colour in the rgba labels.
_BACKGROUND_RGBA = (0, 0, 0, 0)


def _decode_rgba_instances(rgba: np.ndarray) -> np.ndarray:
    """Map each unique RGBA colour to an integer id (background -> 0).

    ``rgba`` is ``(Z, Y, X, 4)``; returns ``(Z, Y, X)`` uint16 with background 0
    and cells numbered 1..N in order of first appearance.
    """
    if rgba.ndim != 4 or rgba.shape[-1] != 4:
        raise ValueError(f"expected an (Z, Y, X, 4) RGBA volume; got shape {rgba.shape}")

    flat = rgba.reshape(-1, 4)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Remap colour indices so the background colour becomes 0 and the rest are
    # a contiguous 1..N (skipping the background).
    bg = np.array(_BACKGROUND_RGBA, dtype=colours.dtype)
    bg_matches = np.where((colours == bg).all(axis=1))[0]
    bg_index = int(bg_matches[0]) if bg_matches.size else -1

    new_ids = np.zeros(len(colours), dtype=np.uint32)
    next_id = 1
    for idx in range(len(colours)):
        if idx == bg_index:
            new_ids[idx] = 0
        else:
            new_ids[idx] = next_id
            next_id += 1

    labels = new_ids[inverse].reshape(rgba.shape[:3])
    n_instances = int(labels.max())
    if n_instances > np.iinfo(np.uint16).max:
        raise ValueError(f"too many instances ({n_instances}) for uint16 output")
    return labels.astype(np.uint16)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=here,
        help="Directory containing the dataset (default: this script's directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the output volumes even if they already exist.",
    )
    args = parser.parse_args()

    dest: Path = args.dest.resolve()
    src = dest / INSTANCE_RGBA
    image = dest / IMAGE_TIF
    semantic_out = dest / SEMANTIC_OUT
    instance_out = dest / INSTANCE_OUT

    if not src.exists():
        print(
            f"ERROR: {src} not found.\n"
            "Download the dataset first: python training_data/download_platelet_em.py",
            file=sys.stderr,
        )
        return 1

    if semantic_out.exists() and instance_out.exists() and not args.force:
        print(
            f"Outputs already present:\n  {semantic_out}\n  {instance_out}\n"
            "Nothing to do (pass --force to rebuild)."
        )
        return 0

    print(f"Reading {src}")
    rgba = tifffile.imread(str(src))
    instances = _decode_rgba_instances(rgba)
    semantic = (instances > 0).astype(np.uint8)

    n_instances = int(instances.max())
    fg_fraction = float(semantic.mean())
    print(f"  decoded {n_instances} cell instances; foreground = {100 * fg_fraction:.1f}% of voxels")

    tifffile.imwrite(str(instance_out), instances)
    tifffile.imwrite(str(semantic_out), semantic)
    print(f"Wrote {instance_out}  (uint16, {n_instances} instances)")
    print(f"Wrote {semantic_out}  (uint8, binary foreground)")
    if image.exists():
        print(f"Training image (use as --data): {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
