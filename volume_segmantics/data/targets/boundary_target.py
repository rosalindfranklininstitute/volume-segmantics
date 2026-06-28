"""Boundary target generator (2D, per-slice).

Wraps the morphological-gradient boundary derivation from
:mod:`volume_segmantics.utilities.derive_boundary_labels` for the
single-slice case the dataset consumes per sample.

Output: ``(H, W) float32`` binary boundary mask in ``{0.0, 1.0}``. The
pipeline version :class:`BoundaryHead` is a 1-channel raw-logits head + the loss
function applies sigmoid internally; this generator emits ``float32``
in ``{0, 1}`` so loss computations don't repeat a uint8 -> float cast.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage


def derive_boundary_target_2d(
    label_slice: np.ndarray,
    *,
    width: int = 3,
) -> np.ndarray:
    """Derive a binary boundary mask from a 2D label slice.

    Same morphological-gradient logic as
    :func:`volume_segmantics.utilities.derive_boundary_labels.derive_boundary_2d_slicewise`,
    but on a single ``(H, W)`` array (not a stack).

    Parameters
    ----------
    label_slice
        2D integer label array. Background is class ``0``; any non-zero
        value is foreground. Multi-class labels are handled by taking
        the per-class boundary then max-merging.
    width
        Boundary width in pixels (the structuring element is
        4-connected and dilated ``width`` times).

    Returns
    -------
    np.ndarray
        ``(H, W) float32`` boundary mask in ``{0.0, 1.0}``.
    """
    if label_slice.ndim != 2:
        raise ValueError(
            f"derive_boundary_target_2d expects a 2D slice; got shape "
            f"{label_slice.shape}"
        )
    if width < 1:
        raise ValueError(f"width must be >= 1; got {width}")

    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, width)

    unique = np.unique(label_slice)
    if len(unique) <= 2:
        binary = (label_slice > 0).astype(np.uint8)
        dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
        eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
        return (dilated - eroded).astype(np.float32)

    boundary = np.zeros_like(label_slice, dtype=np.uint8)
    for cls in unique:
        if cls == 0:
            continue
        binary = (label_slice == cls).astype(np.uint8)
        dilated = ndimage.binary_dilation(binary, structure=struct).astype(np.uint8)
        eroded = ndimage.binary_erosion(binary, structure=struct).astype(np.uint8)
        boundary = np.maximum(boundary, (dilated - eroded).astype(np.uint8))
    return boundary.astype(np.float32)


def make_boundary_target_generator(
    *,
    width: int = 3,
    **_extra: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory for the registry. Returns a callable closing over ``width``.

    Pipeline mode reads ``width`` (and any other generator knobs) from
    ``pipeline.yaml::targets.boundary``.
    """
    def _gen(label_slice: np.ndarray) -> np.ndarray:
        return derive_boundary_target_2d(label_slice, width=width)
    return _gen


__all__ = [
    "derive_boundary_target_2d",
    "make_boundary_target_generator",
]
