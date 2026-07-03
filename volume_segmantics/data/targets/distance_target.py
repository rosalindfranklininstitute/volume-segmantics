"""Distance target generator (2D, per-slice).

Output: ``(H, W) float32`` **per-object normalised** foreground distance
transform — for each connected foreground object, the Euclidean distance
to the nearest background pixel, divided by that object's own maximum, so
every object peaks at ``1.0`` at its centre and decays to ``0`` at its
boundary. Background is ``0``.

This orientation + normalisation is what the ``distance_watershed``
instance producer needs: it seeds the watershed at local maxima inside the
foreground, so the map must have one clean maximum per object. Two reasons
for the per-object normalisation:


:class:`DistanceHead` is a 1-channel identity-output head fed straight
into L1 / L2 loss; this generator emits the float32 target directly.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage as _ndi


def derive_distance_target_2d(
    label_slice: np.ndarray,
    *,
    distance_transform: str = "edt",
) -> np.ndarray:
    """Derive a per-object normalised foreground distance map from a 2D slice.

    Parameters
    ----------
    label_slice
        2D integer label array. Foreground = any non-zero class. Connected
        components of the foreground are treated as the objects to normalise
        over (for non-touching objects this is exactly the instances).
    distance_transform
        Currently only ``"edt"`` is supported. The kwarg exists so
        future b4+ releases can add alternatives without changing the
        registry signature.

    Returns
    -------
    np.ndarray
        ``(H, W) float32`` map in ``[0, 1]``: ``1.0`` at each object's centre,
        decaying to ``0`` at its boundary; ``0`` on background.
    """
    if label_slice.ndim != 2:
        raise ValueError(
            f"derive_distance_target_2d expects a 2D slice; got shape "
            f"{label_slice.shape}"
        )
    if distance_transform != "edt":
        raise ValueError(
            f"distance_transform must be 'edt' in v0.4.0b3; "
            f"got {distance_transform!r}"
        )
    binary = (label_slice > 0).astype(np.uint8)
    edt = _ndi.distance_transform_edt(binary).astype(np.float32)
    # Label connected foreground objects and normalise each object's EDT by
    # its own maximum, so every object peaks at 1.0 at its centre.
    labeled, n_objects = _ndi.label(binary)
    if n_objects == 0:
        return edt  # all background -> all zeros
    # Per-object maxima, indexed 1..n_objects; prepend a background dummy so
    # the array can be indexed directly by the label image.
    obj_max = _ndi.maximum(edt, labeled, index=np.arange(1, n_objects + 1))
    obj_max = np.concatenate([[1.0], np.asarray(obj_max, dtype=np.float32)])
    obj_max[obj_max == 0] = 1.0  # guard 1-pixel/degenerate objects
    out = edt / obj_max[labeled]
    out[labeled == 0] = 0.0
    return out.astype(np.float32)


def make_distance_target_generator(
    *,
    distance_transform: str = "edt",
    **_extra: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory for the registry."""
    def _gen(label_slice: np.ndarray) -> np.ndarray:
        return derive_distance_target_2d(
            label_slice, distance_transform=distance_transform,
        )
    return _gen


__all__ = [
    "derive_distance_target_2d",
    "make_distance_target_generator",
]
