"""Distance target generator (2D, per-slice).

Output: ``(H, W) float32`` Euclidean distance from the nearest
foreground pixel. Zero on FG, positive in BG.
:class:`DistanceHead` is a 1-channel identity-output head fed straight
into L1 / L2 loss so this generator emits float32 distances directly
(no normalisation).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.ndimage import distance_transform_edt


def derive_distance_target_2d(
    label_slice: np.ndarray,
    *,
    distance_transform: str = "edt",
) -> np.ndarray:
    """Derive an EDT distance map from a 2D label slice.


    Parameters
    ----------
    label_slice
        2D integer label array. Foreground = any non-zero class.
    distance_transform
        Currently only ``"edt"`` is supported. The kwarg exists so
        future b4+ releases can add alternatives without changing the
        registry signature.

    Returns
    -------
    np.ndarray
        ``(H, W) float32`` distance map. Zero on FG, positive in BG.
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
    return distance_transform_edt(1 - binary).astype(np.float32)


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
