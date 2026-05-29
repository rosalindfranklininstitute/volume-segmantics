"""Signed distance map target generator (2D, per-slice).


Sign convention
---------------
Positive **inside** foreground; negative **outside**; zero on the
boundary. The b3 :class:`SDMHead` emits ``tanh(...)`` so its output 
is in ``[-1, 1]`` with positive-inside; this generator's output matches that convention so L1 / L2
loss on raw values lines up with the head's tanh-bounded output.

Output range
------------
Distances are divided by ``d_clip`` and clipped to ``[-1, 1]``:

* Voxel deep inside foreground at distance ``> d_clip`` → ``+1.0``.
* Voxel deep in background at distance ``> d_clip`` → ``-1.0``.
* Boundary → ``0.0``.

The default ``d_clip = 10.0`` voxels matches the head's default and
is appropriate for object scales in the 5–50 voxel range. Tune via
``pipeline.yaml::heads.sdm.d_clip`` (head-side; the head reads it but
doesn't apply it — the *target* is what gets clipped).

Variants
--------
* ``binary`` (default) — 1-channel SDM of the union foreground.
* ``per_class`` — ``num_classes - 1`` channels; one SDM per non-bg class.

Output shape
------------
* ``binary`` → ``(H, W) float32``.
* ``per_class`` → ``(num_classes - 1, H, W) float32`` (channel-first).

Loss path
---------
The b3 :class:`PipelineMultiTaskLossCalculator` always feeds head
output and target with matching channel counts to the registered loss
function. ``per_class`` SDM emits ``(C-1, H, W)`` targets that line up
with the head's ``(C-1, H, W)`` channel-wise output 1:1.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.ndimage import distance_transform_edt


def derive_sdm_target_2d(
    label_slice: np.ndarray,
    *,
    variant: str = "binary",
    d_clip: float = 10.0,
    num_classes: int | None = None,
    distance_transform: str = "edt",
) -> np.ndarray:
    """Derive a positive-inside signed distance map from a 2D label slice.

    Parameters
    ----------
    label_slice
        ``(H, W)`` integer label array. Background = 0.
    variant
        ``"binary"`` (default) or ``"per_class"``. Per §1.2.
    d_clip
        Distance scale in pixels. Output values outside
        ``[-d_clip, d_clip]`` (in raw distance units) are clipped to
        ``±1`` after normalisation.
    num_classes
        Required only when ``variant="per_class"``. Output channel
        count is ``num_classes - 1`` (background excluded).
    distance_transform
        Only ``"edt"`` is supported in b3.

    Returns
    -------
    np.ndarray
        ``binary`` → ``(H, W) float32`` in ``[-1, 1]``.
        ``per_class`` → ``(num_classes - 1, H, W) float32`` in ``[-1, 1]``.
    """
    if label_slice.ndim != 2:
        raise ValueError(
            f"derive_sdm_target_2d expects a 2D slice; got shape "
            f"{label_slice.shape}"
        )
    if variant not in ("binary", "per_class"):
        raise ValueError(
            f"variant must be 'binary' or 'per_class'; got {variant!r}"
        )
    if d_clip <= 0.0:
        raise ValueError(f"d_clip must be positive; got {d_clip}")
    if distance_transform != "edt":
        raise ValueError(
            f"distance_transform must be 'edt' in v0.4.0b3; "
            f"got {distance_transform!r}"
        )

    if variant == "binary":
        return _sdm_2d_binary(label_slice, d_clip)

    if num_classes is None or num_classes < 2:
        raise ValueError(
            f"variant='per_class' requires num_classes >= 2 (got "
            f"{num_classes}); output has num_classes - 1 channels"
        )

    out = np.zeros(
        (num_classes - 1, *label_slice.shape), dtype=np.float32,
    )
    for cls in range(1, num_classes):  # skip background = 0
        out[cls - 1] = _sdm_2d_binary(label_slice == cls, d_clip)
    return out


def _sdm_2d_binary(
    binary_slice_or_labels: np.ndarray,
    d_clip: float,
) -> np.ndarray:
    """Inner: positive-inside SDM on a binary mask or label slice.

    Accepts either a boolean / 0-1 mask **or** an integer label slice
    (any non-zero is foreground). Returns ``(H, W) float32`` in
    ``[-1, 1]``.
    """
    binary = (binary_slice_or_labels > 0).astype(np.uint8)
    edt_outside = distance_transform_edt(1 - binary)
    edt_inside = distance_transform_edt(binary)
    # Positive-inside convention: subtract outside from inside.
    # (v0.4's derive_sdf_volume does outside - inside; we flip the sign.)
    sdf = edt_inside - edt_outside
    sdf = sdf / d_clip
    sdf = np.clip(sdf, -1.0, 1.0)
    return sdf.astype(np.float32)


def make_sdm_target_generator(
    *,
    variant: str = "binary",
    d_clip: float = 10.0,
    num_classes: int | None = None,
    distance_transform: str = "edt",
    **_extra: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory for the registry.

    Pipeline mode reads ``variant`` from ``HeadConfig.extra`` and
    ``num_classes`` from the trainer's resolved class count;
    ``d_clip`` reads from ``HeadConfig.extra`` as well (per §5 schema).
    """
    def _gen(label_slice: np.ndarray) -> np.ndarray:
        return derive_sdm_target_2d(
            label_slice,
            variant=variant,
            d_clip=d_clip,
            num_classes=num_classes,
            distance_transform=distance_transform,
        )
    return _gen


__all__ = [
    "derive_sdm_target_2d",
    "make_sdm_target_generator",
]
