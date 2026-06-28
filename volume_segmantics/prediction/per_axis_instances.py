"""Per-axis 2D instance producers.

A :class:`PerAxisInstanceProducer` runs a 2D instance-segmentation
algorithm per slice along each requested axis (``xy`` / ``xz`` /
``yz``) and emits the per-axis instance maps that
:class:`USegment3DAssembler` consumes.

"""

from __future__ import annotations

import logging
from typing import (
    Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple,
    Type, runtime_checkable,
)

import numpy as np
from scipy import ndimage as _ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from volume_segmantics.inference.instance_assembly import (
    InstanceAssemblerInputError,
    PredictionBundle,
)


logger = logging.getLogger(__name__)


VALID_AXES: Tuple[str, ...] = ("xy", "xz", "yz")


#  Helpers 


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Map labels to ``0..N`` keeping ``0`` as background, ``1..N`` dense."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size == 0:
        return np.zeros_like(labels, dtype=np.uint32)
    if unique[0] != 0:
        unique = np.concatenate([[0], unique])
    lut = np.zeros(int(unique.max()) + 1, dtype=np.uint32)
    for new_id, old_id in enumerate(unique):
        lut[int(old_id)] = new_id
    return lut[labels].astype(np.uint32)


def _foreground_mask_from_semantic(
    semantic_argmax: np.ndarray,
    foreground_class_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Boolean foreground mask from ``semantic_argmax``.

    When ``foreground_class_ids`` is empty / ``None``, any non-zero
    class is foreground. Otherwise only voxels whose class is in the
    list are foreground.
    """
    if foreground_class_ids is None or len(foreground_class_ids) == 0:
        return semantic_argmax > 0
    fg = np.zeros_like(semantic_argmax, dtype=bool)
    for cid in foreground_class_ids:
        fg |= (semantic_argmax == int(cid))
    return fg


def _iterate_slices_along_axis(
    semantic_argmax: np.ndarray, axis_name: str,
):
    """Yield 2D slices of the ``(Z, Y, X)`` semantic_argmax along the axis.

    Yields ``(slice_index, slice_shape_2d)`` tuples that call sites
    can use to index parallel arrays (e.g. distance_map). The semantic
    array stays in (Z, Y, X) layout; the producers reorient the
    output stack before returning.
    """
    if axis_name == "xy":
        n = semantic_argmax.shape[0]
        for z in range(n):
            yield z
    elif axis_name == "xz":
        n = semantic_argmax.shape[1]
        for y in range(n):
            yield y
    elif axis_name == "yz":
        n = semantic_argmax.shape[2]
        for x in range(n):
            yield x
    else:
        raise ValueError(f"unknown axis {axis_name!r}")


def _slice_along_axis(
    arr: np.ndarray, axis_name: str, idx: int,
) -> np.ndarray:
    """Extract the 2D slice at ``idx`` along the given axis."""
    if axis_name == "xy":
        return arr[idx, :, :]
    if axis_name == "xz":
        return arr[:, idx, :]
    if axis_name == "yz":
        return arr[:, :, idx]
    raise ValueError(f"unknown axis {axis_name!r}")


def _orient_output_for_axis(
    stacked: np.ndarray, axis_name: str,
) -> np.ndarray:
    """Reorient a stack collected along an axis to the v0.5 convention.

    The producer collects 2D slices along the slice axis, stacks them
    along axis 0. The convention each axis expects:

    * ``xy``: ``(Z, Y, X)`` — already correct (slice axis is Z).
    * ``xz``: ``(Y, Z, X)`` — slice axis is Y; (Y, Y_slice, X) ->
      (Y_slice, Y, X)? No — for axis xz, we slice along Y giving
      (n_y, Z, X) shaped slice-axis-first; v0.5 wants (Y, Z, X)
      which is the same.
    * ``yz``: ``(X, Z, Y)`` — slice along X gives (n_x, Z, Y).
    """
    # When iterating along axis_name, the stacked array is already
    # in slice-axis-first order. v0.5's conventions:
    #   xy -> (Z, Y, X) — slice along Z.
    #   xz -> (Y, Z, X) — slice along Y.
    #   yz -> (X, Z, Y) — slice along X.
    # Our slices are 2D in their natural orientation; for xz the slice
    # is (Z, X) so stacking along axis 0 gives (Y, Z, X) — already
    # matches. For yz, slice is (Z, Y) so stacking along axis 0 gives
    # (X, Z, Y) — also matches.
    return stacked


def _check_bundle_per_axis_input(
    bundle: PredictionBundle, axes: Sequence[str], required_fields: Tuple[str, ...],
) -> None:
    bundle.require(required_fields)
    bad = [a for a in axes if a not in VALID_AXES]
    if bad:
        raise InstanceAssemblerInputError(
            f"per-axis producer: unknown axis names {bad}; "
            f"valid: {list(VALID_AXES)}"
        )


#  Protocol 


@runtime_checkable
class PerAxisInstanceProducer(Protocol):
    """Produces per-axis 2D instance maps from a :class:`PredictionBundle`.

    Attributes
    ----------
    name
        Registered producer name (e.g. ``"distance_watershed"``).
    required_bundle_fields
        Bundle fields the producer reads. Validated via
        :meth:`PredictionBundle.require`.
    """

    name: str
    required_bundle_fields: Tuple[str, ...]

    def produce(
        self,
        bundle: PredictionBundle,
        axes: Sequence[str],
        params: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Return ``{axis: (3D array of uint32 instance ids)}``.

        Output shape per axis follows the v0.5 convention pinned on
        :class:`PredictionBundle.per_axis_instances`.
        """
        ...


#  DistanceWatershedSliceProducer 


class DistanceWatershedSliceProducer:
    """Per-slice watershed seeded by local maxima of the distance head.

    Per slice along each requested axis:

    1. Foreground mask = ``semantic_argmax`` ∈ ``foreground_class_ids``
       (default = "any non-zero class").
    2. Markers = local maxima of the slice's ``distance_map`` with
       ``min_distance >= peak_min_distance``, filtered to the
       foreground mask.
    3. ``skimage.segmentation.watershed(-distance_slice, markers,
       mask=fg_mask)``.
    4. :func:`_relabel_consecutive` to pack IDs.

    Required bundle fields: ``semantic_argmax`` + ``distance_map``.
    """

    name: str = "distance_watershed"
    required_bundle_fields: Tuple[str, ...] = (
        "semantic_argmax", "distance_map",
    )

    def produce(
        self,
        bundle: PredictionBundle,
        axes: Sequence[str],
        params: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        _check_bundle_per_axis_input(bundle, axes, self.required_bundle_fields)
        peak_min_distance = int(params.get("peak_min_distance", 5))
        fg_class_ids = params.get("foreground_class_ids", None) or None
        if peak_min_distance < 1:
            raise ValueError(
                f"distance_watershed: peak_min_distance must be >= 1; "
                f"got {peak_min_distance}"
            )

        sem = np.asarray(bundle.semantic_argmax)
        dist = np.asarray(bundle.distance_map)
        # distance_map is (1, Z, Y, X) or (Z, Y, X) — squeeze leading 1.
        if dist.ndim == 4 and dist.shape[0] == 1:
            dist = dist[0]
        if dist.shape != sem.shape:
            raise ValueError(
                f"distance_watershed: distance_map shape {dist.shape} "
                f"does not match semantic_argmax shape {sem.shape}"
            )

        fg_mask = _foreground_mask_from_semantic(sem, fg_class_ids)

        out: Dict[str, np.ndarray] = {}
        for axis in axes:
            slabs = []
            for idx in _iterate_slices_along_axis(sem, axis):
                fg_slice = _slice_along_axis(fg_mask, axis, idx)
                dist_slice = _slice_along_axis(dist, axis, idx)
                inst_slice = self._watershed_one_slice(
                    fg_slice, dist_slice, peak_min_distance,
                )
                slabs.append(inst_slice)
            stacked = np.stack(slabs, axis=0).astype(np.uint32)
            stacked = _orient_output_for_axis(stacked, axis)
            out[axis] = _relabel_consecutive(stacked)
        return out

    @staticmethod
    def _watershed_one_slice(
        fg_mask: np.ndarray,
        dist_slice: np.ndarray,
        peak_min_distance: int,
    ) -> np.ndarray:
        """Run watershed on a single 2D slice."""
        if not fg_mask.any():
            return np.zeros_like(fg_mask, dtype=np.uint32)
        # Local maxima of the distance map, masked to FG.
        peaks = peak_local_max(
            dist_slice,
            min_distance=peak_min_distance,
            labels=fg_mask.astype(np.uint8),
        )
        if peaks.shape[0] == 0:
            # No peaks -> fall back to single-component watershed seed
            # at the FG centroid.
            ys, xs = np.where(fg_mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            peaks = np.array([[cy, cx]], dtype=int)
        markers = np.zeros(fg_mask.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
        labels = watershed(-dist_slice, markers=markers, mask=fg_mask)
        return labels.astype(np.uint32)


#  SemanticCcSliceProducer 


class SemanticCcSliceProducer:
    """Per-slice connected-components on the semantic foreground mask.

    Cheap fallback for projects without a ``distance`` head, or for
    cases where objects are already well-separated by the semantic
    argmax. Per slice along each requested axis:

    1. Foreground mask = ``semantic_argmax`` ∈ ``foreground_class_ids``.
    2. :func:`scipy.ndimage.label` with the configured ``connectivity``.
    3. :func:`_relabel_consecutive`.

    Required bundle fields: ``semantic_argmax``.

    Connectivity options (``params['connectivity']``):
    * ``1`` (default) — 4-connected (face neighbours only).
    * ``2`` — 8-connected (face + diagonal).
    """

    name: str = "semantic_cc"
    required_bundle_fields: Tuple[str, ...] = ("semantic_argmax",)

    def produce(
        self,
        bundle: PredictionBundle,
        axes: Sequence[str],
        params: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        _check_bundle_per_axis_input(bundle, axes, self.required_bundle_fields)
        connectivity = int(params.get("connectivity", 1))
        if connectivity not in (1, 2):
            raise ValueError(
                f"semantic_cc: connectivity must be 1 (4-conn) or 2 "
                f"(8-conn); got {connectivity}"
            )
        fg_class_ids = params.get("foreground_class_ids", None) or None

        sem = np.asarray(bundle.semantic_argmax)
        fg_mask = _foreground_mask_from_semantic(sem, fg_class_ids)
        struct = _ndi.generate_binary_structure(2, connectivity)

        out: Dict[str, np.ndarray] = {}
        for axis in axes:
            slabs = []
            for idx in _iterate_slices_along_axis(sem, axis):
                fg_slice = _slice_along_axis(fg_mask, axis, idx)
                lbl, _n = _ndi.label(fg_slice, structure=struct)
                slabs.append(lbl.astype(np.uint32))
            stacked = np.stack(slabs, axis=0).astype(np.uint32)
            stacked = _orient_output_for_axis(stacked, axis)
            out[axis] = _relabel_consecutive(stacked)
        return out


#  Registry + auto-selection 


_PRODUCERS: Dict[str, Type] = {}


def register_producer(name: str, cls: Type) -> None:
    if name in _PRODUCERS:
        raise KeyError(
            f"per-axis producer {name!r} already registered "
            f"(existing: {_PRODUCERS[name].__name__})"
        )
    _PRODUCERS[name] = cls


def get_producer(name: str) -> Type:
    if name not in _PRODUCERS:
        raise KeyError(
            f"unknown per-axis producer {name!r}; known: "
            f"{sorted(_PRODUCERS)}"
        )
    return _PRODUCERS[name]


def list_producers() -> List[str]:
    return sorted(_PRODUCERS)


def select_producer_name(
    requested: Optional[str],
    *,
    enabled_heads: Sequence[str],
) -> str:
    """Resolve the producer name from the pipeline.yaml + enabled heads.

    * ``requested`` set -> use it as-is (must be in :data:`_PRODUCERS`).
    * ``requested`` is ``None`` -> default to ``distance_watershed`` if
      the ``distance`` head is enabled, else ``semantic_cc``.
    """
    if requested is not None:
        if requested not in _PRODUCERS:
            raise ValueError(
                f"per-axis producer {requested!r} not registered; "
                f"known: {sorted(_PRODUCERS)}"
            )
        return requested

    if "distance" in enabled_heads:
        return "distance_watershed"
    return "semantic_cc"


#  Import-time registration 


for _name, _cls in (
    ("distance_watershed", DistanceWatershedSliceProducer),
    ("semantic_cc",        SemanticCcSliceProducer),
):
    try:
        register_producer(_name, _cls)
    except KeyError:
        pass


__all__ = [
    "DistanceWatershedSliceProducer",
    "PerAxisInstanceProducer",
    "SemanticCcSliceProducer",
    "VALID_AXES",
    "get_producer",
    "list_producers",
    "register_producer",
    "select_producer_name",
]
