"""USegment3DAssembler adapter.

Adapts DanuserLab's `u-Segment3D <https://github.com/DanuserLab/u-segment3D>`_
to the :class:`InstanceAssembler` Protocol. u-Segment3D treats 3D
instance segmentation as orthogonal-view consensus: per-axis 2D
instance maps (xy / xz / yz) are fused via gradient-based
post-processing into a single 3D volume.

Optional dep — installed via ``pip install volume-segmantics[usegment3d]``.

The module probes for u-Segment3D at import time; when missing it
raises :class:`ImportError`, which the package registry catches and
silently drops the backend from :func:`list_backends`. Calling
``get_backend("usegment3d")`` on a system without the extra surfaces
a clear "install the optional extra" message rather than a confusing
``ModuleNotFoundError`` stack trace.

**Windows note**: u-Segment3D's
default ``dtform_method='cellpose_improve'`` spawns a Dask LocalCluster
which fails without ``if __name__ == '__main__': freeze_support()``.
Tests override ``dtform_method='edt'`` to avoid the spawn entirely.
Production callers on Windows wanting the cellpose flavour must wrap
their entry point in :func:`multiprocessing.freeze_support`.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from volume_segmantics.inference.instance_assembly.base import (
    AssemblyConfig,
    InstanceAssemblerInputError,
)
from volume_segmantics.inference.instance_assembly.prediction_bundle import (
    PredictionBundle,
)


# Probe for u-Segment3D at import time 
# The package probes are deliberate: failing here means
# ``volume_segmantics.inference.instance_assembly.__init__`` skips
# registering the backend, so the system stays usable on machines
# without the optional extra. If you see this raise on a system that
# *should* have u-Segment3D installed, check that the upstream package
# imports cleanly under its expected name (``segment3D``).
try:
    from segment3D import parameters as _u_parameters  # noqa: F401
    from segment3D import usegment3d as _u_segment3d  # noqa: F401
except ImportError as exc:  # pragma: no cover — gating
    raise ImportError(
        "USegment3DAssembler requires the optional 'u-segment3d' "
        "package. Install via: pip install volume-segmantics[usegment3d]"
    ) from exc


_VALID_AXES: tuple = ("xy", "xz", "yz")


def _deep_merge(base: dict, overrides: Mapping[str, Any]) -> dict:
    """Deep-merge ``overrides`` into a deep-copied ``base`` and return it.

    Nested mappings recurse; everything else is replaced wholesale.
    """
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = (
                copy.deepcopy(v) if isinstance(v, (dict, list, tuple)) else v
            )
    return out


def _densify_labels(labels: np.ndarray) -> np.ndarray:
    """Map labels to ``0..N`` keeping ``0`` as background, ``1..N`` dense.

    u-Segment3D may return labels with gaps (e.g. after merging
    instances during gradient aggregation). Densifying matches the
    convention the rest of the pipeline expects.
    """
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


def _derive_img_xy_shape(
    per_axis: Mapping[str, np.ndarray],
    axes: Sequence[str],
) -> tuple:
    """Derive ``img_xy_shape = (Z, Y, X)`` from whichever axis is populated.

    The shape conventions are pinned by
    :class:`PredictionBundle.per_axis_instances`:

    * ``xy`` is ``(Z, Y, X)`` — direct case.
    * ``xz`` is ``(Y, Z, X)`` — reorder ``(sh[1], sh[0], sh[2])``.
    * ``yz`` is ``(X, Z, Y)`` — reorder ``(sh[1], sh[2], sh[0])``.
    """
    if (
        "xy" in axes and per_axis.get("xy") is not None
        and per_axis["xy"].size > 0
    ):
        return tuple(per_axis["xy"].shape)
    if (
        "xz" in axes and per_axis.get("xz") is not None
        and per_axis["xz"].size > 0
    ):
        sh = per_axis["xz"].shape
        return (sh[1], sh[0], sh[2])
    if (
        "yz" in axes and per_axis.get("yz") is not None
        and per_axis["yz"].size > 0
    ):
        sh = per_axis["yz"].shape
        return (sh[1], sh[2], sh[0])
    raise InstanceAssemblerInputError(
        "USegment3DAssembler: cannot derive img_xy_shape; all "
        f"configured axes {list(axes)} have empty arrays."
    )


class USegment3DAssembler:
    """u-Segment3D backend (indirect method).

    Consumes :attr:`PredictionBundle.per_axis_instances` and produces
    a 3D consensus instance volume via u-Segment3D's gradient-
    aggregation pipeline. The per-axis instances are typically
    produced by the pipeline version :class:`DistanceWatershedSliceProducer` or
    :class:`SemanticCcSliceProducer`

    Parameters
    ----------
    axes
        Which axes to consume. Subset of ``("xy", "xz", "yz")``;
        defaults to all three. For 2.5D pipelines (xy + one
        orthogonal), pass e.g. ``("xy", "xz")`` and the third slot is
        filled with a zero-size array (upstream substitutes a zero-
        shaped mask).
    params_overrides
        Optional nested-dict overrides applied on top of
        :func:`segment3D.parameters.get_2D_to_3D_aggregation_params`.
        Deep-merged: ``{"indirect_method": {"dtform_method": "edt"}}``
        overrides only that knob and leaves the rest of
        ``indirect_method`` at the upstream default. Shape:
        ``{section_name: {param_name: value, ...}, ...}``.
    """

    name: str = "usegment3d"
    required_fields = ("per_axis_instances",)

    def __init__(
        self,
        axes: Sequence[str] = _VALID_AXES,
        params_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        for ax in axes:
            if ax not in _VALID_AXES:
                raise ValueError(
                    f"USegment3DAssembler: unknown axis {ax!r}; "
                    f"expected one of {list(_VALID_AXES)}."
                )
        self.axes = tuple(axes)
        self.params_overrides: dict = (
            dict(params_overrides) if params_overrides else {}
        )

    def assemble(
        self,
        bundle: PredictionBundle,
        config: AssemblyConfig,
    ) -> np.ndarray:
        bundle.require(self.required_fields)
        per_axis = bundle.per_axis_instances or {}
        missing_axes = [ax for ax in self.axes if ax not in per_axis]
        if missing_axes:
            raise InstanceAssemblerInputError(
                "USegment3DAssembler: bundle.per_axis_instances is "
                f"missing required axes {missing_axes}. Configured "
                f"axes={list(self.axes)}; available="
                f"{list(per_axis.keys())}. Configure "
                "`prediction.per_axis_instances.axes` to match the "
                "assembler's axes, or shrink "
                "`instance_assembly.params.axes` to the available subset."
            )

        # u-Segment3D's indirect method requires a list of exactly 3
        # in xy / xz / yz order. Axes the caller didn't request get an
        # empty array; upstream code substitutes a zero-shaped mask.
        empty = np.empty(0, dtype=np.uint16)
        seg_xy = per_axis["xy"] if "xy" in self.axes else empty
        seg_xz = per_axis["xz"] if "xz" in self.axes else empty
        seg_yz = per_axis["yz"] if "yz" in self.axes else empty

        img_xy_shape = _derive_img_xy_shape(per_axis, self.axes)

        params = _u_parameters.get_2D_to_3D_aggregation_params()
        if self.params_overrides:
            params = _deep_merge(params, self.params_overrides)

        labels_3d, _aux = (
            _u_segment3d.aggregate_2D_to_3D_segmentation_indirect_method(
                [seg_xy, seg_xz, seg_yz],
                params,
                img_xy_shape,
            )
        )
        return _densify_labels(np.asarray(labels_3d, dtype=np.uint32))


__all__ = ["USegment3DAssembler"]
