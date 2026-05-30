"""Inference-mode dispatch 

Four registered modes:

* ``single_axis`` — v0.4's ``quality: low`` (1 axis).
* ``multi_axis`` — v0.4's ``quality: medium``/``high``/``z_only``.
  Constituent axis-rotation count comes from the legacy ``quality``
  knob (3 / 12 / 4).
* ``sliding_window`` — v0.4's existing sliding-window MONAI inference
  path (``use_sliding_window: True``).
* ``tta_uncertainty`` — runs the multi-axis path under
  :class:`MultiAxisTTAProvider`. Emits ``tta_variance_map`` /
  ``tta_entropy_map`` alongside the merged result.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class InferenceModeDescriptor:
    """Static metadata for a registered inference mode.

    Attributes
    ----------
    name
        Canonical mode name (matches a value in
        :data:`pipeline_loader.KNOWN_INFERENCE_MODES`).
    populates
        Frozen set of zarr-array keys this mode writes by default.
        Used by :mod:`api.predict` to know which writer methods to
        call. Optional head-specific keys are added on top by the
        caller based on the active head set.
    requires_uncertainty_provider
        ``True`` iff the mode requires a populated
        :class:`UncertaintyProvider` to function (currently only
        ``tta_uncertainty``).
    description
        One-line human-readable description for the manifest /
        ``--help``.
    """

    name: str
    populates: FrozenSet[str]
    requires_uncertainty_provider: bool = False
    description: str = ""


#  Registry 


_MODES: Dict[str, InferenceModeDescriptor] = {}


def register_inference_mode(descriptor: InferenceModeDescriptor) -> None:
    if descriptor.name in _MODES:
        raise KeyError(
            f"inference mode {descriptor.name!r} already registered"
        )
    _MODES[descriptor.name] = descriptor


def get_inference_mode(name: str) -> InferenceModeDescriptor:
    if name not in _MODES:
        raise KeyError(
            f"unknown inference mode {name!r}; known: {sorted(_MODES)}"
        )
    return _MODES[name]


def list_inference_modes() -> List[str]:
    return sorted(_MODES)


#  b3 ships four modes 


# Common keys every mode populates.
_BASE_OUTPUTS = frozenset({
    "teacher_argmax", "teacher_probs",
    "semantic_logits", "semantic_probs",
})


register_inference_mode(InferenceModeDescriptor(
    name="single_axis",
    populates=_BASE_OUTPUTS,
    description=(
        "Single-axis prediction. Same as v0.4 quality: low."
    ),
))

register_inference_mode(InferenceModeDescriptor(
    name="multi_axis",
    populates=_BASE_OUTPUTS,
    description=(
        "Multi-axis prediction with max-prob merge across constituent "
        "passes. Constituent count from the legacy quality knob "
        "(medium=3, high=12, z_only=4)."
    ),
))

register_inference_mode(InferenceModeDescriptor(
    name="sliding_window",
    populates=_BASE_OUTPUTS,
    description=(
        "MONAI sliding-window inference for tile-based prediction on "
        "very large volumes."
    ),
))

register_inference_mode(InferenceModeDescriptor(
    name="tta_uncertainty",
    populates=_BASE_OUTPUTS | frozenset({
        "tta_variance_map", "tta_entropy_map",
    }),
    requires_uncertainty_provider=True,
    description=(
        "Multi-axis prediction + per-voxel variance + entropy via the "
        "MultiAxisTTAProvider. Emits all multi_axis outputs plus the "
        "uncertainty maps."
    ),
))


__all__ = [
    "InferenceModeDescriptor",
    "get_inference_mode",
    "list_inference_modes",
    "register_inference_mode",
]
