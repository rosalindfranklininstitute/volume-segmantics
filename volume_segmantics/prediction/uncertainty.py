"""Uncertainty providers.


* :class:`UncertaintyProvider` — Protocol every provider conforms to.
* :class:`UncertaintyOutputs` — the return-type dataclass keyed exactly
  the same as the ``prediction_v1`` zarr layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


logger = logging.getLogger(__name__)


#  UncertaintyOutputs 


@dataclass
class UncertaintyOutputs:
    """Return type of :meth:`UncertaintyProvider.compute`.

    Fields are aligned with the ``prediction_v1`` zarr layout so a
    consumer can pass this object's ``as_dict()`` to
    :func:`volseg_write_prediction_zarr` and get a complete artefact.

    Always-present:

    * ``teacher_argmax`` — final per-voxel class label, ``(Z, Y, X)
      uint8``.

    Mode-dependent / optional:

    * ``teacher_probs`` — final merged probabilities,
      ``(C, Z, Y, X) float32``.
    * ``semantic_logits`` — raw logits, post-merge.
    * ``semantic_probs`` — softmax of merged logits.
    * ``tta_variance_map`` — per-voxel variance across constituent
      probability stacks, ``(Z, Y, X) float32``.
    * ``tta_entropy_map`` — per-voxel entropy of the merged
      probability stack, ``(Z, Y, X) float32``.
    * ``per_axis_stash`` — raw per-axis-rotation probability stacks
      keyed by axis-rotation tag (e.g. ``"z_rot0"``,
      ``"z_rot1"``, ...). Each value is ``(C, Z, Y, X) float32``.
    * ``extra_arrays`` — provider-specific extras for forward-compat.
    """

    teacher_argmax: np.ndarray
    teacher_probs: Optional[np.ndarray] = None
    semantic_logits: Optional[np.ndarray] = None
    semantic_probs: Optional[np.ndarray] = None
    tta_variance_map: Optional[np.ndarray] = None
    tta_entropy_map: Optional[np.ndarray] = None
    per_axis_stash: Dict[str, np.ndarray] = field(default_factory=dict)
    extra_arrays: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """``{key: array}`` dict aligned with the zarr writer's keys.

        Optional keys with ``None`` value are omitted, so the dict
        feeds directly into :func:`volseg_write_prediction_zarr`.
        """
        out: Dict[str, Any] = {"teacher_argmax": self.teacher_argmax}
        for name in (
            "teacher_probs", "semantic_logits", "semantic_probs",
            "tta_variance_map", "tta_entropy_map",
        ):
            val = getattr(self, name)
            if val is not None:
                out[name] = val
        out.update(self.extra_arrays)
        return out


#  Provider Protocol 


@runtime_checkable
class UncertaintyProvider(Protocol):
    """The contract every uncertainty provider satisfies.

    Concrete implementations:

    * b3: :class:`MultiAxisTTAProvider`.
    * b4+: kfold-ensemble, MC-dropout, basic-TTA. Adding a new
      provider is a registration call into
      :func:`register_uncertainty_provider` — no b3 code needs to
      change.
    """

    name: str

    def compute(self, *args: Any, **kwargs: Any) -> UncertaintyOutputs:
        ...


#  Provider registry 


_PROVIDERS: Dict[str, type] = {}


def register_uncertainty_provider(name: str, cls: type) -> None:
    if name in _PROVIDERS:
        raise KeyError(
            f"uncertainty provider {name!r} already registered "
            f"(existing: {_PROVIDERS[name].__name__})"
        )
    _PROVIDERS[name] = cls


def get_uncertainty_provider(name: str) -> type:
    if name not in _PROVIDERS:
        raise KeyError(
            f"unknown uncertainty provider {name!r}; known: "
            f"{sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name]


def list_uncertainty_providers() -> List[str]:
    return sorted(_PROVIDERS)


#  MultiAxisTTAProvider 


class MultiAxisTTAProvider:
    """Per-voxel variance + entropy over the multi-axis predict stacks.

    Reuses v0.4's existing ``quality: medium``/``high``/``z_only``
    multi-axis prediction path (3 / 12 / 4 constituent passes
    respectively). The caller runs that path, captures the per-axis-
    rotation softmax-probability stacks, and hands the dict to
    :meth:`compute_from_stacks`. The provider then:

    1. Stacks the per-axis probabilities along a new "passes" axis.
    2. Merges by **max-prob** along that axis (matches v0.4's
       existing behaviour). The argmax of the max-prob result is
       ``teacher_argmax``.
    3. Computes per-voxel **variance** as the mean across classes of
       the per-class variance over passes.
    4. Computes per-voxel **entropy** of the merged probability
       distribution.

    Sign convention:

    * ``tta_variance_map`` ≥ 0; higher = more disagreement between
      passes.
    * ``tta_entropy_map`` in ``[0, log(C)]`` (natural log); higher =
      more uncertain merged prediction.

    """

    name: str = "tta_uncertainty"

    def __init__(self, *, store_per_axis: bool = True) -> None:
        """
        Parameters
        ----------
        store_per_axis
            When ``True``, the constituent per-axis-rotation probability
            stacks are kept on :attr:`UncertaintyOutputs.per_axis_stash`.
            Set ``False`` for memory-bound runs.
        """
        self.store_per_axis = bool(store_per_axis)

    def compute_from_stacks(
        self,
        per_axis_probs: Mapping[str, np.ndarray],
    ) -> UncertaintyOutputs:
        """Compute uncertainty maps from a dict of per-pass probability stacks.

        Parameters
        ----------
        per_axis_probs
            ``{tag: (C, Z, Y, X) float}``. The tags are arbitrary —
            typical values are ``"z_rot0"``, ``"y_rot0"``, ``"x_rot0"``
            (3 passes for ``quality: medium``) or ``"z_rot0"`` ...
            ``"x_rot3"`` (12 passes for ``quality: high``).

        Returns
        -------
        UncertaintyOutputs
            Aligned with the ``prediction_v1`` zarr layout.
        """
        if not per_axis_probs:
            raise ValueError(
                "MultiAxisTTAProvider.compute_from_stacks: per_axis_probs "
                "is empty — no constituent passes to combine."
            )
        # Validate shape consistency.
        first_tag = next(iter(per_axis_probs))
        ref_shape = per_axis_probs[first_tag].shape
        if len(ref_shape) != 4:
            raise ValueError(
                f"MultiAxisTTAProvider: probability stacks must be (C, Z, Y, X); "
                f"got shape {ref_shape} for tag {first_tag!r}"
            )
        for tag, p in per_axis_probs.items():
            if p.shape != ref_shape:
                raise ValueError(
                    f"MultiAxisTTAProvider: probability shape mismatch — "
                    f"{first_tag!r} has shape {ref_shape} but {tag!r} has "
                    f"shape {p.shape}"
                )

        # Stack along a new passes axis: (P, C, Z, Y, X).
        tags = list(per_axis_probs.keys())
        stack = np.stack(
            [per_axis_probs[t].astype(np.float32, copy=False) for t in tags],
            axis=0,
        )

        # Max-prob merge across passes -> (C, Z, Y, X).
        merged_probs = stack.max(axis=0)
        # Renormalise per voxel so probs sum to 1 along C.
        norm_factor = merged_probs.sum(axis=0, keepdims=True)
        # Avoid divide-by-zero: where the merged pass stack is all-zero
        # (shouldn't happen with softmax outputs, but defensive),
        # fall back to uniform.
        zero_mask = norm_factor < 1e-12
        if zero_mask.any():
            merged_probs = np.where(
                zero_mask, 1.0 / merged_probs.shape[0], merged_probs,
            )
            norm_factor = merged_probs.sum(axis=0, keepdims=True)
        merged_probs = merged_probs / np.clip(norm_factor, 1e-12, None)
        teacher_argmax = merged_probs.argmax(axis=0).astype(np.uint8)

        # Per-voxel variance: per-class var across passes, mean over C.
        # var -> (C, Z, Y, X), mean over axis 0 -> (Z, Y, X).
        per_class_var = stack.var(axis=0)
        tta_variance_map = per_class_var.mean(axis=0).astype(np.float32)

        # Per-voxel entropy of merged distribution: H = -Σ p log p.
        # Natural log; result is non-negative, capped at log(C).
        eps = 1e-12
        log_p = np.log(np.clip(merged_probs, eps, 1.0))
        tta_entropy_map = -(merged_probs * log_p).sum(axis=0).astype(np.float32)

        per_axis_stash: Dict[str, np.ndarray] = {}
        if self.store_per_axis:
            per_axis_stash = {
                tag: per_axis_probs[tag].astype(np.float32, copy=False)
                for tag in tags
            }

        return UncertaintyOutputs(
            teacher_argmax=teacher_argmax,
            teacher_probs=merged_probs.astype(np.float32),
            semantic_probs=merged_probs.astype(np.float32),
            tta_variance_map=tta_variance_map,
            tta_entropy_map=tta_entropy_map,
            per_axis_stash=per_axis_stash,
        )

    def compute(self, *args: Any, **kwargs: Any) -> UncertaintyOutputs:
        """Provider Protocol entry point.

        """
        return self.compute_from_stacks(*args, **kwargs)


# Register the b3 provider at import time.
register_uncertainty_provider("tta_uncertainty", MultiAxisTTAProvider)


__all__ = [
    "MultiAxisTTAProvider",
    "UncertaintyOutputs",
    "UncertaintyProvider",
    "get_uncertainty_provider",
    "list_uncertainty_providers",
    "register_uncertainty_provider",
]
