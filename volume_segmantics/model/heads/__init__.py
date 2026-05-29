"""Geometric + semantic prediction heads for pipeline.yaml.

Currently:

* :class:`SemanticHead` — registered as ``"semantic"``.
* :class:`BoundaryHead` — registered as ``"boundary"``.
* :class:`DistanceHead` — registered as ``"distance"``.
* :class:`SDMHead` — registered as ``"sdm"``.

All heads register themselves at this module's import time via
:func:`volume_segmantics.data.pipeline_registry.register_head`. Users
who import :mod:`volume_segmantics.model.heads` get the registration
side-effect for free; the trainer / pipeline parser do exactly that
to populate the ``_HEADS`` registry before consuming a
``pipeline.yaml`` ``heads:`` block.

"""

from __future__ import annotations

from typing import Any, Dict, List

from volume_segmantics.data import pipeline_registry as _registry
from volume_segmantics.data.pipeline_loader import HeadConfig
from volume_segmantics.model.heads.base import PredictionHead, TargetKind
from volume_segmantics.model.heads.boundary import (
    BOUNDARY_CHANNELS,
    BOUNDARY_HEAD_NAME,
    BoundaryHead,
)
from volume_segmantics.model.heads.distance import (
    DISTANCE_CHANNELS,
    DISTANCE_HEAD_NAME,
    DistanceHead,
)
from volume_segmantics.model.heads.sdm import SDM_HEAD_NAME, SDMHead
from volume_segmantics.model.heads.semantic import (
    SEMANTIC_HEAD_NAME,
    SemanticHead,
)


def _resolve_default_out_channels(
    head_name: str,
    head_config: HeadConfig,
    num_classes: int,
) -> int | None:
    """Per-head default ``out_channels`` when the YAML omits it.


    * ``semantic`` → ``num_classes`` (matches v0.4 behaviour).
    * ``sdm`` → ``1`` for ``binary`` variant, ``num_classes - 1`` for
      ``per_class``. Variant lives in :class:`HeadConfig.extra`; default
      matches :class:`SDMHead`'s own default of ``"binary"``.
    * ``boundary`` / ``distance`` → ``None``: the head's own ``__init__``
      knows the right channel count (1) and we let it default. Avoids
      duplicating that knowledge here.
    * Unknown heads → ``None``: pass through; the head decides.
    """
    if head_name == SEMANTIC_HEAD_NAME:
        return num_classes
    if head_name == SDM_HEAD_NAME:
        variant = head_config.extra.get("variant", "binary")
        if variant == "per_class":
            return num_classes - 1
        return 1
    return None


def build_head_modules(
    heads_cfg: Dict[str, HeadConfig],
    *,
    in_channels: int,
    num_classes: int,
    dim: int = 2,
) -> List[PredictionHead]:
    """Build concrete head instances from a parsed ``heads:`` config dict.

    * Only heads with ``enabled=True`` are built.
    * ``out_channels`` resolution order:

      1. ``head_config.out_channels`` if explicitly set.
      2. Else :func:`_resolve_default_out_channels` (semantic →
         ``num_classes``; sdm → 1 or ``num_classes - 1`` by variant).
      3. Else (boundary / distance) the kwarg is omitted and the head's
         own ``__init__`` default kicks in.

    * Iteration order is the dict's insertion order. The trainer's
      tuple-output contract relies on this — ``MultiTaskLossCalculator``
      iterates the head modules in the same order it sees the head
      configs.
    * Each head is instantiated via
      :func:`pipeline_registry.build_head` so the registry stays the
      single source of truth for head-name → class.
    * ``dim`` selects between 2D (default) and 3D head variants. b3
      raises on ``dim=3`` — the 3D path is deferred. The kwarg is
      preserved for forward-compat: when 3D heads land, callers won't
      need to change.
    """
    if dim != 2:
        raise ValueError(
            f"build_head_modules: only dim=2 is supported in v0.4.0b3; "
            f"got dim={dim}. The 3D path is deferred — see "
            f"docs/v0_4_b3_release_plan.md §0.3."
        )

    out: List[PredictionHead] = []
    for head_name, head_config in heads_cfg.items():
        if not head_config.enabled:
            continue
        if head_config.out_channels is not None:
            resolved_channels = head_config.out_channels
        else:
            resolved_channels = _resolve_default_out_channels(
                head_name, head_config, num_classes,
            )
        head_kwargs: Dict[str, Any] = dict(
            in_channels=in_channels,
            num_classes=num_classes,
            spatial_dims=dim,
            deep_supervision=head_config.deep_supervision,
            config=head_config,
        )
        head_kwargs.update(head_config.extra)
        if resolved_channels is not None:
            head_kwargs["out_channels"] = resolved_channels

        head = _registry.build_head(head_name, **head_kwargs)
        out.append(head)
    return out


# Import-time registration 
# Closed registry: registering each head once, here, is the only
# admission path. Reimporting the module (e.g. in tests that toggle
# ``importlib.reload``) silently no-ops on a duplicate.

for _name, _cls in (
    (SEMANTIC_HEAD_NAME, SemanticHead),
    (BOUNDARY_HEAD_NAME, BoundaryHead),
    (DISTANCE_HEAD_NAME, DistanceHead),
    (SDM_HEAD_NAME, SDMHead),
):
    try:
        _registry.register_head(_name, _cls)
    except KeyError:
        # Already registered — module reimport in the same process.
        # Idempotent: treat as no-op.
        pass


__all__ = [
    "BOUNDARY_CHANNELS",
    "BOUNDARY_HEAD_NAME",
    "BoundaryHead",
    "DISTANCE_CHANNELS",
    "DISTANCE_HEAD_NAME",
    "DistanceHead",
    "PredictionHead",
    "SDM_HEAD_NAME",
    "SDMHead",
    "SEMANTIC_HEAD_NAME",
    "SemanticHead",
    "TargetKind",
    "build_head_modules",
]
