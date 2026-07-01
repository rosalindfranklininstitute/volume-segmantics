"""Instance-assembly backend registry.

"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from volume_segmantics.inference.instance_assembly.base import (
    AssemblyConfig,
    InstanceAssembler,
    InstanceAssemblerInputError,
)
from volume_segmantics.inference.instance_assembly.prediction_bundle import (
    PredictionBundle,
)


logger = logging.getLogger(__name__)


# Registry 


_BACKENDS: Dict[str, Type] = {}


def register_backend(name: str, cls: Type) -> None:
    """Register an :class:`InstanceAssembler` subclass under ``name``.

    Raises :class:`KeyError` on duplicate registration.
    """
    if name in _BACKENDS:
        raise KeyError(
            f"instance-assembly backend {name!r} already registered "
            f"(existing: {_BACKENDS[name].__name__})"
        )
    _BACKENDS[name] = cls


def get_backend(name: str) -> Type:
    """Return the registered backend class for ``name``.

    Raises :class:`KeyError` listing known backends if ``name`` is
    not registered. Common cause: the optional extra for that
    backend is not installed (``[usegment3d]`` for ``usegment3d``).
    """
    if name not in _BACKENDS:
        known = sorted(_BACKENDS)
        if name == "usegment3d" and not known:
            raise KeyError(
                "instance-assembly backend 'usegment3d' requires the "
                "optional extra. Install via "
                "`pip install volume-segmantics[usegment3d]`."
            )
        raise KeyError(
            f"unknown instance-assembly backend {name!r}; known: {known}"
        )
    return _BACKENDS[name]


def list_backends() -> List[str]:
    """Registered backend names in alphabetical order."""
    return sorted(_BACKENDS)


#  Built-in dependency-free backends
from volume_segmantics.inference.instance_assembly.slice_overlap import (
    SliceOverlapAssembler,
)
from volume_segmantics.inference.instance_assembly.watershed_3d import (
    Watershed3DAssembler,
)

register_backend("slice_overlap", SliceOverlapAssembler)
register_backend("watershed_3d", Watershed3DAssembler)


#  Lazy registration of the optional uSegment3D backend
# Importing the adapter triggers an upstream ImportError when the
# `[usegment3d]` extra isn't installed; we catch it so the registry
# stays usable on systems without the extra.

try:
    from volume_segmantics.inference.instance_assembly.usegment3d import (
        USegment3DAssembler,
    )
    register_backend("usegment3d", USegment3DAssembler)
except ImportError as _e:  # pragma: no cover — install-gated
    logger.info(
        "USegment3DAssembler not registered: %s. Install via "
        "`pip install volume-segmantics[usegment3d]` to enable.", _e,
    )


__all__ = [
    "AssemblyConfig",
    "InstanceAssembler",
    "InstanceAssemblerInputError",
    "PredictionBundle",
    "get_backend",
    "list_backends",
    "register_backend",
]
