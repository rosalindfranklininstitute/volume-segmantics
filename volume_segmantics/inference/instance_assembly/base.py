"""InstanceAssembler Protocol and AssemblyConfig.

The Protocol every instance-assembly backend conforms to. 

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


class InstanceAssemblerInputError(ValueError):
    """Raised when a :class:`PredictionBundle` lacks fields a backend
    declared as ``required_fields``.

    The error message lists the missing fields plus the fields that
    *were* available on the bundle, so callers can either supply the
    missing data upstream or switch to a backend that consumes
    different fields.
    """


@runtime_checkable
class InstanceAssembler(Protocol):
    """Produces a 3D instance label volume from a prediction bundle.

    Backends declare what bundle fields they require via
    :attr:`required_fields`; the orchestrator validates this at
    config-load time so misconfiguration fails fast.

    Attributes
    ----------
    name
        Registered backend name (e.g. ``"usegment3d"``).
    required_fields
        Tuple of :class:`PredictionBundle` field names this backend
        consumes. Validated against the bundle in
        :meth:`PredictionBundle.require` before assembly.
    """

    name: str
    required_fields: Tuple[str, ...]

    def assemble(
        self,
        bundle: "PredictionBundle",  
        config: "AssemblyConfig",
    ) -> np.ndarray:
        """Return ``(Z, Y, X) uint32`` instance labels.

        Background voxels are encoded as ``0``; instance IDs start at
        ``1`` and are dense (no gaps). Backend adapters relabel before
        returning if the underlying library produces sparse IDs.
        """
        ...


@dataclass
class AssemblyConfig:
    """Backend-agnostic per-call config.

    Backend-specific tuning lives in the backend's ``__init__``
    kwargs; this dataclass carries only fields the orchestrator might
    want to vary per call (e.g. spatial cropping for ROI-mode
    inference).

    Attributes
    ----------
    foreground_class_ids
        When set, restricts the foreground mask to voxels whose
        ``semantic_argmax`` matches one of these class IDs. ``None``
        treats every non-zero ``semantic_argmax`` as foreground.
    voxel_size
        ``(Z, Y, X)`` voxel size in physical units. Forwarded to
        backends that consume anisotropy.
    """

    foreground_class_ids: Optional[Tuple[int, ...]] = None
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)


__all__ = [
    "AssemblyConfig",
    "InstanceAssembler",
    "InstanceAssemblerInputError",
]
