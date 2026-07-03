"""Inference-time post-processing.

Post-processing primitives that turn per-voxel head outputs into final 3D instance segmentations.

"""

from __future__ import annotations

from volume_segmantics.inference.instance_assembly import (
    AssemblyConfig,
    InstanceAssembler,
    InstanceAssemblerInputError,
    PredictionBundle,
    get_backend,
    list_backends,
    register_backend,
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
