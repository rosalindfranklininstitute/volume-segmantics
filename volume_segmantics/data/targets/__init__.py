"""Per-head target generators for v0.4.0b3 pipeline mode.

"""

from __future__ import annotations

from volume_segmantics.data import pipeline_registry as _registry
from volume_segmantics.data.targets.boundary_target import (
    derive_boundary_target_2d,
    make_boundary_target_generator,
)
from volume_segmantics.data.targets.distance_target import (
    derive_distance_target_2d,
    make_distance_target_generator,
)
from volume_segmantics.data.targets.sdm_target import (
    derive_sdm_target_2d,
    make_sdm_target_generator,
)


# Import-time registration 

for _name, _factory in (
    ("boundary", make_boundary_target_generator),
    ("distance", make_distance_target_generator),
    ("sdm", make_sdm_target_generator),
):
    try:
        _registry.register_target_generator(_name, _factory)
    except KeyError:
        # Already registered — module reimport in the same process.
        pass


__all__ = [
    "derive_boundary_target_2d",
    "derive_distance_target_2d",
    "derive_sdm_target_2d",
    "make_boundary_target_generator",
    "make_distance_target_generator",
    "make_sdm_target_generator",
]
