"""PredictionBundle dataclass.

"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Optional, Tuple

import numpy as np

from volume_segmantics.inference.instance_assembly.base import (
    InstanceAssemblerInputError,
)


@dataclass
class PredictionBundle:
    """All dense prediction outputs from the inference stack.

    Field shapes:

    * ``semantic_argmax``: ``(Z, Y, X) uint8`` — required by foreground masking.
    * ``semantic_probs``: ``(C, Z, Y, X) float32`` — optional, soft masking.
    * ``boundary_map``: ``(1, Z, Y, X) float32`` — sigmoid probs.
    * ``distance_map``: ``(1, Z, Y, X) float32`` — distance values.
    * ``sdm_map``: ``(K, Z, Y, X) float32`` — signed distance map.
    * ``per_axis_instances``: ``{"xy": (Z,Y,X) uint32, "xz": (Y,Z,X)
      uint32, "yz": (X,Z,Y) uint32}`` — populated by per-axis 2D
      producers (B3.G.4) when an instance-assembly backend is
      configured. Each entry is the 3D array whose leading axis is
      the slice axis.

    """

    semantic_argmax: Optional[np.ndarray] = None
    semantic_probs: Optional[np.ndarray] = None
    boundary_map: Optional[np.ndarray] = None
    distance_map: Optional[np.ndarray] = None
    sdm_map: Optional[np.ndarray] = None
    per_axis_instances: Optional[Dict[str, np.ndarray]] = None

    def has(self, field_name: str) -> bool:
        """``True`` when ``field_name`` is set to a non-``None`` value."""
        return getattr(self, field_name, None) is not None

    def require(self, fields: Tuple[str, ...]) -> None:
        """Validate that every name in ``fields`` is set on the bundle.

        Raises :class:`InstanceAssemblerInputError` listing the missing
        fields plus the fields that are present, so the caller can
        either supply the missing data upstream or switch backends.
        """
        missing = [f for f in fields if not self.has(f)]
        if missing:
            present = [
                f.name for f in dataclass_fields(self) if self.has(f.name)
            ]
            raise InstanceAssemblerInputError(
                f"backend requires fields {missing} which are not in the "
                f"bundle. Available fields: {present}"
            )


__all__ = ["PredictionBundle"]
