"""Direct 3D watershed instance assembler (h-maxima seeded).

Unlike the per-slice producers + ``slice_overlap`` stitcher, this works on the
whole volume at once, so it is inherently 3D-consistent (no inter-slice
label stitching, hence no transitive-merge "one giant blob" failure).

Per call:

1. Foreground mask from ``semantic_argmax``.
2. A seeding surface that peaks at object centres — either the learned
   ``distance_map`` head (``source="head"``) or the geometric EDT of the
   foreground (``source="semantic_edt"``), normalised to ``[0, 1]`` and
   Gaussian-smoothed.
3. Seeds = **h-maxima** of that surface (``skimage.morphology.h_maxima``):
   regional maxima taller than ``h`` than their surroundings. ``h`` collapses
   the many small bumps a learned/geometric distance has *within* a single
   non-convex object into ~one seed per object — the key to not
   over-segmenting.
4. ``watershed(-surface, markers, mask=fg)``; relabel consecutive; optional
   ``min_size`` filter.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as _ndi

from volume_segmantics.inference.instance_assembly.base import (
    InstanceAssemblerInputError,
)


def _foreground(sem: np.ndarray, fg_class_ids: Optional[Sequence[int]]) -> np.ndarray:
    if fg_class_ids:
        out = np.zeros(sem.shape, dtype=bool)
        for c in fg_class_ids:
            out |= sem == int(c)
        return out
    return sem > 0


class Watershed3DAssembler:
    """3D h-maxima watershed. Robust for single-axis / non-convex objects.

    Params (from ``instance_assembly.params``):

    * ``source`` (str, default ``"head"``): ``"head"`` uses the learned
      ``distance_map``; ``"semantic_edt"`` uses the geometric EDT of the
      predicted foreground (no distance head required).
    * ``h`` (float, default 0.35): h-maxima height in ``[0, 1]`` surface units.
      Higher merges more bumps per object (fewer seeds); lower keeps more.
    * ``smooth_sigma`` (float, default 1.0): Gaussian smoothing of the surface.
    * ``sampling`` (tuple, default ``(1, 1, 1)``): voxel spacing for the
      ``semantic_edt`` EDT (anisotropy); ignored for ``source="head"``.
    * ``min_size`` (int, default 0): drop instances smaller than this.
    """

    name: str = "watershed_3d"
    required_fields: Tuple[str, ...] = ("semantic_argmax", "distance_map")

    def __init__(
        self,
        *,
        source: str = "head",
        h: float = 0.35,
        smooth_sigma: float = 1.0,
        sampling: Sequence[float] = (1.0, 1.0, 1.0),
        min_size: int = 0,
        **_extra: Any,
    ) -> None:
        if source not in ("head", "semantic_edt"):
            raise ValueError(
                f"watershed_3d: source must be 'head' or 'semantic_edt'; "
                f"got {source!r}"
            )
        self.source = source
        self.h = float(h)
        self.smooth_sigma = float(smooth_sigma)
        self.sampling = tuple(float(s) for s in sampling)
        self.min_size = max(0, int(min_size))
        # ``head`` does not need a distance head only when source is semantic_edt
        self.required_fields = (
            ("semantic_argmax", "distance_map") if source == "head"
            else ("semantic_argmax",)
        )

    def assemble(self, bundle: "Any", config: "Any") -> np.ndarray:
        bundle.require(self.required_fields)
        sem = np.asarray(bundle.semantic_argmax)
        fg = _foreground(sem, getattr(config, "foreground_class_ids", None))
        if not fg.any():
            return np.zeros(sem.shape, dtype=np.uint32)

        if self.source == "head":
            surf = np.asarray(bundle.distance_map, dtype=np.float32)
            if surf.ndim == 4 and surf.shape[0] == 1:  # (1, Z, Y, X) -> (Z, Y, X)
                surf = surf[0]
            if surf.shape != sem.shape:
                raise InstanceAssemblerInputError(
                    f"watershed_3d: distance_map shape {surf.shape} != "
                    f"semantic_argmax shape {sem.shape}"
                )
            surf = np.where(fg, surf, 0.0)
        else:  # semantic_edt
            surf = _ndi.distance_transform_edt(fg, sampling=self.sampling).astype(
                np.float32
            )

        # Normalise to [0, 1] so ``h`` is in consistent units.
        smax = float(surf.max())
        if smax > 0:
            surf = surf / smax
        if self.smooth_sigma > 0:
            surf = _ndi.gaussian_filter(surf, self.smooth_sigma)

        # h-maxima seeds -> markers -> watershed.
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed

        seeds = h_maxima(surf, self.h) * fg
        markers, n_markers = _ndi.label(seeds)
        if n_markers == 0:
            # Fall back to a single seed at the global max so we still split fg.
            idx = np.unravel_index(int(np.argmax(surf)), surf.shape)
            markers = np.zeros(surf.shape, dtype=np.int32)
            markers[idx] = 1
        labels = watershed(-surf, markers=markers, mask=fg)

        out = labels.astype(np.int64)
        if self.min_size > 0:
            ids, sizes = np.unique(out, return_counts=True)
            remap = np.arange(int(out.max()) + 1, dtype=np.int64)
            for inst_id, size in zip(ids, sizes):
                if inst_id != 0 and size < self.min_size:
                    remap[inst_id] = 0
            out = remap[out]
        # repack consecutive ids from 1
        kept = np.unique(out)
        repack = np.zeros(int(out.max()) + 1, dtype=np.int64)
        repack[kept] = np.arange(len(kept))
        return repack[out].astype(np.uint32)


__all__ = ["Watershed3DAssembler"]
