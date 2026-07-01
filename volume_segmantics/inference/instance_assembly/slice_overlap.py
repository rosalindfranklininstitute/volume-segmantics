"""Simple single-axis instance assembler: link 2D slices by overlap.

A dependency-free baseline (and the natural choice for *single-axis*
prediction, where the multi-axis-consensus ``usegment3d`` backend
over-fragments). It takes one per-axis 2D instance map — labels restart
per slice — and stitches it into a 3D volume by union-find over pixel
overlap between consecutive slices:

* a 2D object in slice ``z`` and one in slice ``z+1`` that share at least
  ``min_overlap`` foreground pixels are merged into the same 3D instance;
* this also *heals* per-slice over-segmentation — two fragments of one
  object in slice ``z`` that both overlap the same object in slice
  ``z+1`` get unified.

Optionally drops instances smaller than ``min_size`` voxels.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from volume_segmantics.inference.instance_assembly.base import (
    InstanceAssemblerInputError,
)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


class SliceOverlapAssembler:
    """Stitch one per-axis 2D instance map into 3D by inter-slice overlap.

    Params (from ``instance_assembly.params``):

    * ``axis`` (str, default ``"xy"``): which per-axis map to consume. Its
      slicing axis must be axis 0 of the stored array (true for ``"xy"`` ->
      ``(Z, Y, X)``).
    * ``min_overlap`` (int, default 1): minimum shared foreground pixels
      between two slices' objects to link them.
    * ``min_size`` (int, default 0): drop 3D instances smaller than this many
      voxels (0 = keep all).
    """

    name: str = "slice_overlap"
    required_fields: Tuple[str, ...] = ("per_axis_instances",)

    def __init__(
        self,
        *,
        axis: str = "xy",
        min_overlap: int = 1,
        min_size: int = 0,
        **_extra: Any,
    ) -> None:
        self.axis = str(axis)
        self.min_overlap = max(1, int(min_overlap))
        self.min_size = max(0, int(min_size))

    def assemble(self, bundle: "Any", config: "Any") -> np.ndarray:
        bundle.require(self.required_fields)
        per_axis: Mapping[str, np.ndarray] = bundle.per_axis_instances or {}
        if self.axis not in per_axis or per_axis[self.axis] is None:
            raise InstanceAssemblerInputError(
                f"slice_overlap: per_axis_instances missing axis "
                f"{self.axis!r}; available={list(per_axis.keys())}. Configure "
                f"`instance_assembly.params.axis` to an available axis."
            )
        vol = np.asarray(per_axis[self.axis])
        if vol.ndim != 3:
            raise ValueError(
                f"slice_overlap: expected a 3D per-axis map; got shape "
                f"{vol.shape}"
            )

        n_slices = vol.shape[0]
        # Make per-slice labels globally unique (offset each slice by the
        # running max) so each (slice, label) is its own union-find node.
        glob = np.zeros(vol.shape, dtype=np.int64)
        running_max = 0
        for z in range(n_slices):
            sl = vol[z]
            fg = sl > 0
            if not fg.any():
                continue
            glob[z][fg] = sl[fg].astype(np.int64) + running_max
            running_max = int(glob[z].max())

        if running_max == 0:
            return np.zeros(vol.shape, dtype=np.uint32)  # nothing foreground

        uf = _UnionFind(running_max + 1)
        # Link objects that overlap between consecutive slices.
        for z in range(n_slices - 1):
            a, b = glob[z], glob[z + 1]
            both = (a > 0) & (b > 0)
            if not both.any():
                continue
            pairs, counts = np.unique(
                np.stack([a[both], b[both]], axis=1), axis=0, return_counts=True,
            )
            for (pa, pb), cnt in zip(pairs, counts):
                if cnt >= self.min_overlap:
                    uf.union(int(pa), int(pb))

        # Relabel by union-find root, packed to consecutive ids from 1.
        roots = np.zeros(running_max + 1, dtype=np.int64)
        for node in range(1, running_max + 1):
            roots[node] = uf.find(node)
        flat = glob.ravel()
        out = np.zeros(flat.shape, dtype=np.int64)
        fg_mask = flat > 0
        _, inverse = np.unique(roots[flat[fg_mask]], return_inverse=True)
        out[fg_mask] = inverse + 1
        out = out.reshape(vol.shape)

        if self.min_size > 0:
            ids, sizes = np.unique(out, return_counts=True)
            remap = np.arange(int(out.max()) + 1, dtype=np.int64)
            for inst_id, size in zip(ids, sizes):
                if inst_id != 0 and size < self.min_size:
                    remap[inst_id] = 0
            out = remap[out]
            # repack consecutive after dropping small instances
            kept = np.unique(out)
            repack = np.zeros(int(out.max()) + 1, dtype=np.int64)
            repack[kept] = np.arange(len(kept))
            out = repack[out]

        return out.astype(np.uint32)


__all__ = ["SliceOverlapAssembler"]
