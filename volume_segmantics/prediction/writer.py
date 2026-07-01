"""Zarr wroter.


Layout::

    prediction.zarr/
    ├ .zattrs                     # manifest (see below)
    ├ teacher_argmax/             # (Z, Y, X) uint8         [always present]
    ├ teacher_probs/              # (C, Z, Y, X) float32    [optional]
    ├ semantic_logits/            # (C, Z, Y, X) float32    [optional]
    ├ semantic_probs/             # (C, Z, Y, X) float32    [optional]
    ├ boundary_map/               # (1, Z, Y, X) float32    [optional]
    ├ distance_map/               # (1, Z, Y, X) float32    [optional]
    ├ sdm_map/                    # (K, Z, Y, X) float32    [optional]
    ├ tta_variance_map/           # (Z, Y, X) float32       [optional]
    ├ tta_entropy_map/            # (Z, Y, X) float32       [optional]
    ├ per_axis_instances/         # group, optional
    │   ├ xy                      # (Z, Y, X) uint32
    │   ├ xz                      # (Y, Z, X) uint32
    │   └ yz                      # (X, Z, Y) uint32
    └ instance_labels/            # (Z, Y, X) uint32        [optional]

"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TypeVar

import numpy as np
import zarr

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _retry_on_oserror(
    fn: Callable[[], _T],
    *,
    what: str = "zarr write",
    attempts: int = 10,
    base_delay: float = 0.1,
) -> _T:
    """Run ``fn``; retry transient filesystem errors before giving up.

    zarr's :class:`LocalStore` writes metadata atomically (write a ``.partial``
    temp file, then ``os.replace`` it onto ``zarr.json``). On Windows/WSL mounts
    that replace intermittently fails with ``PermissionError`` (WinError 5) when
    an antivirus/search-indexer briefly holds a handle on the just-created file.
    The lock is transient, so a short bounded retry succeeds. Re-raises the last
    error after ``attempts`` so genuine failures still surface.
    """
    last: Optional[OSError] = None
    for i in range(attempts):
        try:
            return fn()
        except OSError as exc:  # PermissionError is an OSError subclass
            last = exc
            time.sleep(base_delay * (i + 1))
    logger.warning("%s still failing after %d retries; re-raising.", what, attempts)
    assert last is not None
    raise last


PREDICTION_SCHEMA_VERSION: str = "prediction_v1"

#: Inference modes recognised in the manifest. Source of truth lives in
#: :data:`pipeline_loader.KNOWN_INFERENCE_MODES`; we duplicate the set
#: here so the writer can validate without taking on a circular import.
_VALID_INFERENCE_MODES: frozenset = frozenset({
    "single_axis", "multi_axis", "sliding_window", "tta_uncertainty",
})

#: Default chunk shape for spatial arrays. Aligned to a typical
#: vol-seg slice dimension; callers can override per-write.
_DEFAULT_SPATIAL_CHUNKS: Tuple[int, int, int] = (32, 256, 256)

#: Region into a 3D volume (Z, Y, X) — slice triplet.
Region3D = Tuple[slice, slice, slice]


def _iso_utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _volseg_version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("volume-segmantics")
    except Exception:  # noqa: BLE001 — best-effort provenance
        return "unknown"


class PredictionZarrWriter:
    """Stream arrays into a ``prediction_v1`` zarr store.

    Parameters
    ----------
    output_path
        Directory path for the zarr store. The path is created on
        construction; pass ``overwrite=True`` to clobber an existing
        store at the same path.
    volume_shape
        ``(Z, Y, X)`` of the source volume. All spatial arrays use
        this shape unless the array carries a leading channel/class
        axis (then the spatial dims are appended).
    inference_mode
        One of :data:`_VALID_INFERENCE_MODES`. Written verbatim to
        ``.zattrs["inference_mode"]``.
    class_metadata
        JSON-serialisable mapping of class ids -> ``{"name", "color",
        ...}``. Written verbatim to ``.zattrs["class_metadata"]``.
        Pass ``{}`` if unknown; the manifest still records the empty
        dict.
    inference_config
        Optional JSON-serialisable mapping of mode-specific knobs
        (tile size, stride, fusion strategy, etc.).
    source_volume_path, source_volume_hash
        Provenance strings for the input volume. Optional.
    model_checkpoint_hash
        SHA-256 of the model checkpoint. Optional. Set via
        :meth:`finalize(extra_attrs=...)` if not known at construction.
    voxel_size_nm
        ``(z, y, x)`` voxel size in nanometres. Optional.
    chunks
        Spatial chunk shape ``(Z, Y, X)``. Default
        :data:`_DEFAULT_SPATIAL_CHUNKS`.
    overwrite
        Clobber an existing store at ``output_path``.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        volume_shape: Sequence[int],
        inference_mode: str,
        class_metadata: Mapping[str, Any],
        *,
        inference_config: Optional[Mapping[str, Any]] = None,
        source_volume_path: Optional[Union[str, Path]] = None,
        source_volume_hash: Optional[str] = None,
        model_checkpoint_hash: Optional[str] = None,
        voxel_size_nm: Optional[Sequence[float]] = None,
        chunks: Optional[Sequence[int]] = None,
        overwrite: bool = False,
    ) -> None:
        if inference_mode not in _VALID_INFERENCE_MODES:
            raise ValueError(
                f"inference_mode must be one of "
                f"{sorted(_VALID_INFERENCE_MODES)}; got {inference_mode!r}"
            )
        if len(volume_shape) != 3:
            raise ValueError(
                f"volume_shape must be (Z, Y, X); got {tuple(volume_shape)}"
            )

        self.output_path = Path(output_path)
        self.volume_shape: Tuple[int, int, int] = tuple(
            int(v) for v in volume_shape
        )
        self.inference_mode = inference_mode
        self.class_metadata = dict(class_metadata)
        self.inference_config = dict(inference_config or {})
        self.source_volume_path = (
            str(source_volume_path) if source_volume_path else None
        )
        self.source_volume_hash = source_volume_hash
        self.model_checkpoint_hash = model_checkpoint_hash
        self.voxel_size_nm: Optional[Tuple[float, float, float]] = (
            tuple(float(v) for v in voxel_size_nm)
            if voxel_size_nm is not None else None
        )
        self.chunks: Tuple[int, int, int] = (
            tuple(int(c) for c in chunks)
            if chunks is not None else _DEFAULT_SPATIAL_CHUNKS
        )

        # Pre-create the output directory and root group. zarr 2.x
        # opens with ``DirectoryStore`` for filesystem layout; zarr 3.x
        # replaced it with ``zarr.storage.LocalStore``.
        if hasattr(zarr, "DirectoryStore"):
            self._store = zarr.DirectoryStore(str(self.output_path))
        else:
            self._store = zarr.storage.LocalStore(str(self.output_path))
        self._root = zarr.group(store=self._store, overwrite=overwrite)
        self._created_at = _iso_utc_now()
        self._finalized = False

        # Lazy array creation: arrays land in self._arrays on first
        # write; finalize() inspects this dict to know which heads /
        # signals were populated.
        self._arrays: Dict[str, zarr.Array] = {}
        self._per_axis_group: Optional[zarr.Group] = None

    #  Always-present array 

    def write_teacher_argmax(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(Z, Y, X) uint8`` per-voxel class label.

        The one array the writer requires before :meth:`finalize`. Any
        integer dtype is accepted on input; the array stores ``uint8``.
        """
        self._check_not_finalized()
        if data.dtype.kind not in ("i", "u"):
            raise TypeError(
                f"teacher_argmax data must be integer; got {data.dtype}"
            )
        # teacher_argmax is stored as uint8 (the prediction_v1 format). A blind
        # astype(uint8) would silently wrap any label outside 0-255 (e.g. class
        # 300 -> 44), corrupting the labels. Reject out-of-range values instead.
        if data.size:
            dmin, dmax = int(data.min()), int(data.max())
            if dmin < 0 or dmax > 255:
                raise ValueError(
                    f"teacher_argmax stores uint8 (0-255) but data has values in "
                    f"[{dmin}, {dmax}]. The prediction_v1 zarr format supports at "
                    f"most 256 classes; relabel or use a format that stores wider "
                    f"integer labels."
                )
        arr = self._get_or_create(
            "teacher_argmax", self.volume_shape, np.uint8, self.chunks,
        )
        self._write_into(arr, data.astype(np.uint8, copy=False), region)

    #  Per-class arrays 

    def write_teacher_probs(
        self, data: np.ndarray,
        *, num_classes: Optional[int] = None,
        region: Optional[Region3D] = None,
    ) -> None:
        """``(C, Z, Y, X) float32`` softmax-merged probabilities."""
        self._write_per_class("teacher_probs", data, num_classes, region)

    def write_semantic_logits(
        self, data: np.ndarray,
        *, num_classes: Optional[int] = None,
        region: Optional[Region3D] = None,
    ) -> None:
        """``(C, Z, Y, X) float32`` raw merged semantic logits."""
        self._write_per_class("semantic_logits", data, num_classes, region)

    def write_semantic_probs(
        self, data: np.ndarray,
        *, num_classes: Optional[int] = None,
        region: Optional[Region3D] = None,
    ) -> None:
        """``(C, Z, Y, X) float32`` softmax of merged semantic logits."""
        self._write_per_class("semantic_probs", data, num_classes, region)

    def _write_per_class(
        self,
        key: str,
        data: np.ndarray,
        num_classes: Optional[int],
        region: Optional[Region3D],
    ) -> None:
        self._check_not_finalized()
        if data.ndim != 4:
            raise ValueError(
                f"{key}: expected (C, Z, Y, X); got shape {data.shape}"
            )
        c = int(data.shape[0])
        if num_classes is not None and num_classes != c:
            raise ValueError(
                f"{key}: data.shape[0]={c} != num_classes={num_classes}"
            )
        arr_shape = (c,) + self.volume_shape
        arr_chunks = (c,) + self.chunks
        arr = self._get_or_create(key, arr_shape, np.float32, arr_chunks)
        if region is None:
            arr[:] = data.astype(np.float32, copy=False)
        else:
            write_region = (slice(None),) + tuple(region)
            arr[write_region] = data.astype(np.float32, copy=False)

    #  Per-head geometric arrays 

    def write_boundary_map(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(1, Z, Y, X) float32`` sigmoid-prob boundary map."""
        self._write_single_channel_geometric(
            "boundary_map", data, region,
        )

    def write_distance_map(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(1, Z, Y, X) float32`` distance map."""
        self._write_single_channel_geometric(
            "distance_map", data, region,
        )

    def write_sdm_map(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(K, Z, Y, X) float32`` signed distance map.

        ``K=1`` for the binary SDM variant; ``K=num_classes-1`` for
        per-class. Channel count is fixed on first write.
        """
        self._check_not_finalized()
        if data.ndim != 4:
            raise ValueError(
                f"sdm_map: expected (K, Z, Y, X); got shape {data.shape}"
            )
        k = int(data.shape[0])
        arr_shape = (k,) + self.volume_shape
        arr_chunks = (k,) + self.chunks
        arr = self._get_or_create(
            "sdm_map", arr_shape, np.float32, arr_chunks,
        )
        if region is None:
            arr[:] = data.astype(np.float32, copy=False)
        else:
            write_region = (slice(None),) + tuple(region)
            arr[write_region] = data.astype(np.float32, copy=False)

    def _write_single_channel_geometric(
        self, key: str, data: np.ndarray, region: Optional[Region3D],
    ) -> None:
        self._check_not_finalized()
        # Accept either (1, Z, Y, X) or (Z, Y, X); coerce to (1, Z, Y, X).
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        if data.ndim != 4 or data.shape[0] != 1:
            raise ValueError(
                f"{key}: expected (1, Z, Y, X) or (Z, Y, X); got shape "
                f"{data.shape}"
            )
        arr_shape = (1,) + self.volume_shape
        arr_chunks = (1,) + self.chunks
        arr = self._get_or_create(key, arr_shape, np.float32, arr_chunks)
        if region is None:
            arr[:] = data.astype(np.float32, copy=False)
        else:
            write_region = (slice(None),) + tuple(region)
            arr[write_region] = data.astype(np.float32, copy=False)

    #  TTA uncertainty arrays 

    def write_tta_variance_map(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(Z, Y, X) float32`` per-voxel variance across TTA passes."""
        self._check_not_finalized()
        if data.ndim != 3:
            raise ValueError(
                f"tta_variance_map: expected (Z, Y, X); got shape {data.shape}"
            )
        arr = self._get_or_create(
            "tta_variance_map", self.volume_shape, np.float32, self.chunks,
        )
        self._write_into(arr, data.astype(np.float32, copy=False), region)

    def write_tta_entropy_map(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(Z, Y, X) float32`` per-voxel entropy of merged probs."""
        self._check_not_finalized()
        if data.ndim != 3:
            raise ValueError(
                f"tta_entropy_map: expected (Z, Y, X); got shape {data.shape}"
            )
        arr = self._get_or_create(
            "tta_entropy_map", self.volume_shape, np.float32, self.chunks,
        )
        self._write_into(arr, data.astype(np.float32, copy=False), region)

    #  Instance assembly arrays 

    def write_per_axis_instances(
        self,
        per_axis: Mapping[str, np.ndarray],
    ) -> None:
        """Per-axis 2D instance maps under ``per_axis_instances/``.

        Shape conventions match the producer + uSegment3D upstream:

        * ``xy``: ``(Z, Y, X) uint32`` (slicing along Z).
        * ``xz``: ``(Y, Z, X) uint32`` (slicing along Y).
        * ``yz``: ``(X, Z, Y) uint32`` (slicing along X).
        """
        self._check_not_finalized()
        if not per_axis:
            return
        if self._per_axis_group is None:
            self._per_axis_group = _retry_on_oserror(
                lambda: self._root.create_group("per_axis_instances"),
                what="create per_axis_instances group",
            )
        for axis_name, arr_data in per_axis.items():
            if axis_name not in ("xy", "xz", "yz"):
                raise ValueError(
                    f"per_axis_instances axis must be 'xy'|'xz'|'yz'; "
                    f"got {axis_name!r}"
                )
            if arr_data.ndim != 3:
                raise ValueError(
                    f"per_axis_instances/{axis_name}: expected 3D array; "
                    f"got shape {arr_data.shape}"
                )
            arr = _retry_on_oserror(
                lambda axis_name=axis_name, arr_data=arr_data: (
                    self._per_axis_group.create_dataset(
                        axis_name,
                        shape=arr_data.shape,
                        dtype=np.uint32,
                        chunks=tuple(min(c, s) for c, s in zip(
                            self.chunks, arr_data.shape,
                        )),
                        overwrite=False,
                    )
                ),
                what=f"create per_axis_instances/{axis_name}",
            )
            arr[:] = arr_data.astype(np.uint32, copy=False)
            self._arrays[f"per_axis_instances/{axis_name}"] = arr

    def write_instance_labels(
        self, data: np.ndarray, region: Optional[Region3D] = None,
    ) -> None:
        """``(Z, Y, X) uint32`` final 3D instance labels."""
        self._check_not_finalized()
        if data.ndim != 3:
            raise ValueError(
                f"instance_labels: expected (Z, Y, X); got shape {data.shape}"
            )
        arr = self._get_or_create(
            "instance_labels", self.volume_shape, np.uint32, self.chunks,
        )
        self._write_into(arr, data.astype(np.uint32, copy=False), region)

    #  Finalize 

    def finalize(
        self,
        *,
        extra_attrs: Optional[Mapping[str, Any]] = None,
        heads_present: Optional[Sequence[str]] = None,
        instance_assembly_backend: Optional[str] = None,
        uncertainty_provider: Optional[str] = None,
    ) -> Path:
        """Write the root ``.zattrs`` manifest and lock further writes.

        Required: ``teacher_argmax`` must have been written. The
        manifest enumerates which optional arrays / heads were
        populated; consumers use that to know what's present without
        scanning the zarr store.

        Parameters
        ----------
        extra_attrs
            Extra root attributes merged into ``.zattrs``.
        heads_present
            Override the auto-derived ``heads_present`` list. Use when
            the caller knows the head set independently (e.g. from
            the model's ``head_names``).
        instance_assembly_backend
            Name of the assembly backend that produced
            ``instance_labels`` / ``per_axis_instances``. ``None`` if
            no assembly ran. Written to the manifest.
        uncertainty_provider
            Name of the uncertainty provider that produced
            ``tta_variance_map`` / ``tta_entropy_map``. ``None`` if
            none ran.

        Returns
        -------
        Path
            ``self.output_path`` (the directory containing
            ``.zattrs``).
        """
        if self._finalized:
            raise RuntimeError(
                "PredictionZarrWriter.finalize already called."
            )
        if "teacher_argmax" not in self._arrays:
            raise RuntimeError(
                "teacher_argmax was never written; prediction_v1 requires "
                "it. Call write_teacher_argmax before finalize."
            )

        if heads_present is None:
            heads_present = self._derive_heads_present()

        manifest: Dict[str, Any] = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "volseg_version": _volseg_version(),
            "created_at": self._created_at,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "inference_mode": self.inference_mode,
            "inference_config": self.inference_config,
            "source_volume_path": self.source_volume_path,
            "source_volume_hash": self.source_volume_hash,
            "volume_shape": list(self.volume_shape),
            "voxel_size_nm": (
                list(self.voxel_size_nm) if self.voxel_size_nm is not None
                else None
            ),
            "class_metadata": self.class_metadata,
            "heads_present": list(heads_present),
            "instance_assembly_backend": instance_assembly_backend,
            "uncertainty_provider": uncertainty_provider,
        }
        if extra_attrs:
            manifest.update(dict(extra_attrs))

        # Write the manifest to root attrs. Prefer a single batched metadata
        # write (zarr 3 ``update_attributes``) over one-rewrite-per-key, and
        # retry transient FS locks (Windows atomic-replace races). Setting the
        # full manifest is idempotent, so retrying the whole block is safe.
        if hasattr(self._root, "update_attributes"):
            _retry_on_oserror(
                lambda: self._root.update_attributes(dict(manifest)),
                what="zarr root attrs",
            )
        else:
            def _write_manifest() -> None:
                for k, v in manifest.items():
                    self._root.attrs[k] = v

            _retry_on_oserror(_write_manifest, what="zarr root attrs")

        self._finalized = True
        logger.info(
            "Finalized prediction.zarr at %s — heads_present=%s, "
            "instance_backend=%s, uncertainty=%s",
            self.output_path, heads_present, instance_assembly_backend,
            uncertainty_provider,
        )
        return self.output_path

    #  Internals 

    def _derive_heads_present(self) -> List[str]:
        """Map written-array names -> canonical head names.

        Used by :meth:`finalize` to populate the manifest's
        ``heads_present`` list when the caller doesn't override it.
        """
        present = []
        if "teacher_argmax" in self._arrays or "semantic_logits" in self._arrays:
            present.append("semantic")
        if "boundary_map" in self._arrays:
            present.append("boundary")
        if "distance_map" in self._arrays:
            present.append("distance")
        if "sdm_map" in self._arrays:
            present.append("sdm")
        return present

    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError(
                "PredictionZarrWriter is finalized; further writes refused."
            )

    def _get_or_create(
        self,
        name: str,
        shape: Sequence[int],
        dtype: type,
        chunks: Sequence[int],
    ) -> zarr.Array:
        if name in self._arrays:
            return self._arrays[name]
        arr = _retry_on_oserror(
            lambda: self._root.create_dataset(
                name,
                shape=tuple(shape),
                dtype=dtype,
                chunks=tuple(min(c, s) for c, s in zip(chunks, shape)),
                overwrite=False,
            ),
            what=f"create array {name!r}",
        )
        self._arrays[name] = arr
        return arr

    def _write_into(
        self,
        arr: zarr.Array,
        data: np.ndarray,
        region: Optional[Region3D],
    ) -> None:
        if region is None:
            arr[:] = data
        else:
            arr[tuple(region)] = data


#  Standalone helper for in-memory -> zarr persistence 


def volseg_write_prediction_zarr(
    arrays: Mapping[str, Any],
    output_path: Union[str, Path],
    *,
    volume_shape: Optional[Sequence[int]] = None,
    inference_mode: str = "multi_axis",
    class_metadata: Optional[Mapping[str, Any]] = None,
    inference_config: Optional[Mapping[str, Any]] = None,
    source_volume_path: Optional[Union[str, Path]] = None,
    source_volume_hash: Optional[str] = None,
    model_checkpoint_hash: Optional[str] = None,
    voxel_size_nm: Optional[Sequence[float]] = None,
    heads_present: Optional[Sequence[str]] = None,
    instance_assembly_backend: Optional[str] = None,
    uncertainty_provider: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Persist an in-memory ``arrays`` dict as a ``prediction_v1`` zarr.

    """
    if "teacher_argmax" not in arrays:
        raise ValueError(
            "volseg_write_prediction_zarr: 'teacher_argmax' is required "
            "in the arrays dict."
        )
    teacher_argmax = np.asarray(arrays["teacher_argmax"])
    if volume_shape is None:
        volume_shape = teacher_argmax.shape

    writer = PredictionZarrWriter(
        output_path=output_path,
        volume_shape=volume_shape,
        inference_mode=inference_mode,
        class_metadata=class_metadata or {},
        inference_config=inference_config,
        source_volume_path=source_volume_path,
        source_volume_hash=source_volume_hash,
        model_checkpoint_hash=model_checkpoint_hash,
        voxel_size_nm=voxel_size_nm,
        overwrite=overwrite,
    )

    writer.write_teacher_argmax(teacher_argmax)
    if "teacher_probs" in arrays:
        writer.write_teacher_probs(np.asarray(arrays["teacher_probs"]))
    if "semantic_logits" in arrays:
        writer.write_semantic_logits(np.asarray(arrays["semantic_logits"]))
    if "semantic_probs" in arrays:
        writer.write_semantic_probs(np.asarray(arrays["semantic_probs"]))
    if "boundary_map" in arrays:
        writer.write_boundary_map(np.asarray(arrays["boundary_map"]))
    if "distance_map" in arrays:
        writer.write_distance_map(np.asarray(arrays["distance_map"]))
    if "sdm_map" in arrays:
        writer.write_sdm_map(np.asarray(arrays["sdm_map"]))
    if "tta_variance_map" in arrays:
        writer.write_tta_variance_map(np.asarray(arrays["tta_variance_map"]))
    if "tta_entropy_map" in arrays:
        writer.write_tta_entropy_map(np.asarray(arrays["tta_entropy_map"]))
    if "per_axis_instances" in arrays:
        per_axis = {
            k: np.asarray(v) for k, v in arrays["per_axis_instances"].items()
        }
        writer.write_per_axis_instances(per_axis)
    if "instance_labels" in arrays:
        writer.write_instance_labels(np.asarray(arrays["instance_labels"]))

    return writer.finalize(
        heads_present=heads_present,
        instance_assembly_backend=instance_assembly_backend,
        uncertainty_provider=uncertainty_provider,
    )


__all__ = [
    "PREDICTION_SCHEMA_VERSION",
    "PredictionZarrWriter",
    "volseg_write_prediction_zarr",
]
