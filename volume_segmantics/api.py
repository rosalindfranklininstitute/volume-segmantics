"""Public Python API 3.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import get_settings_data
from volume_segmantics.data.pipeline_loader import (
    PipelineConfig,
    legacy_settings_to_pipeline_config,
    load_pipeline_yaml,
)
from volume_segmantics.inference.instance_assembly import (
    AssemblyConfig,
    PredictionBundle,
    get_backend as get_instance_assembly_backend,
)
from volume_segmantics.prediction import (
    MultiAxisTTAProvider,
    PredictionZarrWriter,
    UncertaintyOutputs,
    get_inference_mode,
    volseg_write_prediction_zarr,
)
from volume_segmantics.prediction.per_axis_instances import (
    get_producer as get_per_axis_producer,
    select_producer_name,
)


logger = logging.getLogger(__name__)


# PredictionResult 


@dataclass
class PredictionResult:
    """Bundle of arrays + manifest from :func:`predict`.

    Attributes
    ----------
    arrays
        Dict keyed exactly the same as the ``prediction_v1`` zarr
        layout: ``"teacher_argmax"``, ``"teacher_probs"``,
        ``"semantic_logits"``, ``"semantic_probs"``,
        ``"boundary_map"``, ``"distance_map"``, ``"sdm_map"``,
        ``"tta_variance_map"``, ``"tta_entropy_map"``,
        ``"per_axis_instances"`` (a nested dict), ``"instance_labels"``.
        Optional keys are absent (not ``None``) when the mode didn't
        produce them.
    manifest
        Provenance + mode metadata. Keys mirror the zarr manifest:
        ``"schema_version"``, ``"inference_mode"``, ``"heads_present"``,
        ``"uncertainty_provider"``, ``"instance_assembly_backend"``,
        ``"volume_shape"``, ``"voxel_size_nm"``, ``"class_metadata"``.
    output_zarr
        Path to the written ``prediction.zarr`` if
        :func:`predict` was called with ``output_zarr=...``. ``None``
        otherwise.
    """

    arrays: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    output_zarr: Optional[Path] = None


#  Predict 


def predict(
    model_path: Union[str, Path],
    data_vol_path: Union[str, Path],
    *,
    settings: Optional[SimpleNamespace] = None,
    settings_path: Optional[Union[str, Path]] = None,
    pipeline_config: Optional[PipelineConfig] = None,
    inference_mode: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    output_zarr: Optional[Union[str, Path]] = None,
    return_arrays: bool = True,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    instance_assembly_backend: Optional[str] = None,
    uncertainty_provider: Optional[str] = None,
    overwrite_zarr: bool = False,
) -> PredictionResult:
    """Run prediction against a trained vol-seg model.

    Parameters
    ----------
    model_path
        Path to a ``.pytorch`` checkpoint (b3 or v0.4 format).
    data_vol_path
        Path to the input volume (HDF5 / TIFF / MRC).
    settings
        Already-loaded legacy settings. Pass either this or
        ``settings_path``.
    settings_path
        Path to ``volseg-settings/2d_model_predict_settings.yaml``.
        Read when ``settings`` is ``None``.
    pipeline_config
        Optional :class:`PipelineConfig`. When ``None``, synthesised
        from the legacy settings via
        :func:`legacy_settings_to_pipeline_config`.
    inference_mode
        Override the ``pipeline_config.prediction.inference_mode``.
        Defaults to the pipeline config's value if both ``pipeline_config``
        is provided and this is ``None``.
    output_path
        Legacy single-file output (TIFF/HDF5/MRC) — written by
        :class:`VolSeg2DPredictionManager` for back-compat. Pass
        ``None`` to skip.
    output_zarr
        ``prediction_v1`` zarr output path. Pass ``None`` to skip.
        Independent of ``return_arrays`` — both can be set together.
    return_arrays
        When ``True``, the returned :class:`PredictionResult` carries
        the in-memory arrays. When ``False``, ``arrays`` is empty
        (caller must read from the zarr or single-file output).
    voxel_size
        Voxel size in nanometres ``(z, y, x)``. Written to the zarr
        manifest.
    instance_assembly_backend
        Name of the instance-assembly backend to run after the
        prediction mode (e.g. ``"usegment3d"``). When ``None``, falls
        back to ``pipeline_config.instance_assembly.backend``; when
        both are ``None``, no assembly runs. The backend reads
        :attr:`PipelineConfig.instance_assembly.params` for its
        ``__init__`` kwargs (e.g. ``axes``, ``params_overrides``).
    uncertainty_provider
        Override the auto-derived uncertainty provider. ``None``
        means auto: ``MultiAxisTTAProvider`` for
        ``tta_uncertainty`` mode, else no provider.
    overwrite_zarr
        Clobber an existing zarr at ``output_zarr``.

    Returns
    -------
    PredictionResult
        Arrays + manifest + zarr path. Always returned.

    """
    # Local import to avoid a circular dep against the trainer's
    # eager imports of api.predict.
    from volume_segmantics.model.operations.vol_seg_2d_predictor import (
        VolSeg2dPredictor,
    )

    model_path = Path(model_path)
    data_vol_path = Path(data_vol_path)

    if settings is None:
        if settings_path is None:
            raise ValueError(
                "predict: provide either ``settings`` or ``settings_path``."
            )
        settings = get_settings_data(Path(settings_path))

    if pipeline_config is None:
        pipeline_config = legacy_settings_to_pipeline_config(settings)
    settings.pipeline_config = pipeline_config

    if inference_mode is None:
        inference_mode = pipeline_config.prediction.inference_mode

    # Validate the requested mode against the registry. Raises if
    # unknown — better fail-fast than silently fall through to a
    # default branch.
    mode_descriptor = get_inference_mode(inference_mode)
    logger.info(
        "api.predict: mode=%s (populates=%s)",
        inference_mode, sorted(mode_descriptor.populates),
    )

    # Auto-pick uncertainty provider for tta_uncertainty.
    if (
        uncertainty_provider is None
        and mode_descriptor.requires_uncertainty_provider
    ):
        uncertainty_provider = "tta_uncertainty"

    #  Load data + predictor 
    data_vol, _ = utils.get_numpy_from_path(
        data_vol_path,
        internal_path=getattr(settings, "data_hdf5_path", "/data"),
    )
    logger.info(
        "Loaded volume %s shape=%s dtype=%s",
        data_vol_path, data_vol.shape, data_vol.dtype,
    )

    predictor = VolSeg2dPredictor(str(model_path), settings)

    #  Mode dispatch 
    arrays: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}

    if inference_mode in ("single_axis", "sliding_window"):
        labels, probs, logits = predictor._predict_single_axis(
            data_vol, output_probs=True, axis=utils.Axis.Z,
        )
        arrays["teacher_argmax"] = labels.astype(np.uint8, copy=False)
        if probs is not None:
            # _predict_single_axis returns probs as channel-LAST
            # (Z, Y, X, C); transpose to (C, Z, Y, X) for the zarr
            # writer + downstream consumers.
            probs_cf = np.transpose(probs, (3, 0, 1, 2)).astype(np.float32)
            arrays["teacher_probs"] = probs_cf
            arrays["semantic_probs"] = probs_cf.copy()
        if logits is not None:
            arrays["semantic_logits"] = np.transpose(
                logits, (3, 0, 1, 2),
            ).astype(np.float32)

    elif inference_mode == "multi_axis":
        labels, probs = predictor._predict_3_ways_max_probs(
            data_vol,
            z_smooth_sigma=getattr(settings, "z_smooth_sigma", None),
        )[:2]
        arrays["teacher_argmax"] = labels.astype(np.uint8, copy=False)
        if probs is not None:
            # _predict_3_ways_max_probs returns max-prob-per-voxel
            # (Z, Y, X) float16 — the per-class probability stack
            # is not preserved across the merge in  multi_axis
            # mode therefore writes ``teacher_argmax`` only;
            # ``teacher_probs`` is omitted (consumer reads from the
            # tta_uncertainty mode if it wants the full stack).
            pass
        _stash_per_head_maps(predictor, arrays)

    elif inference_mode == "tta_uncertainty":
        # Self-contained 3-axis loop that captures the per-axis
        # softmax-probability stacks. Hands them to
        # MultiAxisTTAProvider for variance + entropy + max-prob
        # merge. Same as multi_axis for the merged result;
        # adds the uncertainty maps on top.
        provider_uo = _run_tta_uncertainty(
            predictor, data_vol,
        )
        arrays.update(provider_uo.as_dict())
        # Per-axis stash dropped from the arrays dict — we don't
        # write it to the zarr today (it's stash-only on the
        # UncertaintyOutputs for forward-compat).

    else:
        raise NotImplementedError(
            f"api.predict: inference_mode {inference_mode!r} routing not "
            f"yet wired through the api module. Use the legacy "
            f"`model-predict-2d` script for now."
        )

    #  Instance assembly dispatch  
    # kwarg overrides pipeline_config; both null -> no assembly.
    effective_backend = instance_assembly_backend
    if effective_backend is None:
        effective_backend = pipeline_config.instance_assembly.backend
    if effective_backend is not None:
        _run_instance_assembly(
            arrays=arrays,
            backend_name=effective_backend,
            pipeline_config=pipeline_config,
            voxel_size=voxel_size,
        )

    #  Optional zarr write 
    written_zarr: Optional[Path] = None
    if output_zarr is not None:
        written_zarr = volseg_write_prediction_zarr(
            arrays,
            output_path=Path(output_zarr),
            volume_shape=tuple(data_vol.shape),
            inference_mode=inference_mode,
            class_metadata={"label_codes": predictor.label_codes},
            inference_config={
                "quality": getattr(settings, "quality", None),
                "use_sliding_window": getattr(
                    settings, "use_sliding_window", False,
                ),
            },
            source_volume_path=data_vol_path,
            voxel_size_nm=voxel_size,
            uncertainty_provider=uncertainty_provider,
            instance_assembly_backend=effective_backend,
            overwrite=overwrite_zarr,
        )

    # Optional legacy single-file output 
    # Defer the legacy single-file path to the existing
    # `model-predict-2d` script. The api.predict caller can chain
    # this themselves via VolSeg2DPredictionManager if needed; the
    # b3 path exists primarily for the zarr + return_arrays flow.

    #  Manifest 
    heads_present = ["semantic"]
    for head_name in ("boundary", "distance", "sdm"):
        if f"{head_name}_map" in arrays:
            heads_present.append(head_name)
    manifest: Dict[str, Any] = {
        "schema_version": "prediction_v1",
        "inference_mode": inference_mode,
        "heads_present": heads_present,
        "uncertainty_provider": uncertainty_provider,
        "instance_assembly_backend": effective_backend,
        "volume_shape": list(data_vol.shape),
        "voxel_size_nm": list(voxel_size),
        "class_metadata": {"label_codes": predictor.label_codes},
    }

    if not return_arrays:
        arrays = {}

    return PredictionResult(
        arrays=arrays,
        manifest=manifest,
        output_zarr=written_zarr,
    )




def _stash_per_head_maps(predictor: Any, arrays: Dict[str, Any]) -> None:
    """Read multi-head outputs from the predictor and write them into ``arrays``.

    """
    if not predictor._is_multitask_model():
        return
    task_outputs = predictor.get_additional_task_outputs() or {}
    if not task_outputs:
        return
    underlying = predictor.model
    if hasattr(underlying, "module"):
        underlying = underlying.module
    head_names = list(getattr(underlying, "head_names", ()))
    if len(head_names) < 2:
        return
    # task1-> head_names[1], task2-> head_names[2], … (head 0 = semantic).
    sorted_keys = sorted(
        task_outputs,
        key=lambda k: int(k.replace("task", "")) if k.startswith("task") else 0,
    )
    sigmoid_heads = {"boundary"}
    for task_idx, task_key in enumerate(sorted_keys, start=1):
        if task_idx >= len(head_names):
            break
        head_name = head_names[task_idx]
        task_data = task_outputs[task_key]
        # Source-field preference:
        # - boundary: post-sigmoid 'probs' (matches PredictionBundle's
        #   "sigmoid probs" contract for boundary_map).
        # - distance / sdm: prefer raw 'logits', but the predictor's
        #   multi-axis merge (_predict_12_ways_max_probs) only retains
        #   'labels' + 'probs' on the merged result — fall back to
        #   'probs' (sigmoid-distorted but still a smooth signal that
        #   watershed can find peaks in).
        if head_name in sigmoid_heads:
            raw = task_data.get("probs")
        else:
            raw = task_data.get("logits")
            if raw is None:
                raw = task_data.get("probs")
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 4:
            arr = np.transpose(arr, (1, 0, 2, 3))
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        arrays[f"{head_name}_map"] = arr


# tta_uncertainty: self-contained 3-axis loop 


def _run_tta_uncertainty(
    predictor: Any, data_vol: np.ndarray,
) -> UncertaintyOutputs:
    """Run a 3-axis predict loop and feed the stacks to MultiAxisTTAProvider.

    This duplicates a small slice of v0.4's
    :meth:`VolSeg2dPredictor._predict_3_ways_max_probs` to capture
    the per-axis probability stacks before the max-prob merge — the
    existing method discards them. Bit-similar to multi_axis for the
    merged result + adds the uncertainty maps.
    """
    axis_tags = [
        ("z_rot0", utils.Axis.Z),
        ("y_rot0", utils.Axis.Y),
        ("x_rot0", utils.Axis.X),
    ]
    per_axis_probs: Dict[str, np.ndarray] = {}
    for tag, axis_enum in axis_tags:
        logger.info("api.predict[tta_uncertainty]: predicting axis %s", tag)
        _labels, probs, _logits = predictor._predict_single_axis(
            data_vol, output_probs=True, axis=axis_enum,
        )
        if probs is None:
            raise RuntimeError(
                f"api.predict[tta_uncertainty]: axis {tag} returned no "
                f"probability stack — predictor returned probs=None"
            )
        # Transpose (Z, Y, X, C) -> (C, Z, Y, X).
        per_axis_probs[tag] = np.transpose(
            probs.astype(np.float32, copy=False), (3, 0, 1, 2),
        )

    provider = MultiAxisTTAProvider(store_per_axis=False)
    return provider.compute_from_stacks(per_axis_probs)


#  Instance assembly dispatch (B3.G.5) 


def _run_instance_assembly(
    *,
    arrays: Dict[str, Any],
    backend_name: str,
    pipeline_config: PipelineConfig,
    voxel_size: Tuple[float, float, float],
) -> None:
    """Run per-axis 2D producer + 3D assembler; mutate ``arrays`` in place.

    Adds two keys to ``arrays``:

    * ``per_axis_instances`` — ``{axis: ndarray}`` from the producer.
    * ``instance_labels`` — ``(Z, Y, X) uint32`` from the assembler.

    Configuration sources:

    * Producer kwargs: ``pipeline_config.prediction.per_axis_instances.params``.
    * Assembler kwargs: ``pipeline_config.instance_assembly.params``.
      The reserved keys ``foreground_class_ids`` (lifted onto
      :class:`AssemblyConfig`) and ``axes`` (passed to the backend's
      ``axes`` kwarg) are popped first; the rest are passed verbatim
      to the backend's ``__init__``.
    """
    if "teacher_argmax" not in arrays:
        raise RuntimeError(
            "instance assembly: arrays dict has no 'teacher_argmax' "
            "(produced by every inference mode); cannot proceed."
        )

    available_heads = ["semantic"]
    if "boundary_map" in arrays:
        available_heads.append("boundary")
    if "distance_map" in arrays:
        available_heads.append("distance")
    if "sdm_map" in arrays:
        available_heads.append("sdm")

    bundle = PredictionBundle(
        semantic_argmax=arrays.get("teacher_argmax"),
        semantic_probs=arrays.get("semantic_probs"),
        boundary_map=arrays.get("boundary_map"),
        distance_map=arrays.get("distance_map"),
        sdm_map=arrays.get("sdm_map"),
    )

    paxis_cfg = pipeline_config.prediction.per_axis_instances
    producer_name = select_producer_name(
        paxis_cfg.producer, enabled_heads=available_heads,
    )
    logger.info(
        "api.predict[instance_assembly]: producer=%s axes=%s backend=%s",
        producer_name, list(paxis_cfg.axes), backend_name,
    )
    producer = get_per_axis_producer(producer_name)()
    per_axis_maps = producer.produce(
        bundle,
        axes=tuple(paxis_cfg.axes),
        params=dict(paxis_cfg.params),
    )
    bundle.per_axis_instances = per_axis_maps
    arrays["per_axis_instances"] = per_axis_maps

    asm_params = dict(pipeline_config.instance_assembly.params or {})
    fg_class_ids = asm_params.pop("foreground_class_ids", None)
    if fg_class_ids is not None:
        fg_class_ids = tuple(int(c) for c in fg_class_ids)

    backend_cls = get_instance_assembly_backend(backend_name)
    assembler = backend_cls(**asm_params)
    assembly_cfg = AssemblyConfig(
        foreground_class_ids=fg_class_ids,
        voxel_size=tuple(voxel_size),
    )
    instance_labels = assembler.assemble(bundle, assembly_cfg)
    arrays["instance_labels"] = np.asarray(instance_labels, dtype=np.uint32)


def load_data_extra(path: Union[str, Path]) -> Any:
    raise NotImplementedError(
        "volume_segmantics.api.load_data_extra is deferred"
    )


__all__ = [
    "PredictionResult",
    "load_data_extra",
    "predict",
]
