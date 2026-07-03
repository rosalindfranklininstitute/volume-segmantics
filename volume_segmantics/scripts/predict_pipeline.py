#!/usr/bin/env python
"""Pipeline prediction CLI — wraps :func:`volume_segmantics.api.predict`.


The CLI reads:

* ``<settings_dir>/2d_model_predict_settings.yaml`` — the legacy
  predict settings.
* ``<settings_dir>/pipeline.yaml`` (optional) — when present, drives
  the inference mode + instance-assembly backend.

and calls :func:`volume_segmantics.api.predict` with overrides taken
from CLI flags. 
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import volume_segmantics.utilities.config as cfg
from volume_segmantics.api import predict
from volume_segmantics.data import get_settings_data
from volume_segmantics.data.pipeline_loader import (
    legacy_settings_to_pipeline_config,
    load_pipeline_yaml,
)


def _resolve_pipeline_config(settings, root_path: Path):
    pipeline_yaml_path = root_path / cfg.SETTINGS_DIR / "pipeline.yaml"
    if pipeline_yaml_path.exists():
        logging.info("Loading pipeline config from %s", pipeline_yaml_path)
        return load_pipeline_yaml(pipeline_yaml_path)
    logging.info(
        "No pipeline.yaml found; synthesising semantic-only pipeline "
        "config from legacy settings."
    )
    return legacy_settings_to_pipeline_config(settings)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline prediction CLI: run api.predict with optional "
            "instance-assembly + zarr output."
        ),
    )
    parser.add_argument("model_path", type=str, help="Path to the trained checkpoint.")
    parser.add_argument("data_path", type=str, help="Path to the input volume.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path.cwd()),
        help="Project root containing volseg-settings/.",
    )
    parser.add_argument(
        "--inference-mode", type=str, default=None,
        help="Override pipeline_config.prediction.inference_mode "
             "(single_axis | multi_axis | sliding_window | tta_uncertainty).",
    )
    parser.add_argument(
        "--instance-assembly-backend", type=str, default=None,
        help="Override the pipeline_config.instance_assembly.backend.",
    )
    parser.add_argument(
        "--output-zarr", type=str, default=None,
        help="Path to write the prediction_v1 zarr.",
    )
    parser.add_argument(
        "--overwrite-zarr", action="store_true", default=False,
        help="Clobber an existing zarr at --output-zarr.",
    )
    parser.add_argument(
        "--voxel-size-zyx", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        help="Voxel size in physical units (z y x). Default: 1 1 1.",
    )
    args = parser.parse_args()

    root_path = Path(args.data_dir).resolve()
    settings_path = root_path / cfg.SETTINGS_DIR / cfg.PREDICTION_SETTINGS_FN
    settings = get_settings_data(settings_path)
    settings.pipeline_config = _resolve_pipeline_config(settings, root_path)

    result = predict(
        model_path=args.model_path,
        data_vol_path=args.data_path,
        settings=settings,
        pipeline_config=settings.pipeline_config,
        inference_mode=args.inference_mode,
        instance_assembly_backend=args.instance_assembly_backend,
        output_zarr=Path(args.output_zarr) if args.output_zarr else None,
        return_arrays=False,
        voxel_size=tuple(args.voxel_size_zyx),
        overwrite_zarr=args.overwrite_zarr,
    )
    if result.output_zarr is not None:
        print(f"[OK] prediction zarr written: {result.output_zarr}")
    else:
        print("[OK] prediction complete (no --output-zarr supplied; "
              "result.arrays may be empty unless return_arrays=True).")


if __name__ == "__main__":
    main()
