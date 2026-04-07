#!/usr/bin/env python
"""
Config-driven real-data regression testing harness framework.

Supports reusable step types:
- train          : run train_2d_model
- predict        : run predict_2d_model with a registered model
- unlabeled_slicer : slice unlabeled volumes into 2D images
- pytest         : run targeted pytest checks
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class HarnessContext:
    root_dir: Path
    output_root: Path
    image_path: Path
    label_path: Optional[Path]
    unlabeled_path: Optional[Path]
    unlabeled_paths: List[Path]
    train_template: Path
    pred_template: Path
    global_train_overrides: Dict[str, Any]
    global_pred_overrides: Dict[str, Any]
    task2_path: Optional[Path] = None
    task3_path: Optional[Path] = None
    dry_run: bool = False
    models: Dict[str, Path] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _run_command(command: List[str], cwd: Path, root_dir: Path, dry_run: bool = False) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    if dry_run:
        print(f"\n[DRY-RUN] {' '.join(command)}")
        return
    print(f"\n[RUN] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def _write_settings_pair(
    ctx: HarnessContext,
    run_dir: Path,
    train_overrides: Optional[Dict[str, Any]] = None,
    pred_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Path, Path]:
    settings_dir = run_dir / "volseg-settings"
    settings_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = _load_yaml(ctx.train_template)
    pred_cfg = _load_yaml(ctx.pred_template)

    _deep_update(train_cfg, copy.deepcopy(ctx.global_train_overrides))
    _deep_update(pred_cfg, copy.deepcopy(ctx.global_pred_overrides))
    if train_overrides:
        _deep_update(train_cfg, copy.deepcopy(train_overrides))
    if pred_overrides:
        _deep_update(pred_cfg, copy.deepcopy(pred_overrides))

    train_settings = settings_dir / "2d_model_train_settings.yaml"
    pred_settings = settings_dir / "2d_model_predict_settings.yaml"
    _save_yaml(train_settings, train_cfg)
    _save_yaml(pred_settings, pred_cfg)
    return train_settings, pred_settings


def _expected_model_path(run_dir: Path, train_cfg: Dict[str, Any]) -> Path:
    model_cfg = train_cfg.get("model", {})
    model_type = str(model_cfg.get("type", "U_Net")).upper()
    model_output_fn = str(train_cfg.get("model_output_fn", "trained_2d_model"))
    return run_dir / f"{date.today()}_{model_type}_{model_output_fn}.pytorch"


def _assert_globs(
    run_dir: Path, should_exist: List[str], should_not_exist: List[str], dry_run: bool = False
) -> None:
    if dry_run:
        if should_exist:
            print(f"[DRY-RUN] Skip should_exist assertions in {run_dir}: {should_exist}")
        if should_not_exist:
            print(f"[DRY-RUN] Skip should_not_exist assertions in {run_dir}: {should_not_exist}")
        return
    for pattern in should_exist:
        matches = list(run_dir.glob(pattern))
        if not matches:
            raise AssertionError(
                f"Expected files matching '{pattern}' in {run_dir}, found none."
            )
    for pattern in should_not_exist:
        matches = list(run_dir.glob(pattern))
        if matches:
            raise AssertionError(
                f"Did not expect files matching '{pattern}' in {run_dir}, "
                f"found: {[str(item) for item in matches]}"
            )


def _resolve_token(raw: str, ctx: HarnessContext, run_dir: Path) -> str:
    if not isinstance(raw, str):
        return raw
    if raw.startswith("${artifact:") and raw.endswith("}"):
        key = raw[len("${artifact:"):-1]
        if key not in ctx.artifacts:
            raise KeyError(f"Missing artifact key '{key}'")
        return str(ctx.artifacts[key])
    if raw.startswith("${model:") and raw.endswith("}"):
        key = raw[len("${model:"):-1]
        if key not in ctx.models:
            raise KeyError(f"Missing model key '{key}'")
        return str(ctx.models[key])
    if raw.startswith("${path:") and raw.endswith("}"):
        key = raw[len("${path:"):-1]
        mapping = {
            "image": ctx.image_path,
            "label": ctx.label_path,
            "unlabeled": ctx.unlabeled_path,
            "unlabeled_all": ctx.unlabeled_paths,
            "task2": ctx.task2_path,
            "task3": ctx.task3_path,
            "run_dir": run_dir,
        }
        if key not in mapping or mapping[key] is None:
            raise KeyError(f"Missing path key '{key}'")
        value = mapping[key]
        if isinstance(value, list):
            return [str(item) for item in value]
        return str(value)
    return raw


def _resolve_tokens(values: List[str], ctx: HarnessContext, run_dir: Path) -> List[str]:
    resolved: List[str] = []
    for value in values:
        token_value = _resolve_token(value, ctx, run_dir)
        if isinstance(token_value, list):
            resolved.extend(str(item) for item in token_value)
        else:
            resolved.append(str(token_value))
    return resolved


def _run_train_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    name = str(step_cfg["name"])
    model_key = str(step_cfg.get("model_key", name))
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_overrides = step_cfg.get("overrides", {}).get("train", {})
    pred_overrides = step_cfg.get("overrides", {}).get("predict", {})
    train_settings, _ = _write_settings_pair(ctx, run_dir, train_overrides, pred_overrides)
    train_cfg = _load_yaml(train_settings)

    image_value = step_cfg.get("image", "${path:image}")
    label_value = step_cfg.get("label", "${path:label}")

    cmd = [
        sys.executable,
        "-m",
        "volume_segmantics.scripts.train_2d_model",
        "--data",
        _resolve_token(str(image_value), ctx, run_dir),
        "--data_dir",
        str(run_dir),
    ]
    if label_value is not None:
        cmd.extend(["--labels", _resolve_token(str(label_value), ctx, run_dir)])

    extra_args = step_cfg.get("args", [])
    cmd.extend(_resolve_tokens(list(extra_args), ctx, run_dir))
    _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)

    model_path = _expected_model_path(run_dir, train_cfg)
    if not ctx.dry_run and not model_path.exists():
        # Training may have crossed midnight ? try glob for any date
        model_cfg = train_cfg.get("model", {})
        model_type = str(model_cfg.get("type", "U_Net")).upper()
        model_output_fn = str(train_cfg.get("model_output_fn", "trained_2d_model"))
        pattern = f"*_{model_type}_{model_output_fn}.pytorch"
        candidates = sorted(run_dir.glob(pattern))
        if candidates:
            model_path = candidates[-1]  # most recent
        else:
            raise FileNotFoundError(
                f"Training finished but model not found: {model_path} "
                f"(also tried glob '{pattern}' in {run_dir})"
            )
    ctx.models[model_key] = model_path
    status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
    print(f"{status} Registered model '{model_key}' -> {model_path}")


def _run_predict_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    name = str(step_cfg["name"])
    model_path_value = step_cfg.get("model_path")
    if model_path_value is not None:
        model_path = Path(_resolve_token(str(model_path_value), ctx, ctx.output_root)).resolve()
        if not ctx.dry_run and not model_path.exists():
            raise FileNotFoundError(
                f"Predict step '{name}' model_path does not exist: {model_path}"
            )
    else:
        model_key = str(step_cfg["model_key"])
        if model_key not in ctx.models:
            raise KeyError(f"Predict step '{name}' references unknown model_key '{model_key}'")
        model_path = ctx.models[model_key]
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_overrides = step_cfg.get("overrides", {}).get("train", {})
    pred_overrides = step_cfg.get("overrides", {}).get("predict", {})
    _write_settings_pair(ctx, run_dir, train_overrides, pred_overrides)

    image_value = step_cfg.get("image", "${path:image}")
    cmd = [
        sys.executable,
        "-m",
        "volume_segmantics.scripts.predict_2d_model",
        str(model_path),
        _resolve_token(str(image_value), ctx, run_dir),
        "--data_dir",
        str(run_dir),
    ]
    extra_args = step_cfg.get("args", [])
    cmd.extend(_resolve_tokens(list(extra_args), ctx, run_dir))
    _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)

    for artifact_cfg in step_cfg.get("register_artifacts", []):
        key = str(artifact_cfg["key"])
        rel_path = str(artifact_cfg["path"])
        artifact_path = run_dir / rel_path
        ctx.artifacts[key] = artifact_path
        status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
        print(f"{status} Registered artifact '{key}' -> {artifact_path}")

    assert_cfg = step_cfg.get("assert_outputs", {})
    _assert_globs(
        run_dir=run_dir,
        should_exist=assert_cfg.get("should_exist_globs", []),
        should_not_exist=assert_cfg.get("should_not_exist_globs", []),
        dry_run=ctx.dry_run,
    )
    status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
    print(f"{status} Assertions passed for '{name}'")


def _run_unlabeled_slicer_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    name = str(step_cfg["name"])
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_overrides = step_cfg.get("overrides", {}).get("train", {})
    _write_settings_pair(ctx, run_dir, train_overrides, {})

    output_key = str(step_cfg["output_key"])
    output_rel = str(step_cfg.get("output_dir", "unlabeled_data"))
    output_dir = run_dir / output_rel
    output_dir.mkdir(parents=True, exist_ok=True)

    volumes_cfg = step_cfg.get("unlabeled_volumes")
    if volumes_cfg is None:
        single_value = step_cfg.get("unlabeled_volume")
        if single_value is not None:
            volumes_cfg = [single_value]
        else:
            volumes_cfg = ["${path:unlabeled_all}"]
    resolved_volumes = _resolve_tokens(list(volumes_cfg), ctx, run_dir)
    if not resolved_volumes:
        raise ValueError(f"Unlabeled slicer step '{name}' resolved no unlabeled volumes.")

    for idx, raw_volume in enumerate(resolved_volumes):
        volume_path = Path(raw_volume).resolve()
        if not ctx.dry_run and not volume_path.exists():
            raise FileNotFoundError(f"Unlabeled volume does not exist: {volume_path}")
        step_output_dir = output_dir / f"vol_{idx:02d}_{volume_path.stem}"
        step_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "volume_segmantics.scripts.train_2d_model",
            "--mode",
            "slicer",
            "--data",
            str(volume_path),
            "--unlabeled_data_dir",
            str(step_output_dir),
            "--data_dir",
            str(run_dir),
        ]
        extra_args = step_cfg.get("args", [])
        cmd.extend(_resolve_tokens(list(extra_args), ctx, run_dir))
        _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    if not ctx.dry_run:
        for idx, raw_volume in enumerate(resolved_volumes):
            volume_path = Path(raw_volume).resolve()
            step_output_dir = output_dir / f"vol_{idx:02d}_{volume_path.stem}"
            for src in sorted(step_output_dir.iterdir()):
                if not src.is_file():
                    continue
                target = merged_dir / src.name
                if target.exists():
                    target = merged_dir / f"{volume_path.stem}_{src.name}"
                shutil.copy2(src, target)
        if not any(merged_dir.iterdir()):
            raise RuntimeError(
                f"Slicer step '{name}' did not generate merged unlabeled slices in {merged_dir}"
            )
    else:
        print(f"[DRY-RUN] Skip merging sliced outputs for '{name}'")

    ctx.artifacts[output_key] = merged_dir
    status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
    print(f"{status} Registered artifact '{output_key}' -> {merged_dir}")


def _run_pytest_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    name = str(step_cfg["name"])
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    test_ids = [str(item) for item in step_cfg.get("tests", [])]
    if not test_ids:
        raise ValueError(f"Pytest step '{name}' requires non-empty 'tests'")

    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", *test_ids]
    extra_args = step_cfg.get("args", [])
    cmd.extend(_resolve_tokens(list(extra_args), ctx, run_dir))
    _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)


def _build_context(root_dir: Path, cfg: Dict[str, Any], dry_run: bool = False) -> HarnessContext:
    paths = cfg.get("paths", {})
    settings = cfg.get("settings_templates", {})
    global_overrides = cfg.get("global_overrides", {})

    image_path = (root_dir / paths.get("image", "training_data/vessels_256cube_DATA.h5")).resolve()
    label_raw = paths.get("label", "training_data/vessels_256cube_LABELS.h5")
    unlabeled_raw = paths.get("unlabeled")
    task2_raw = paths.get("task2")
    task3_raw = paths.get("task3")
    label_path = (root_dir / label_raw).resolve() if label_raw else None
    task2_path = (root_dir / str(task2_raw)).resolve() if task2_raw else None
    task3_path = (root_dir / str(task3_raw)).resolve() if task3_raw else None
    unlabeled_paths: List[Path] = []
    if isinstance(unlabeled_raw, list):
        unlabeled_paths = [(root_dir / str(item)).resolve() for item in unlabeled_raw]
    elif unlabeled_raw:
        unlabeled_paths = [(root_dir / str(unlabeled_raw)).resolve()]
    unlabeled_path = unlabeled_paths[0] if unlabeled_paths else None

    train_template = (
        root_dir / settings.get("train", "volseg-settings/2d_model_train_settings.yaml")
    ).resolve()
    pred_template = (
        root_dir / settings.get("predict", "volseg-settings/2d_model_predict_settings.yaml")
    ).resolve()
    output_root = (root_dir / cfg.get("output_root", "tmp/real_data_smoke")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dry_run and not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")
    if not dry_run and label_path is not None and not label_path.exists():
        raise FileNotFoundError(f"Label path not found: {label_path}")
    for unlabeled_item in unlabeled_paths:
        if not dry_run and not unlabeled_item.exists():
            raise FileNotFoundError(f"Unlabeled path not found: {unlabeled_item}")
    if not dry_run and task2_path is not None and not task2_path.exists():
        raise FileNotFoundError(f"Task2 path not found: {task2_path}")
    if not dry_run and task3_path is not None and not task3_path.exists():
        raise FileNotFoundError(f"Task3 path not found: {task3_path}")
    if not train_template.exists():
        raise FileNotFoundError(f"Train template not found: {train_template}")
    if not pred_template.exists():
        raise FileNotFoundError(f"Predict template not found: {pred_template}")

    return HarnessContext(
        root_dir=root_dir,
        output_root=output_root,
        image_path=image_path,
        label_path=label_path,
        unlabeled_path=unlabeled_path,
        unlabeled_paths=unlabeled_paths,
        train_template=train_template,
        pred_template=pred_template,
        global_train_overrides=global_overrides.get("train", {}),
        global_pred_overrides=global_overrides.get("predict", {}),
        task2_path=task2_path,
        task3_path=task3_path,
        dry_run=dry_run,
    )


def _run_derive_boundary_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    """Derive boundary labels from a segmentation label volume."""
    name = str(step_cfg["name"])
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    input_value = step_cfg.get("input", "${path:label}")
    input_path = _resolve_token(str(input_value), ctx, run_dir)

    output_key = str(step_cfg.get("output_key", "boundary_labels"))
    output_fn = str(step_cfg.get("output_filename", "boundary_labels.h5"))
    output_path = run_dir / output_fn

    width = int(step_cfg.get("width", 3))
    mode = str(step_cfg.get("mode", "3d"))
    hdf5_path = str(step_cfg.get("hdf5_path", "/data"))

    cmd = [
        sys.executable, "-m",
        "volume_segmantics.utilities.derive_boundary_labels",
        str(input_path),
        str(output_path),
        "--width", str(width),
        "--mode", mode,
        "--hdf5_path", hdf5_path,
    ]
    _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)

    ctx.artifacts[output_key] = output_path
    ctx.task2_path = output_path
    status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
    print(f"{status} Registered artifact '{output_key}' -> {output_path}")
    print(f"{status} Updated ${{path:task2}} -> {output_path}")


def _run_derive_distance_step(ctx: HarnessContext, step_cfg: Dict[str, Any]) -> None:
    """Derive distance map or SDF labels from a segmentation label volume."""
    name = str(step_cfg["name"])
    run_dir = ctx.output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    input_value = step_cfg.get("input", "${path:label}")
    input_path = _resolve_token(str(input_value), ctx, run_dir)

    output_key = str(step_cfg.get("output_key", "distance_labels"))
    output_fn = str(step_cfg.get("output_filename", "distance_labels.h5"))
    output_path = run_dir / output_fn

    dist_type = str(step_cfg.get("dist_type", "edt"))
    mode = str(step_cfg.get("mode", "3d"))
    hdf5_path = str(step_cfg.get("hdf5_path", "/data"))
    normalize = bool(step_cfg.get("normalize", False))
    clip_max = step_cfg.get("clip_max", None)

    cmd = [
        sys.executable, "-m",
        "volume_segmantics.utilities.derive_distance_labels",
        str(input_path),
        str(output_path),
        "--type", dist_type,
        "--mode", mode,
        "--hdf5_path", hdf5_path,
    ]
    if normalize:
        cmd.append("--normalize")
    if clip_max is not None:
        cmd.extend(["--clip_max", str(clip_max)])

    _run_command(cmd, cwd=ctx.root_dir, root_dir=ctx.root_dir, dry_run=ctx.dry_run)

    ctx.artifacts[output_key] = output_path
    ctx.task3_path = output_path
    status = "[DRY-RUN]" if ctx.dry_run else "[OK]"
    print(f"{status} Registered artifact '{output_key}' -> {output_path}")
    print(f"{status} Updated ${{path:task3}} -> {output_path}")


STEP_DISPATCH = {
    "train": _run_train_step,
    "predict": _run_predict_step,
    "unlabeled_slicer": _run_unlabeled_slicer_step,
    "pytest": _run_pytest_step,
    "derive_boundary": _run_derive_boundary_step,
    "derive_distance": _run_derive_distance_step,
}


def run_harness(
    config_path: Path,
    dry_run: bool = False,
    path_overrides: Optional[Dict[str, Any]] = None,
    output_suffix: Optional[str] = None,
) -> None:
    root_dir = Path(__file__).resolve().parents[2]
    cfg_path = config_path if config_path.is_absolute() else (root_dir / config_path)
    cfg_path = cfg_path.resolve()
    cfg = _load_yaml(cfg_path)

    # Apply data-profile path overrides (image, label, unlabeled, etc.)
    if path_overrides:
        if "paths" not in cfg:
            cfg["paths"] = {}
        _deep_update(cfg["paths"], path_overrides)

    # Append suffix to output_root to keep profiles separate
    if output_suffix:
        cfg["output_root"] = cfg.get("output_root", "tmp/smoke") + f"_{output_suffix}"

    config_dry_run = bool(cfg.get("dry_run", False))
    effective_dry_run = bool(dry_run or config_dry_run)
    context = _build_context(root_dir, cfg, dry_run=effective_dry_run)
    tests = cfg.get("tests", [])
    if not tests:
        raise ValueError(f"No tests configured in {cfg_path}")

    mode_suffix = " [DRY-RUN]" if effective_dry_run else ""
    print(f"Running {len(tests)} configured steps from {cfg_path}{mode_suffix}")
    for test_cfg in tests:
        test_type = str(test_cfg.get("type", "")).lower()
        name = str(test_cfg.get("name", "<unnamed>"))
        print(f"\n=== {name} ({test_type}) ===")
        handler = STEP_DISPATCH.get(test_type)
        if handler is None:
            raise ValueError(f"Unsupported step type '{test_type}' in '{name}'")
        handler(context, test_cfg)

    print("\nAll configured smoke steps completed successfully.")
    print(f"Outputs: {context.output_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run config-driven real-data smoke scenarios."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path (absolute or relative to repository root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print workflow commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_harness(Path(args.config), dry_run=bool(getattr(args, "dry_run", False)))


if __name__ == "__main__":
    main()
