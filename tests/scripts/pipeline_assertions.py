#!/usr/bin/env python
"""smoke-gate assertion helpers.

"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# Helpers 


def _open_zarr(zarr_path: Path):
    import zarr
    return zarr.open(str(zarr_path), mode="r")


def _load_torch_checkpoint(ckpt_path: Path) -> dict:
    import torch
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


# checkpoint-trained 


def assert_checkpoint_trained(
    ckpt_path: Path, *,
    min_num_batches_tracked: int = 1,
    min_running_var_std: float = 0.01,
) -> None:
    state = _load_torch_checkpoint(ckpt_path)
    sd = state.get("model_state_dict") or state
    nbts = [v for k, v in sd.items() if k.endswith("num_batches_tracked")]
    rvars = [v for k, v in sd.items() if k.endswith("running_var")]
    if not nbts:
        raise AssertionError(
            f"checkpoint-trained: no BN num_batches_tracked tensors found "
            f"in {ckpt_path}; cannot validate."
        )
    if not rvars:
        raise AssertionError(
            f"checkpoint-trained: no BN running_var tensors found in "
            f"{ckpt_path}; cannot validate."
        )
    max_nbts = max(int(t.item()) for t in nbts)
    if max_nbts < min_num_batches_tracked:
        raise AssertionError(
            f"checkpoint-trained: max BN num_batches_tracked={max_nbts} < "
            f"{min_num_batches_tracked}; the checkpoint was saved before "
            f"any training step ran (v0.5 Bug 2 guard)."
        )
    flat = np.concatenate(
        [t.float().detach().cpu().numpy().ravel() for t in rvars],
    )
    rvar_std = float(flat.std())
    rvar_mean = float(flat.mean())
    if rvar_std < min_running_var_std:
        raise AssertionError(
            f"checkpoint-trained: BN running_var std={rvar_std:.6f} < "
            f"{min_running_var_std}; the running_var values are essentially "
            f"constant (mean={rvar_mean:.4f}), suggesting fresh-init "
            f"(v0.5 Bug 2 guard)."
        )
    print(
        f"[OK] checkpoint-trained: {ckpt_path.name} — max nbts={max_nbts}, "
        f"running_var mean={rvar_mean:.4f} std={rvar_std:.4f}"
    )


# zarr-keys 


def assert_zarr_keys(zarr_path: Path, expected_keys: Sequence[str]) -> None:
    """Every name in ``expected_keys`` must appear in the zarr root.

    Names with a slash (``per_axis_instances/xy``) are checked at
    sub-group depth.
    """
    g = _open_zarr(zarr_path)
    missing: List[str] = []
    for key in expected_keys:
        cur = g
        ok = True
        for part in key.split("/"):
            if part not in cur:
                ok = False
                break
            cur = cur[part]
        if not ok:
            missing.append(key)
    if missing:
        raise AssertionError(
            f"zarr-keys: missing expected keys in {zarr_path}: {missing}; "
            f"present at root: {sorted(list(g.keys()))}"
        )
    print(f"[OK] zarr-keys: {zarr_path.name} contains {list(expected_keys)}")


# zarr-array-range 


def assert_zarr_array_range(
    zarr_path: Path, *, key: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    tol: float = 1e-5,
) -> None:
    g = _open_zarr(zarr_path)
    cur = g
    for part in key.split("/"):
        cur = cur[part]
    arr = np.asarray(cur[:])
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if minimum is not None and arr_min < (minimum - tol):
        raise AssertionError(
            f"zarr-array-range: {key} min={arr_min} < {minimum} (tol={tol})"
        )
    if maximum is not None and arr_max > (maximum + tol):
        raise AssertionError(
            f"zarr-array-range: {key} max={arr_max} > {maximum} (tol={tol})"
        )
    print(
        f"[OK] zarr-array-range: {key} min={arr_min:.4f} max={arr_max:.4f}"
    )


# zarr-array-shape 


def assert_zarr_array_shape(
    zarr_path: Path, *, key: str, expected_shape: Sequence[int],
) -> None:
    g = _open_zarr(zarr_path)
    cur = g
    for part in key.split("/"):
        cur = cur[part]
    if tuple(cur.shape) != tuple(expected_shape):
        raise AssertionError(
            f"zarr-array-shape: {key} shape={cur.shape} != expected "
            f"{tuple(expected_shape)}"
        )
    print(f"[OK] zarr-array-shape: {key} shape={tuple(cur.shape)}")


# instance-count-plausible 


def _load_label_volume(path: Path) -> np.ndarray:
    """Load a label volume from H5 / TIFF / MRC."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for k in ("data", "/data", "labels", "label"):
                if k.lstrip("/") in keys:
                    return np.asarray(f[k.lstrip("/")][:])
            return np.asarray(f[keys[0]][:])
    if suffix in (".tif", ".tiff"):
        import tifffile
        return np.asarray(tifffile.imread(str(path)))
    if suffix == ".mrc":
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as m:
            return np.asarray(m.data)
    raise ValueError(f"unsupported label volume extension: {suffix}")


def assert_instance_count_plausible(
    zarr_path: Path,
    *,
    gt_label_path: Path,
    min_count: int = 2,
    max_multiplier: float = 100.0,
    min_multiplier: float = 0.0,
    gt_mode: str = "cc",
    instance_labels_key: str = "instance_labels",
) -> None:
    """Predicted instance count vs a ground-truth reference count.

    ``gt_mode`` selects how the reference is derived from ``gt_label_path``:

    * ``"cc"`` (default) — connected components of the binarised foreground.
      Right for binary/semantic GT where individual objects aren't labelled.
    * ``"labels"`` — the number of distinct non-zero label values. Right when
      the GT is already an instance label volume (each object a unique id);
      ``"cc"`` would undercount touching objects.

    The predicted count must lie in
    ``[min_multiplier * gt_count, max_multiplier * gt_count]`` (and be at
    least ``min_count``). ``min_multiplier=0`` (default) imposes no lower bound,
    preserving the original behaviour.
    """
    g = _open_zarr(zarr_path)
    inst_arr = np.asarray(g[instance_labels_key][:])
    pred_count = int(np.unique(inst_arr).size) - 1  # subtract bg.
    if pred_count < min_count:
        raise AssertionError(
            f"instance-count: zarr has {pred_count} instances (min "
            f"{min_count}); the assembler likely produced a degenerate "
            f"output."
        )
    gt_vol = _load_label_volume(gt_label_path)
    if gt_mode == "labels":
        gt_count = int((np.unique(gt_vol) != 0).sum())
    elif gt_mode == "cc":
        from scipy import ndimage as ndi
        _, gt_count = ndi.label((gt_vol > 0).astype(np.uint8))
    else:
        raise ValueError(f"gt_mode must be 'cc' or 'labels'; got {gt_mode!r}")
    gt_count = max(int(gt_count), 1)

    upper = int(max_multiplier * gt_count)
    if pred_count > upper:
        raise AssertionError(
            f"instance-count: zarr has {pred_count} instances; gt "
            f"count={gt_count} (mode={gt_mode}); ratio="
            f"{pred_count / gt_count:.1f}x exceeds max_multiplier="
            f"{max_multiplier}."
        )
    if min_multiplier > 0:
        lower = math.floor(min_multiplier * gt_count)
        if pred_count < lower:
            raise AssertionError(
                f"instance-count: zarr has {pred_count} instances; gt "
                f"count={gt_count} (mode={gt_mode}); ratio="
                f"{pred_count / gt_count:.2f}x is below min_multiplier="
                f"{min_multiplier} (need >= {lower}). The assembler is "
                f"merging or dropping instances."
            )
    print(
        f"[OK] instance-count: pred={pred_count}, gt_count={gt_count} "
        f"(mode={gt_mode}), ratio={pred_count / gt_count:.2f}x"
    )


#  instance-detection (IoU / Hungarian matched object detection)


def _instance_detection_scores(pred, gt, iou_threshold):
    """Optimal (Hungarian) IoU matching between predicted and GT instances.

    Returns ``(tp, fp, fn, precision, recall, f1, mean_matched_iou)``. This is
    the honest instance-segmentation metric: counting components is *not* —
    one giant blob + many specks can hit the right component count while
    matching almost no real objects.
    """
    from scipy.optimize import linear_sum_assignment

    pred = pred.ravel().astype(np.int64)
    gt = gt.ravel().astype(np.int64)
    n_pred, n_gt = int(pred.max()), int(gt.max())
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt, 0.0, 0.0, 0.0, 0.0

    # Pairwise intersections via a combined (pred, gt) key.
    key = pred * (n_gt + 1) + gt
    vals, counts = np.unique(key, return_counts=True)
    area_pred = np.bincount(pred, minlength=n_pred + 1)
    area_gt = np.bincount(gt, minlength=n_gt + 1)
    iou = np.zeros((n_pred + 1, n_gt + 1), dtype=np.float64)
    for k, inter in zip(vals, counts):
        a, b = int(k // (n_gt + 1)), int(k % (n_gt + 1))
        if a and b:
            iou[a, b] = inter / (area_pred[a] + area_gt[b] - inter)

    rows, cols = linear_sum_assignment(-iou[1:, 1:])
    matched = [(r + 1, c + 1) for r, c in zip(rows, cols)
               if iou[r + 1, c + 1] >= iou_threshold]
    tp = len(matched)
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean([iou[p, q] for p, q in matched])) if matched else 0.0
    return tp, fp, fn, precision, recall, f1, mean_iou


def assert_instance_detection_f1(
    zarr_path: Path,
    *,
    gt_label_path: Path,
    iou_threshold: float = 0.5,
    min_f1: float = 0.4,
    min_tp: int = 0,
    instance_labels_key: str = "instance_labels",
) -> None:
    """Object-detection F1 between predicted instances and GT instances.

    Each predicted instance is optimally matched to a GT instance by IoU; a
    match with ``IoU >= iou_threshold`` is a true positive. Asserts the F1 is
    at least ``min_f1`` (and optionally at least ``min_tp`` true positives).
    """
    g = _open_zarr(zarr_path)
    pred = np.squeeze(np.asarray(g[instance_labels_key][:]))
    gt = np.squeeze(_load_label_volume(gt_label_path))
    if pred.shape != gt.shape:
        raise AssertionError(
            f"instance-detection: shape mismatch pred {pred.shape} vs gt "
            f"{gt.shape}"
        )
    tp, fp, fn, precision, recall, f1, mean_iou = _instance_detection_scores(
        pred, gt, iou_threshold,
    )
    detail = (
        f"TP={tp} FP={fp} FN={fn} | precision={precision:.3f} recall={recall:.3f} "
        f"F1={f1:.3f} mean_IoU={mean_iou:.3f} (IoU>={iou_threshold}; "
        f"gt={tp + fn}, pred_components={tp + fp})"
    )
    if f1 < min_f1:
        raise AssertionError(
            f"instance-detection: {detail}; F1 below min_f1={min_f1}. The "
            f"assembler is not separating objects (e.g. one merged blob + "
            f"specks counts components but matches no real objects)."
        )
    if tp < min_tp:
        raise AssertionError(
            f"instance-detection: {detail}; TP below min_tp={min_tp}."
        )
    print(f"[OK] instance-detection: {detail}")


#  variance-low-max-prob 


def assert_variance_correlates_with_low_max_prob(
    zarr_path: Path,
    *,
    min_spearman: float = 0.3,
    sample_size: int = 100_000,
) -> None:
    """High variance ↔ low max-prob, tested via Spearman rank corr.

    Sub-samples voxels to keep memory bounded.
    """
    from scipy.stats import spearmanr
    g = _open_zarr(zarr_path)
    var = np.asarray(g["tta_variance_map"][:]).ravel().astype(np.float64)
    if "teacher_probs" not in g:
        raise AssertionError(
            "variance-low-max-prob: teacher_probs missing from zarr; cannot "
            "compute max-prob."
        )
    probs = np.asarray(g["teacher_probs"][:]).astype(np.float32)
    max_prob = probs.max(axis=0).ravel().astype(np.float64)
    n = min(sample_size, var.size)
    rng = np.random.default_rng(0)
    idx = rng.choice(var.size, size=n, replace=False) if var.size > n else np.arange(var.size)
    rho, _p = spearmanr(var[idx], -max_prob[idx])  # high var ↔ low max-prob.
    if math.isnan(rho):
        raise AssertionError(
            "variance-low-max-prob: Spearman correlation is NaN — typical "
            "cause is a constant variance or constant max-prob array."
        )
    if rho < min_spearman:
        raise AssertionError(
            f"variance-low-max-prob: Spearman rho={rho:.3f} < {min_spearman}; "
            f"variance is not tracking uncertainty as expected."
        )
    print(
        f"[OK] variance-low-max-prob: Spearman rho={rho:.3f} >= {min_spearman}"
    )


# loss-decreasing 


def assert_loss_decreasing(
    metrics_csv: Path, *, column: str, max_increase_ratio: float = 1.05,
) -> None:
    """Per-head loss monotonically (or near-monotonically) decreasing.

    Real training has noise, so we tolerate transient upticks up to
    ``max_increase_ratio`` times the running min.
    """
    import csv
    with metrics_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(
            f"loss-decreasing: {metrics_csv} has no rows."
        )
    if column not in rows[0]:
        raise AssertionError(
            f"loss-decreasing: column {column!r} not in CSV header "
            f"{list(rows[0].keys())}"
        )
    values: List[float] = []
    for r in rows:
        v = r[column]
        if v in (None, "", "nan"):
            continue
        values.append(float(v))
    if len(values) < 2:
        raise AssertionError(
            f"loss-decreasing: {column} has fewer than 2 valid rows; cannot "
            f"check trend."
        )
    if values[-1] >= values[0]:
        raise AssertionError(
            f"loss-decreasing: {column} did not improve overall: "
            f"start={values[0]:.4f} end={values[-1]:.4f}"
        )
    running_min = values[0]
    for i, v in enumerate(values[1:], start=1):
        running_min = min(running_min, v)
        if v > running_min * max_increase_ratio:
            raise AssertionError(
                f"loss-decreasing: {column} step {i}={v:.4f} exceeds "
                f"running min {running_min:.4f} by ratio "
                f"{v / running_min:.2f}x > {max_increase_ratio}."
            )
    print(
        f"[OK] loss-decreasing: {column} {values[0]:.4f} -> {values[-1]:.4f} "
        f"(min {running_min:.4f})"
    )


# val-dice-within 


def assert_val_dice_within(
    ckpt_a: Path, ckpt_b: Path, *, max_drift: float = 0.03,
) -> None:
    """Read ``best_val_loss`` / ``best_val_dice`` / ``loss_val`` from two
    checkpoints and assert |a - b| <= max_drift.

    ``loss_val`` is the key the pipeline version trainer (and the legacy raw trainer)
    actually persists; ``best_val_dice`` and ``best_val_loss`` are
    forward-compatible candidates for future trainer revisions.
    """
    a = _load_torch_checkpoint(ckpt_a)
    b = _load_torch_checkpoint(ckpt_b)
    candidates = ("best_val_dice", "best_val_loss", "loss_val")
    val_a = val_b = None
    for k in candidates:
        if k in a and val_a is None:
            val_a = float(a[k])
        if k in b and val_b is None:
            val_b = float(b[k])
    if val_a is None or val_b is None:
        raise AssertionError(
            f"val-dice-within: neither checkpoint exposes any of {candidates}; "
            f"a-keys={list(a.keys())[:8]} b-keys={list(b.keys())[:8]}"
        )
    drift = abs(val_a - val_b)
    if drift > max_drift:
        raise AssertionError(
            f"val-dice-within: drift={drift:.4f} > {max_drift} "
            f"(a={val_a:.4f}, b={val_b:.4f})"
        )
    print(
        f"[OK] val-dice-within: a={val_a:.4f}, b={val_b:.4f}, drift={drift:.4f}"
    )


#  CLI 


# segmentation-dice


def _resolve_pred_path(pred_arg: str) -> Path:
    """Resolve a predicted-volume path that may be a glob (newest match wins)."""
    if any(ch in pred_arg for ch in "*?["):
        matches = sorted(glob.glob(pred_arg), key=lambda p: Path(p).stat().st_mtime)
        if not matches:
            raise AssertionError(f"segmentation-dice: no file matches {pred_arg!r}")
        return Path(matches[-1])
    return Path(pred_arg)


def assert_segmentation_dice(
    pred_path: str,
    *,
    gt_label_path: Path,
    min_dice: float = 0.7,
    min_fg_fraction: Optional[float] = None,
    max_fg_fraction: Optional[float] = None,
    pred_key: Optional[str] = None,
) -> None:
    """Foreground/background Dice between a predicted volume and the GT mask.

    Catches the "edge-detector" failure mode where the semantic head predicts a
    thin rim of foreground (low Dice, tiny foreground fraction) instead of solid
    cell bodies. ``pred_path`` may be a glob; the newest match is used. When
    ``pred_key`` is given, ``pred_path`` is treated as a zarr store and the named
    array (e.g. ``teacher_argmax``) is read -- this lets the same assertion check
    the pipeline's semantic output, not just a legacy ``*_vol_pred`` tiff.
    """
    if pred_key is not None:
        g = _open_zarr(_resolve_pred_path(pred_path))
        pv = np.squeeze(np.asarray(g[pred_key][:]))
        pred = Path(pred_path)
    else:
        pred = _resolve_pred_path(pred_path)
        pv = np.squeeze(_load_label_volume(pred))
    gv = np.squeeze(_load_label_volume(gt_label_path))
    if pv.shape != gv.shape:
        raise AssertionError(
            f"segmentation-dice: shape mismatch pred {pv.shape} vs gt {gv.shape}"
        )
    p_fg = pv > 0
    g_fg = gv > 0
    intersection = int(np.logical_and(p_fg, g_fg).sum())
    denom = int(p_fg.sum() + g_fg.sum())
    dice = (2.0 * intersection / denom) if denom > 0 else 1.0
    pred_frac = float(p_fg.mean())
    gt_frac = float(g_fg.mean())

    detail = (
        f"dice={dice:.3f}, pred_fg={100 * pred_frac:.1f}%, gt_fg={100 * gt_frac:.1f}% "
        f"(file={pred.name})"
    )
    if dice < min_dice:
        raise AssertionError(
            f"segmentation-dice: {detail}; below min_dice={min_dice}. The model is "
            f"not producing a solid foreground segmentation."
        )
    if min_fg_fraction is not None and pred_frac < min_fg_fraction:
        raise AssertionError(
            f"segmentation-dice: {detail}; predicted foreground fraction below "
            f"min_fg_fraction={min_fg_fraction} (edge-detector pathology)."
        )
    if max_fg_fraction is not None and pred_frac > max_fg_fraction:
        raise AssertionError(
            f"segmentation-dice: {detail}; predicted foreground fraction above "
            f"max_fg_fraction={max_fg_fraction} (over-segmenting background)."
        )
    print(f"[OK] segmentation-dice: {detail} >= min_dice={min_dice}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pipeline version smoke-gate assertions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("checkpoint-trained")
    sp.add_argument("ckpt", type=Path)
    sp.add_argument("--min-nbts", type=int, default=1)
    sp.add_argument("--min-rvar-std", type=float, default=0.01)

    sp = sub.add_parser("zarr-keys")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--keys", nargs="+", required=True)

    sp = sub.add_parser("zarr-array-range")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--key", required=True)
    sp.add_argument("--min", type=float, default=None, dest="minimum")
    sp.add_argument("--max", type=float, default=None, dest="maximum")
    sp.add_argument("--tol", type=float, default=1e-5)

    sp = sub.add_parser("zarr-array-shape")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--key", required=True)
    sp.add_argument("--shape", type=int, nargs="+", required=True)

    sp = sub.add_parser("instance-count-plausible")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--gt-label", type=Path, required=True)
    sp.add_argument("--min-count", type=int, default=2)
    sp.add_argument("--max-multiplier", type=float, default=100.0)
    sp.add_argument("--min-multiplier", type=float, default=0.0)
    sp.add_argument("--gt-mode", choices=("cc", "labels"), default="cc")
    sp.add_argument("--instance-key", default="instance_labels")

    sp = sub.add_parser("instance-detection")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--gt-label", type=Path, required=True)
    sp.add_argument("--iou-threshold", type=float, default=0.5)
    sp.add_argument("--min-f1", type=float, default=0.4)
    sp.add_argument("--min-tp", type=int, default=0)
    sp.add_argument("--instance-key", default="instance_labels")

    sp = sub.add_parser("variance-low-max-prob")
    sp.add_argument("zarr", type=Path)
    sp.add_argument("--min-spearman", type=float, default=0.3)
    sp.add_argument("--sample-size", type=int, default=100_000)

    sp = sub.add_parser("loss-decreasing")
    sp.add_argument("metrics_csv", type=Path)
    sp.add_argument("--column", required=True)
    sp.add_argument("--max-increase-ratio", type=float, default=1.05)

    sp = sub.add_parser("val-dice-within")
    sp.add_argument("ckpt_a", type=Path)
    sp.add_argument("ckpt_b", type=Path)
    sp.add_argument("--max-drift", type=float, default=0.03)

    sp = sub.add_parser("segmentation-dice")
    sp.add_argument("pred", help="Predicted volume path (may be a glob; newest wins).")
    sp.add_argument("--gt-label", type=Path, required=True)
    sp.add_argument("--min-dice", type=float, default=0.7)
    sp.add_argument("--min-fg-fraction", type=float, default=None)
    sp.add_argument("--max-fg-fraction", type=float, default=None)
    sp.add_argument(
        "--pred-key",
        default=None,
        help="If set, treat 'pred' as a zarr store and read this array (e.g. teacher_argmax).",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "checkpoint-trained":
        assert_checkpoint_trained(
            args.ckpt,
            min_num_batches_tracked=args.min_nbts,
            min_running_var_std=args.min_rvar_std,
        )
    elif args.cmd == "zarr-keys":
        assert_zarr_keys(args.zarr, args.keys)
    elif args.cmd == "zarr-array-range":
        assert_zarr_array_range(
            args.zarr,
            key=args.key,
            minimum=args.minimum,
            maximum=args.maximum,
            tol=args.tol,
        )
    elif args.cmd == "zarr-array-shape":
        assert_zarr_array_shape(
            args.zarr, key=args.key, expected_shape=args.shape,
        )
    elif args.cmd == "instance-count-plausible":
        assert_instance_count_plausible(
            args.zarr,
            gt_label_path=args.gt_label,
            min_count=args.min_count,
            max_multiplier=args.max_multiplier,
            min_multiplier=args.min_multiplier,
            gt_mode=args.gt_mode,
            instance_labels_key=args.instance_key,
        )
    elif args.cmd == "instance-detection":
        assert_instance_detection_f1(
            args.zarr,
            gt_label_path=args.gt_label,
            iou_threshold=args.iou_threshold,
            min_f1=args.min_f1,
            min_tp=args.min_tp,
            instance_labels_key=args.instance_key,
        )
    elif args.cmd == "variance-low-max-prob":
        assert_variance_correlates_with_low_max_prob(
            args.zarr,
            min_spearman=args.min_spearman,
            sample_size=args.sample_size,
        )
    elif args.cmd == "loss-decreasing":
        assert_loss_decreasing(
            args.metrics_csv,
            column=args.column,
            max_increase_ratio=args.max_increase_ratio,
        )
    elif args.cmd == "val-dice-within":
        assert_val_dice_within(
            args.ckpt_a, args.ckpt_b, max_drift=args.max_drift,
        )
    elif args.cmd == "segmentation-dice":
        assert_segmentation_dice(
            args.pred,
            gt_label_path=args.gt_label,
            min_dice=args.min_dice,
            min_fg_fraction=args.min_fg_fraction,
            max_fg_fraction=args.max_fg_fraction,
            pred_key=args.pred_key,
        )
    else:
        raise SystemExit(f"unknown subcommand {args.cmd!r}")


if __name__ == "__main__":
    main()
