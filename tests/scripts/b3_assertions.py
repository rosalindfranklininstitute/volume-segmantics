#!/usr/bin/env python
"""smoke-gate assertion helpers.

"""

from __future__ import annotations

import argparse
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
    instance_labels_key: str = "instance_labels",
) -> None:
    """Predicted instance count vs the GT connected-component count.

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
    # Compare against GT cc count.
    gt_vol = _load_label_volume(gt_label_path)
    gt_fg = (gt_vol > 0).astype(np.uint8)
    from scipy import ndimage as ndi
    _, gt_cc = ndi.label(gt_fg)
    if gt_cc < 1:
        gt_cc = 1
    upper = int(max_multiplier * gt_cc)
    if pred_count > upper:
        raise AssertionError(
            f"instance-count: zarr has {pred_count} instances; gt cc "
            f"count={gt_cc}; ratio={pred_count / gt_cc:.1f}x exceeds "
            f"max_multiplier={max_multiplier}. Likely 53k-noise pathology "
            f"(v0.5 dev-history risk)."
        )
    print(
        f"[OK] instance-count: pred={pred_count}, gt_cc={gt_cc}, "
        f"ratio={pred_count / max(gt_cc, 1):.2f}x"
    )


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

    ``loss_val`` is the key the b3 trainer (and the legacy raw trainer)
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="b3 smoke-gate assertions.")
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
    else:
        raise SystemExit(f"unknown subcommand {args.cmd!r}")


if __name__ == "__main__":
    main()
