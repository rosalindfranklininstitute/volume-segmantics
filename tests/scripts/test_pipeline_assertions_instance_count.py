"""The ``instance-count-plausible`` gate bounds the count both ways by default.

Its defaults used to be ``max_multiplier=100`` with no lower bound, and the
one config using it passed ``--max-multiplier 6000``, so predictions with 5
to 105 times the ground-truth count passed as if checked. A smoke config that
cannot judge counts now says ``--structural-only`` and the log says so.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.scripts.pipeline_assertions import assert_instance_count_plausible


def _gt(path: Path, n_objects: int) -> Path:
    """A label volume whose foreground is ``n_objects`` separated cubes."""
    import tifffile
    vol = np.zeros((4, 4, 8 * n_objects), dtype=np.uint8)
    for i in range(n_objects):
        vol[1:3, 1:3, 8 * i + 2: 8 * i + 4] = 255
    tifffile.imwrite(str(path), vol)
    return path


def _pred(path: Path, n_instances: int, shape=(4, 4, 80)) -> Path:
    inst = np.zeros(shape, dtype=np.uint32)
    flat = inst.reshape(-1)
    flat[:n_instances] = np.arange(1, n_instances + 1)
    g = zarr.open(str(path), mode="w")
    g.create_array("instance_labels", data=inst)
    return path


def test_five_times_the_ground_truth_fails_by_default(tmp_path):
    gt = _gt(tmp_path / "gt.tif", 10)
    with pytest.raises(AssertionError, match=r"ratio=5\.0x exceeds max_multiplier=3\.0"):
        assert_instance_count_plausible(_pred(tmp_path / "p.zarr", 50), gt_label_path=gt)


def test_a_fifth_of_the_ground_truth_fails_by_default(tmp_path):
    gt = _gt(tmp_path / "gt.tif", 10)
    with pytest.raises(AssertionError, match=r"below min_multiplier=0\.33"):
        assert_instance_count_plausible(_pred(tmp_path / "p.zarr", 2), gt_label_path=gt)


def test_within_a_factor_of_three_passes(tmp_path):
    gt = _gt(tmp_path / "gt.tif", 10)
    for n in (4, 10, 30):
        assert_instance_count_plausible(_pred(tmp_path / f"p{n}.zarr", n), gt_label_path=gt)


def test_structural_only_checks_min_count_and_says_so(tmp_path, capsys):
    gt = _gt(tmp_path / "gt.tif", 10)
    assert_instance_count_plausible(
        _pred(tmp_path / "p.zarr", 50), gt_label_path=gt, structural_only=True,
    )
    assert "structural only" in capsys.readouterr().out
    with pytest.raises(AssertionError, match="degenerate"):
        assert_instance_count_plausible(
            _pred(tmp_path / "q.zarr", 1), gt_label_path=gt, structural_only=True,
        )


def test_an_inverted_band_is_refused(tmp_path):
    gt = _gt(tmp_path / "gt.tif", 10)
    with pytest.raises(ValueError, match="min_multiplier"):
        assert_instance_count_plausible(
            _pred(tmp_path / "p.zarr", 10), gt_label_path=gt, min_multiplier=4, max_multiplier=2,
        )
