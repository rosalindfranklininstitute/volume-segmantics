import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from volume_segmantics.model.operations.trainer_metrics import (
    MetricsCalculator,
    ensure_tuple_output,
    get_eval_metric,
)


# Fixtures


@pytest.fixture
def pred_target_2x8x8():
    """Pred and target class indices (B=2, H=W=8) for dice tests."""
    torch.manual_seed(42)
    pred = torch.randint(0, 4, (2, 8, 8))
    target = torch.randint(0, 4, (2, 8, 8))
    return pred, target


@pytest.fixture
def seg_logits_and_onehot():
    """Seg logits (B, C, H, W) and one-hot targets (B, C, H, W) for compute_eval_metrics."""
    torch.manual_seed(43)
    B, C, H, W = 2, 4, 8, 8
    logits = torch.randn(B, C, H, W)
    # One-hot: argmax over random probs
    probs = F.softmax(torch.randn(B, C, H, W), dim=1)
    onehot = F.one_hot(probs.argmax(dim=1), num_classes=C).permute(0, 3, 1, 2).float()
    return logits, onehot


@pytest.fixture
def settings_weighted():
    """Settings with dice_weight_mode for weighted averaging."""
    return SimpleNamespace(dice_weight_mode="inverse_sqrt_freq")


#  ensure_tuple_output


def test_ensure_tuple_output_single():
    """Pass a single tensor; returns tuple of length 1."""
    x = torch.rand(2, 3)
    out = ensure_tuple_output(x)
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0] is x


def test_ensure_tuple_output_list():
    """Pass list of tensors; returns tuple with same elements."""
    a, b = torch.rand(1), torch.rand(2)
    out = ensure_tuple_output([a, b])
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] is a and out[1] is b


def test_ensure_tuple_output_tuple():
    """Pass tuple; returns same tuple (or equal content)."""
    a, b = torch.rand(1), torch.rand(2)
    t = (a, b)
    out = ensure_tuple_output(t)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] is a and out[1] is b


def test_ensure_tuple_output_none_or_int():
    """Pass non-tensor (e.g. int or None); returns single-element tuple."""
    out_none = ensure_tuple_output(None)
    assert out_none == (None,)
    out_int = ensure_tuple_output(3)
    assert out_int == (3,)


# -MetricsCalculator ? compute_multiclass_dice 


def test_metrics_calculator_init():
    """Create MetricsCalculator(num_classes=4); assert attributes."""
    calc = MetricsCalculator(num_classes=4, exclude_background=True, dice_averaging="macro")
    assert calc.num_classes == 4
    assert calc.exclude_background is True
    assert calc.dice_averaging == "macro"


def test_compute_multiclass_dice_returns_tuple(pred_target_2x8x8):
    """Returns (dice_per_class, mean_dice); dice_per_class length == num_classes."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert isinstance(dice_per_class, list)
    assert len(dice_per_class) == 4
    assert isinstance(mean_dice, float)


def test_compute_multiclass_dice_perfect_match():
    """pred == target -> mean_dice 1.0, per-class dice 1.0 for present classes."""
    torch.manual_seed(44)
    pred = torch.randint(0, 3, (2, 8, 8))
    target = pred.clone()
    calc = MetricsCalculator(num_classes=3, exclude_background=False)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert mean_dice == pytest.approx(1.0, abs=1e-5)
    for d in dice_per_class:
        if not math.isnan(d):
            assert d == pytest.approx(1.0, abs=1e-5)


def test_compute_multiclass_dice_no_match():
    """Pred and target disjoint classes where possible -> mean_dice 0 or low; no crash."""
    # All pred=0, all target=1
    pred = torch.zeros(2, 8, 8, dtype=torch.long)
    target = torch.ones(2, 8, 8, dtype=torch.long)
    calc = MetricsCalculator(num_classes=2, exclude_background=False)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert mean_dice == pytest.approx(0.0, abs=1e-5)
    assert dice_per_class[0] == pytest.approx(0.0, abs=1e-5)
    assert dice_per_class[1] == pytest.approx(0.0, abs=1e-5)


def test_compute_multiclass_dice_nan_absent_class():
    """When a class has no pixels in pred or target, that class gets NaN; mean_dice averages only valid."""
    # Only class 0 and 1 present
    pred = torch.zeros(2, 8, 8, dtype=torch.long)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    pred[0, 0:4, :] = 1
    target[0, 0:4, :] = 1
    calc = MetricsCalculator(num_classes=3, exclude_background=False)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert math.isnan(dice_per_class[2])
    valid = [d for d in dice_per_class if not math.isnan(d)]
    assert len(valid) == 2
    assert mean_dice == pytest.approx(sum(valid) / 2, abs=1e-5)


def test_compute_multiclass_dice_exclude_background():
    """exclude_background=True -> mean_dice computed over classes 1..num_classes-1; class 0 excluded from mean."""
    pred = torch.ones(2, 8, 8, dtype=torch.long)  # all class 1
    target = torch.ones(2, 8, 8, dtype=torch.long)
    calc = MetricsCalculator(num_classes=2, exclude_background=True)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert mean_dice == pytest.approx(1.0, abs=1e-5)
    # Class 0 may be NaN (no foreground in "background"); class 1 is 1.0
    assert dice_per_class[1] == pytest.approx(1.0, abs=1e-5)


def test_compute_multiclass_dice_override_params():
    """Pass num_classes=3, exclude_background=False explicitly; result uses those."""
    pred, target = torch.randint(0, 3, (2, 8, 8)), torch.randint(0, 3, (2, 8, 8))
    calc = MetricsCalculator(num_classes=4, exclude_background=True)
    dice_per_class, _ = calc.compute_multiclass_dice(
        pred, target, num_classes=3, exclude_background=False
    )
    assert len(dice_per_class) == 3


def test_compute_multiclass_dice_value_range(pred_target_2x8x8):
    """All valid (non-NaN) dice in [0, 1]; mean_dice in [0, 1]."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    for d in dice_per_class:
        if not math.isnan(d):
            assert 0 <= d <= 1
    assert 0 <= mean_dice <= 1


# MetricsCalculator ? compute_weighted_multiclass_dice 


def test_compute_weighted_returns_three(pred_target_2x8x8):
    """Returns (dice_per_class, weighted_mean_dice, weights_used); lengths match num_classes."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    dice_per_class, weighted_mean_dice, weights_used = calc.compute_weighted_multiclass_dice(
        pred, target
    )
    assert len(dice_per_class) == 4
    assert len(weights_used) == 4
    assert isinstance(weighted_mean_dice, float)


def test_compute_weighted_uniform():
    """weight_mode='uniform' with mixed class counts; weights sum to 1; weighted_mean_dice in [0, 1]."""
    torch.manual_seed(45)
    pred = torch.randint(0, 3, (2, 8, 8))
    target = pred.clone()
    calc = MetricsCalculator(num_classes=3, exclude_background=False)
    _, weighted_mean_dice, weights = calc.compute_weighted_multiclass_dice(
        pred, target, weight_mode="uniform"
    )
    assert abs(sum(weights) - 1.0) < 1e-5 or sum(weights) == 0
    assert 0 <= weighted_mean_dice <= 1


def test_compute_weighted_inverse_freq(pred_target_2x8x8):
    """weight_mode='inverse_freq' runs; result scalar and in [0, 1]."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    _, weighted_mean_dice, _ = calc.compute_weighted_multiclass_dice(
        pred, target, weight_mode="inverse_freq"
    )
    assert isinstance(weighted_mean_dice, float)
    assert 0 <= weighted_mean_dice <= 1


def test_compute_weighted_inverse_sqrt_freq(pred_target_2x8x8):
    """weight_mode='inverse_sqrt_freq' runs; same sanity checks."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    _, weighted_mean_dice, _ = calc.compute_weighted_multiclass_dice(
        pred, target, weight_mode="inverse_sqrt_freq"
    )
    assert isinstance(weighted_mean_dice, float)
    assert 0 <= weighted_mean_dice <= 1


def test_compute_weighted_pixel_count(pred_target_2x8x8):
    """weight_mode='pixel_count' runs; weighted_mean_dice in valid range."""
    pred, target = pred_target_2x8x8
    calc = MetricsCalculator(num_classes=4)
    _, weighted_mean_dice, weights = calc.compute_weighted_multiclass_dice(
        pred, target, weight_mode="pixel_count"
    )
    assert 0 <= weighted_mean_dice <= 1
    assert len(weights) == 4


def test_compute_weighted_exclude_background():
    """exclude_background=True; class 0 weight 0; weighted mean only over foreground."""
    pred = torch.ones(2, 8, 8, dtype=torch.long)
    target = torch.ones(2, 8, 8, dtype=torch.long)
    calc = MetricsCalculator(num_classes=2, exclude_background=True)
    _, weighted_mean_dice, weights = calc.compute_weighted_multiclass_dice(
        pred, target, weight_mode="uniform"
    )
    assert weights[0] == 0.0
    assert weighted_mean_dice == pytest.approx(1.0, abs=1e-5)


# MetricsCalculator ? compute_eval_metrics 

def test_compute_eval_metrics_seg_only(seg_logits_and_onehot):
    """outputs=(seg_logits,), targets one-hot; returns dict with seg_dice, dice_class_*; seg_dice in [0, 1]."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics((logits,), onehot)
    assert "seg_dice" in metrics
    assert 0 <= metrics["seg_dice"] <= 1
    for c in range(4):
        assert f"dice_class_{c}" in metrics
        assert 0 <= metrics[f"dice_class_{c}"] <= 1


def test_compute_eval_metrics_target_tensor(seg_logits_and_onehot):
    """Targets as single tensor (one-hot seg); seg_dice present."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics((logits,), onehot)
    assert "seg_dice" in metrics


def test_compute_eval_metrics_target_dict(seg_logits_and_onehot):
    """Targets = {'seg': tensor}; same seg_dice behaviour."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics((logits,), {"seg": onehot})
    assert "seg_dice" in metrics
    assert 0 <= metrics["seg_dice"] <= 1


def test_compute_eval_metrics_macro(seg_logits_and_onehot):
    """dice_averaging='macro'; compute_eval_metrics runs; seg_dice consistent with compute_multiclass_dice."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(num_classes=4, dice_averaging="macro")
    metrics = calc.compute_eval_metrics((logits,), onehot)
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    gt = torch.argmax(onehot, dim=1)
    _, expected_mean = calc.compute_multiclass_dice(pred, gt)
    assert metrics["seg_dice"] == pytest.approx(expected_mean, abs=1e-5)


def test_compute_eval_metrics_weighted(seg_logits_and_onehot, settings_weighted):
    """dice_averaging='weighted' and settings with dice_weight_mode; runs and returns seg_dice."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(
        num_classes=4, dice_averaging="weighted", settings=settings_weighted
    )
    metrics = calc.compute_eval_metrics((logits,), onehot)
    assert "seg_dice" in metrics
    assert 0 <= metrics["seg_dice"] <= 1


def test_compute_eval_metrics_boundary(seg_logits_and_onehot):
    """outputs=(seg_logits, boundary_logits), targets with 'boundary'; returns boundary_dice, boundary_dice_prob; stats non-empty."""
    logits, onehot = seg_logits_and_onehot
    B, _, H, W = logits.shape
    boundary_logits = torch.randn(B, 2, H, W)
    boundary_target = (torch.rand(B, 2, H, W) > 0.7).float()
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics(
        (logits, boundary_logits),
        {"seg": onehot, "boundary": boundary_target},
    )
    assert "boundary_dice" in metrics
    assert "boundary_dice_prob" in metrics
    assert 0 <= metrics["boundary_dice"] <= 1
    assert 0 <= metrics["boundary_dice_prob"] <= 1
    assert len(calc.get_boundary_stats()) == 1


def test_compute_eval_metrics_boundary_channel_mismatch(seg_logits_and_onehot):
    """boundary_output has more channels than boundary_target; code clips to target channels; no crash."""
    logits, onehot = seg_logits_and_onehot
    B, _, H, W = logits.shape
    boundary_logits = torch.randn(B, 5, H, W)  # 5 channels
    boundary_target = torch.rand(B, 2, H, W)  # 2 channels
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics(
        (logits, boundary_logits),
        {"seg": onehot, "boundary": boundary_target},
    )
    assert "boundary_dice" in metrics


def test_compute_eval_metrics_boundary_sparse(seg_logits_and_onehot):
    """Boundary target with very low positive ratio (< 0.1); adaptive threshold path runs; boundary_dice in [0, 1]."""
    logits, onehot = seg_logits_and_onehot
    B, _, H, W = logits.shape
    boundary_logits = torch.randn(B, 1, H, W)
    boundary_target = torch.zeros(B, 1, H, W)
    boundary_target[:, :, 0, 0] = 1.0  # single positive per batch
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics(
        (logits, boundary_logits),
        {"seg": onehot, "boundary": boundary_target},
    )
    assert "boundary_dice" in metrics
    assert 0 <= metrics["boundary_dice"] <= 1


# MetricsCalculator ? boundary stats 


def test_get_boundary_stats_empty():
    """New MetricsCalculator; get_boundary_stats() returns []."""
    calc = MetricsCalculator(num_classes=2)
    assert calc.get_boundary_stats() == []


def test_clear_boundary_stats(seg_logits_and_onehot):
    """After compute_eval_metrics with boundary, get_boundary_stats() has entries; clear_boundary_stats(); empty."""
    logits, onehot = seg_logits_and_onehot
    B, _, H, W = logits.shape
    boundary_logits = torch.randn(B, 1, H, W)
    boundary_target = torch.rand(B, 1, H, W)
    calc = MetricsCalculator(num_classes=4)
    calc.compute_eval_metrics(
        (logits, boundary_logits),
        {"seg": onehot, "boundary": boundary_target},
    )
    assert len(calc.get_boundary_stats()) >= 1
    calc.clear_boundary_stats()
    assert calc.get_boundary_stats() == []


# --- 7. get_eval_metric ---


def test_get_eval_metric_mean_iou():
    """get_eval_metric('MeanIoU') returns callable metric instance."""
    metric = get_eval_metric("MeanIoU")
    assert metric is not None
    assert callable(metric)


def test_get_eval_metric_dice_coefficient():
    """get_eval_metric('DiceCoefficient') returns callable metric instance."""
    metric = get_eval_metric("DiceCoefficient")
    assert metric is not None
    assert callable(metric)


def test_get_eval_metric_unknown_exits():
    """get_eval_metric('UnknownMetric') raises SystemExit."""
    with pytest.raises(SystemExit):
        get_eval_metric("UnknownMetric")


# Edge cases

def test_compute_multiclass_dice_all_same_class():
    """pred/target all same class (e.g. all 0); that class dice 1.0; others NaN."""
    pred = torch.zeros(2, 8, 8, dtype=torch.long)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    calc = MetricsCalculator(num_classes=2, exclude_background=False)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert dice_per_class[0] == pytest.approx(1.0, abs=1e-5)
    assert math.isnan(dice_per_class[1])
    assert mean_dice == pytest.approx(1.0, abs=1e-5)


def test_compute_multiclass_dice_single_pixel():
    """Single pixel (1, 1, 1); num_classes=2; no crash; mean_dice defined."""
    pred = torch.zeros(1, 1, 1, dtype=torch.long)
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    calc = MetricsCalculator(num_classes=2, exclude_background=False)
    dice_per_class, mean_dice = calc.compute_multiclass_dice(pred, target)
    assert len(dice_per_class) == 2
    assert not math.isnan(mean_dice)
    assert 0 <= mean_dice <= 1


def test_compute_eval_metrics_dict_seg_only_no_boundary(seg_logits_and_onehot):
    """Single output, targets dict with only 'seg'; no boundary key; boundary branch not taken."""
    logits, onehot = seg_logits_and_onehot
    calc = MetricsCalculator(num_classes=4)
    metrics = calc.compute_eval_metrics((logits,), {"seg": onehot})
    assert "seg_dice" in metrics
    assert "boundary_dice" not in metrics
    assert calc.get_boundary_stats() == []
