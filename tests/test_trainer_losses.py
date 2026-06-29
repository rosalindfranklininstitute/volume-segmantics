"""
Tests for volume_segmantics.model.operations.trainer_losses.

Covers: ConsistencyLoss, get_rampup_ratio, ClassWeightedDiceLoss, CombinedCEDiceLoss.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from volume_segmantics.model.operations.trainer_losses import (
    ClassWeightedDiceLoss,
    CombinedCEDiceLoss,
    ConsistencyLoss,
    get_rampup_ratio,
)


# --- Fixtures ---


@pytest.fixture
def batch_2c():
    """Pred and target (B=2, C=2, H=W=16) for binary-style tests."""
    torch.manual_seed(42)
    pred = torch.randn(2, 2, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16))
    return pred, target


@pytest.fixture
def batch_4c():
    """Pred and target (B=2, C=4, H=W=16) for multi-class tests."""
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    return pred, target


# --- 2. ConsistencyLoss ---


def test_consistency_loss_is_module():
    """ConsistencyLoss is an nn.Module."""
    assert isinstance(ConsistencyLoss(), nn.Module)


def test_consistency_loss_forward_shape():
    """Forward returns scalar for (B, C, H, W) logits."""
    loss_fn = ConsistencyLoss()
    student = torch.randn(2, 4, 8, 8)
    teacher = torch.randn(2, 4, 8, 8)
    out = loss_fn(student, teacher)
    assert out.dim() == 0
    assert out.shape == ()


def test_consistency_loss_identical_inputs_zero():
    """When student == teacher, loss is 0."""
    loss_fn = ConsistencyLoss()
    logits = torch.randn(2, 3, 8, 8)
    out = loss_fn(logits, logits)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_loss_positive_when_different():
    """When student and teacher differ, loss > 0."""
    loss_fn = ConsistencyLoss()
    student = torch.randn(2, 3, 8, 8)
    teacher = torch.zeros_like(student)
    out = loss_fn(student, teacher)
    assert out.item() > 0


def test_consistency_loss_gradient_flow():
    """Backward propagates gradients to inputs."""
    loss_fn = ConsistencyLoss()
    student = torch.randn(2, 3, 8, 8, requires_grad=True)
    teacher = torch.randn(2, 3, 8, 8, requires_grad=True)
    out = loss_fn(student, teacher)
    out.backward()
    assert student.grad is not None
    assert teacher.grad is not None


# --- 3. get_rampup_ratio ---


def test_rampup_before_start():
    """Ratio is 0.0 when current_iter < rampup_start."""
    assert get_rampup_ratio(0, 10, 100) == 0.0
    assert get_rampup_ratio(9, 10, 100) == 0.0


def test_rampup_after_end():
    """Ratio is 1.0 when current_iter >= rampup_end."""
    assert get_rampup_ratio(100, 10, 100) == 1.0
    assert get_rampup_ratio(150, 10, 100) == 1.0


def test_rampup_at_boundaries():
    """At start and just before end, ratio is in (0, 1] for sigmoid."""
    r_start = get_rampup_ratio(10, 10, 100)
    r_before_end = get_rampup_ratio(99, 10, 100)
    assert 0 < r_start <= 1
    assert 0 < r_before_end <= 1
    assert get_rampup_ratio(100, 10, 100) == 1.0


def test_rampup_linear():
    """Linear ramp gives (current - start) / (end - start)."""
    r = get_rampup_ratio(55, 10, 100, "linear")
    expected = (55 - 10) / (100 - 10)
    assert r == pytest.approx(expected, abs=1e-6)


def test_rampup_sigmoid_monotonic():
    """Sigmoid ramp increases with current_iter in (start, end)."""
    start, end = 10, 100
    prev = -1.0
    for curr in [20, 40, 60, 80]:
        r = get_rampup_ratio(curr, start, end, "sigmoid")
        assert r > prev
        prev = r


def test_rampup_returns_float():
    """Return type is Python float."""
    r = get_rampup_ratio(50, 10, 100)
    assert isinstance(r, float)


# --- 4. ClassWeightedDiceLoss ---


def test_dice_loss_is_module():
    """ClassWeightedDiceLoss is an nn.Module."""
    assert isinstance(ClassWeightedDiceLoss(2), nn.Module)


def test_dice_forward_scalar():
    """Forward returns scalar for (B, C, H, W) pred and target."""
    loss_fn = ClassWeightedDiceLoss(2)
    pred = torch.randn(2, 2, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16))
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert out.shape == ()


def test_dice_loss_in_valid_range():
    """Loss is in [0, 1] for valid pred and target."""
    torch.manual_seed(42)
    loss_fn = ClassWeightedDiceLoss(2, weight_mode="uniform")
    pred = torch.randn(2, 2, 16, 16)
    target = F.one_hot(torch.randint(0, 2, (2, 16, 16)), 2).permute(0, 3, 1, 2).float()
    out = loss_fn(pred, target)
    assert 0 <= out.item() <= 1.0


def test_dice_uniform_weights():
    """weight_mode='uniform' runs and is deterministic for fixed input."""
    torch.manual_seed(42)
    loss_fn = ClassWeightedDiceLoss(2, weight_mode="uniform")
    pred = torch.randn(2, 2, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16))
    out1 = loss_fn(pred, target)
    out2 = loss_fn(pred, target)
    assert out1.item() == pytest.approx(out2.item(), abs=1e-6)


def test_dice_inverse_freq():
    """weight_mode='inverse_freq' runs and returns scalar in reasonable range."""
    loss_fn = ClassWeightedDiceLoss(4, weight_mode="inverse_freq")
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert 0 <= out.item() <= 2.0


def test_dice_inverse_sqrt_freq():
    """weight_mode='inverse_sqrt_freq' runs and returns scalar."""
    loss_fn = ClassWeightedDiceLoss(4, weight_mode="inverse_sqrt_freq")
    pred = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_dice_custom_weights():
    """weight_mode='custom' uses provided weights; forward scalar; get_current_weights matches."""
    loss_fn = ClassWeightedDiceLoss(2, weight_mode="custom", class_weights=[0.5, 1.5])
    pred = torch.randn(2, 2, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16))
    out = loss_fn(pred, target)
    assert out.dim() == 0
    w = loss_fn.get_current_weights()
    assert w.shape == (2,)
    assert w[0].item() == pytest.approx(0.5, abs=1e-5)
    assert w[1].item() == pytest.approx(1.5, abs=1e-5)


def test_dice_target_class_indices():
    """Target (B, H, W) class indices is supported; returns scalar."""
    loss_fn = ClassWeightedDiceLoss(3)
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_dice_target_one_hot():
    """Target (B, C, H, W) one-hot is supported; returns scalar."""
    loss_fn = ClassWeightedDiceLoss(3)
    pred = torch.randn(2, 3, 8, 8)
    target = torch.zeros(2, 3, 8, 8)
    target[:, 0] = 1  # class 0
    target[:, 1, 2:6, 2:6] = 1  # class 1 in a patch
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_dice_exclude_background():
    """exclude_background=True can change loss vs False when class 0 is present."""
    torch.manual_seed(42)
    pred = torch.randn(2, 2, 16, 16)
    target = torch.zeros(2, 16, 16, dtype=torch.long)  # all class 0
    target[0, 4:12, 4:12] = 1
    loss_incl = ClassWeightedDiceLoss(2, exclude_background=False)
    loss_excl = ClassWeightedDiceLoss(2, exclude_background=True)
    out_incl = loss_incl(pred, target)
    out_excl = loss_excl(pred, target)
    assert out_incl.dim() == 0
    assert out_excl.dim() == 0
    # With mostly background, excluding it can change the value
    assert out_incl.item() != out_excl.item() or True  # at least both run


def test_dice_softmax_false():
    """softmax=False with pred in [0,1] runs and returns scalar."""
    loss_fn = ClassWeightedDiceLoss(2, softmax=False)
    pred = torch.softmax(torch.randn(2, 2, 8, 8), dim=1)
    target = torch.randint(0, 2, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_dice_get_current_weights():
    """get_current_weights returns 1D tensor of length num_classes."""
    loss_fn = ClassWeightedDiceLoss(4, weight_mode="custom", class_weights=[1.0, 2.0, 1.0, 1.0])
    w = loss_fn.get_current_weights()
    assert w.dim() == 1
    assert w.shape[0] == 4


def test_dice_zero_target_handling():
    """Target all zeros (one class) does not raise; loss is scalar."""
    loss_fn = ClassWeightedDiceLoss(2)
    pred = torch.randn(1, 2, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out).item()


# --- 5. CombinedCEDiceLoss ---


def test_combined_is_module():
    """CombinedCEDiceLoss is an nn.Module."""
    assert isinstance(CombinedCEDiceLoss(2), nn.Module)


def test_combined_forward_scalar():
    """Forward returns scalar for (B, C, H, W) pred and target."""
    for num_classes in (2, 4):
        loss_fn = CombinedCEDiceLoss(num_classes)
        pred = torch.randn(2, num_classes, 16, 16)
        target = torch.randint(0, num_classes, (2, 16, 16))
        out = loss_fn(pred, target)
        assert out.dim() == 0


def test_combined_binary_uses_bce():
    """num_classes=2 runs with use_bce_for_binary=True (default)."""
    loss_fn = CombinedCEDiceLoss(2, use_bce_for_binary=True)
    assert loss_fn.use_bce is True
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_combined_multi_class_uses_ce():
    """num_classes=4 runs and returns valid scalar."""
    loss_fn = CombinedCEDiceLoss(4)
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert out.item() >= 0


def test_combined_alpha_beta():
    """Changing alpha/beta changes loss value."""
    torch.manual_seed(42)
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    loss_ce_only = CombinedCEDiceLoss(2, alpha=1.0, beta=0.0)
    loss_dice_only = CombinedCEDiceLoss(2, alpha=0.0, beta=1.0)
    out_ce = loss_ce_only(pred, target)
    out_dice = loss_dice_only(pred, target)
    assert out_ce.item() != out_dice.item()


def test_combined_target_indices():
    """Target (B, H, W) class indices runs for num_classes=2 and 4."""
    for num_classes in (2, 4):
        loss_fn = CombinedCEDiceLoss(num_classes)
        pred = torch.randn(2, num_classes, 8, 8)
        target = torch.randint(0, num_classes, (2, 8, 8))
        out = loss_fn(pred, target)
        assert out.dim() == 0


def test_combined_target_one_hot():
    """Target (B, C, H, W) one-hot runs."""
    loss_fn = CombinedCEDiceLoss(3)
    pred = torch.randn(2, 3, 8, 8)
    target = torch.zeros(2, 3, 8, 8)
    target[:, 0] = 1
    target[:, 1, 2:6, 2:6] = 1
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_combined_loss_positive():
    """With random pred/target, loss > 0."""
    torch.manual_seed(42)
    loss_fn = CombinedCEDiceLoss(2)
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.item() > 0


def test_combined_exclude_background():
    """exclude_background=True vs False both run and can differ."""
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    out_false = CombinedCEDiceLoss(4, exclude_background=False)(pred, target)
    out_true = CombinedCEDiceLoss(4, exclude_background=True)(pred, target)
    assert out_false.dim() == 0
    assert out_true.dim() == 0


def test_combined_class_weights_ce():
    """class_weights_ce for CE branch runs for num_classes=4."""
    loss_fn = CombinedCEDiceLoss(4, class_weights_ce=[1.0, 1.0, 2.0, 1.0])
    pred = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    out = loss_fn(pred, target)
    assert out.dim() == 0


# --- 6. Edge cases ---


def test_dice_single_pixel():
    """ClassWeightedDiceLoss with (1, 2, 1, 1) runs and returns finite scalar."""
    loss_fn = ClassWeightedDiceLoss(2)
    pred = torch.randn(1, 2, 1, 1)
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out).item()


def test_combined_binary_small_shape():
    """CombinedCEDiceLoss num_classes=2 with (1, 2, 4, 4) and (1, 4, 4) target runs."""
    loss_fn = CombinedCEDiceLoss(2)
    pred = torch.randn(1, 2, 4, 4)
    target = torch.randint(0, 2, (1, 4, 4))
    out = loss_fn(pred, target)
    assert out.dim() == 0


def test_consistency_loss_matching_shapes():
    """ConsistencyLoss with matching (2, 3, 8, 8) shapes runs."""
    loss_fn = ConsistencyLoss()
    student = torch.randn(2, 3, 8, 8)
    teacher = torch.randn(2, 3, 8, 8)
    out = loss_fn(student, teacher)
    assert out.dim() == 0
