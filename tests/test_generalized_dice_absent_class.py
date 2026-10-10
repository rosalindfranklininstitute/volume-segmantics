"""Generalized Dice must not let a class that is absent from the batch decide
the loss.
"""

import torch

from volume_segmantics.data.pytorch3dunet_losses import GeneralizedDiceLoss


def _three_class_case(absent_everywhere: bool):
    """Logits that predict classes 0 and 1 confidently in two halves of a 16x16
    image, plus a target in which class 2 never appears."""
    target = torch.zeros(1, 3, 16, 16)
    target[0, 0, :, :8] = 1.0
    target[0, 1, :, 8:] = 1.0
    logits = torch.full((1, 3, 16, 16), -8.0)
    if absent_everywhere:
        logits[0, 2] = 8.0  # the absent class predicted everywhere
    else:
        logits[0, 0, :, :8] = 8.0
        logits[0, 1, :, 8:] = 8.0
    return logits.requires_grad_(True), target


def test_near_perfect_prediction_with_an_absent_class_has_a_small_loss():
    logits, target = _three_class_case(absent_everywhere=False)
    loss = GeneralizedDiceLoss(normalization="softmax")(logits, target)
    assert loss.item() < 0.05, loss.item()


def test_predicting_the_absent_class_everywhere_is_still_penalised():
    logits, target = _three_class_case(absent_everywhere=True)
    loss = GeneralizedDiceLoss(normalization="softmax")(logits, target)
    assert loss.item() > 0.9, loss.item()


def test_the_present_classes_receive_gradient_when_one_class_is_absent():
    logits, target = _three_class_case(absent_everywhere=False)
    logits_wrong = logits.detach().clone()
    logits_wrong[0, 0, :, :8] = 0.0  # weaken class 0 where it is the truth
    logits_wrong.requires_grad_(True)
    loss = GeneralizedDiceLoss(normalization="softmax")(logits_wrong, target)
    loss.backward()
    grad_class0 = logits_wrong.grad[0, 0, :, :8].abs().sum().item()
    grad_class2 = logits_wrong.grad[0, 2].abs().sum().item()
    assert grad_class0 > 0
    assert grad_class0 > 0.1 * grad_class2, (grad_class0, grad_class2)


def test_an_empty_target_gives_a_finite_loss_and_gradient():
    logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 3, 8, 8)
    loss = GeneralizedDiceLoss(normalization="softmax")(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
