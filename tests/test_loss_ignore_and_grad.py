"""Loss correctness under ignore-index, plus numerical gradient checks.

Covers the failure modes for ignore labels and backward-pass correctness. 
These are CPU, double-precision where gradcheck needs it, and use hand-constructed inputs.
"""

import torch
import torch.nn as nn

from volume_segmantics.data.pytorch3dunet_losses import (
    _MaskingLossWrapper,
    compute_per_channel_dice,
)


def test_masking_wrapper_matches_explicitly_zeroed_input():
    """Masking ignore_index positions == zeroing input & target there."""
    inner = nn.MSELoss()
    ignore = 9
    target = torch.tensor([[1.0, 9.0], [3.0, 9.0]])
    inp = torch.tensor([[2.0, 5.0], [4.0, 7.0]])

    wrapped = _MaskingLossWrapper(inner, ignore_index=ignore)
    got = wrapped(inp.clone(), target.clone())

    # Oracle: positions where target == ignore contribute as input=0, target=0.
    mask = target.ne(ignore).float()
    expected = inner(inp * mask, target * mask)
    assert torch.allclose(got, expected)


def test_masking_wrapper_zeroes_gradient_at_ignored_positions():
    """The whole point: no gradient flows where target == ignore_index."""
    inner = nn.MSELoss()
    ignore = 9
    target = torch.tensor([[1.0, 9.0], [9.0, 4.0]])
    inp = torch.tensor([[2.0, 5.0], [6.0, 7.0]], requires_grad=True)

    _MaskingLossWrapper(inner, ignore_index=ignore)(inp, target).backward()

    ignored = target.eq(ignore)
    assert torch.all(inp.grad[ignored] == 0)
    assert torch.any(inp.grad[~ignored] != 0)  # gradient still flows elsewhere


def test_per_channel_dice_perfect_and_disjoint_oracle():
    """Hand-computed boundary values: identical -> 1, disjoint -> 0."""
    target = torch.zeros(1, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0

    # Perfect prediction -> dice == 1
    dice_perfect = compute_per_channel_dice(target.clone(), target.clone())
    assert torch.allclose(dice_perfect, torch.ones(1), atol=1e-4)

    # Disjoint prediction -> dice ~ 0
    pred = torch.zeros(1, 1, 2, 2)
    pred[0, 0, 1, 1] = 1.0
    dice_disjoint = compute_per_channel_dice(pred, target)
    assert torch.allclose(dice_disjoint, torch.zeros(1), atol=1e-4)


def test_per_channel_dice_half_overlap_oracle():
    # input ones over 4 voxels, target ones over 2 of them:
    # intersect=2, |input|^2=4, |target|^2=2 -> dice = 2*2/(4+2) = 0.6667
    inp = torch.ones(1, 1, 1, 4)
    target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
    dice = compute_per_channel_dice(inp, target)
    assert torch.allclose(dice, torch.tensor([2 / 3]), atol=1e-4)


def test_per_channel_dice_gradcheck():
    """Numerically verify the dice backward pass (float64)."""
    torch.manual_seed(0)
    inp = torch.rand(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    target = (torch.rand(1, 2, 3, 3, dtype=torch.float64) > 0.5).double()

    def fn(x):
        return compute_per_channel_dice(x, target).sum()

    assert torch.autograd.gradcheck(fn, (inp,), eps=1e-6, atol=1e-4)
