"""The Dice losses and the boundary losses calculate sums in float32 for all
logit precisions.
"""
import pytest
import torch


def _boundary_batch(dtype):
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(8, 1, 512, 512, generator=generator).to(dtype).requires_grad_(True)
    target = (torch.rand(8, 1, 512, 512, generator=generator) > 0.9).float()
    return logits, target


@pytest.mark.parametrize("loss_name", ["boundary_bce_dice", "boundary_dice"])
def test_boundary_losses_survive_float16_logits(loss_name):
    """In float16, the Dice sums of a batch of slices became inf. Then the
    loss was a constant and the head got no gradient. CUDA autocast calculates
    ``sum`` in float32, thus ``16-mixed`` training did not have this fault.
    This test is for float16 logits that go to the loss outside autocast."""
    import volume_segmantics.model.loss_registry  # noqa: F401 - register losses
    from volume_segmantics.data import pipeline_registry

    loss_fn = pipeline_registry.build_loss(loss_name, num_classes=2)
    half_logits, target = _boundary_batch(torch.float16)
    half = loss_fn(half_logits, target)
    half.backward()
    full_logits, _ = _boundary_batch(torch.float32)
    full = loss_fn(full_logits, target)
    assert torch.isfinite(half) and half.dtype == torch.float32
    assert float(half) == pytest.approx(float(full), abs=2e-3)
    assert torch.isfinite(half_logits.grad).all() and float(half_logits.grad.abs().sum()) > 0


@pytest.mark.gpu
def test_boundary_bce_dice_trains_under_cuda_autocast():
    """This is the path of 16-mixed training. It was also correct before the float32 cast."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import volume_segmantics.model.loss_registry  # noqa: F401 - register losses
    from volume_segmantics.data import pipeline_registry

    loss_fn = pipeline_registry.build_loss("boundary_bce_dice", num_classes=2).cuda()
    head = torch.nn.Conv2d(1, 1, kernel_size=1).cuda()
    image, target = _boundary_batch(torch.float32)
    image, target = image.detach().cuda(), target.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = head(image)
        assert logits.dtype == torch.float16
        loss = loss_fn(logits, target)
    loss.backward()
    with torch.no_grad():
        reference = loss_fn(head(image).float(), target)
    assert torch.isfinite(loss) and float(loss) == pytest.approx(float(reference), abs=2e-3)
    assert float(head.weight.grad.abs().sum()) > 0
