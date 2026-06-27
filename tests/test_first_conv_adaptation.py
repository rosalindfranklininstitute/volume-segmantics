"""Tests for first-conv channel adaptation across encoder families.

`_adapt_first_conv_for_channels` rewrites the encoder's input stem when the
number of input channels changes (e.g. 1 -> 3 for 2.5D prediction). It locates
the stem as the first Conv2d in encoder traversal order; these tests pin that
this is the *true* stem for each supported encoder family by checking that a
forward pass with the new channel count succeeds and preserves the output shape
(a wrong layer would leave a channel mismatch and the forward would raise).

These run on CPU with randomly-initialised encoders (encoder_weights=None) so
no pretrained weights are downloaded.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

smp = pytest.importorskip("segmentation_models_pytorch")

from volume_segmantics.model.model_2d import _adapt_first_conv_for_channels


def _first_encoder_conv(model):
    return next(
        m for _, m in model.encoder.named_modules() if isinstance(m, nn.Conv2d)
    )


@pytest.mark.parametrize(
    "encoder", ["resnet18", "efficientnet-b0", "mobilenet_v2", "vgg11"]
)
def test_adapt_one_to_three_channels(encoder):
    model = smp.Unet(
        encoder_name=encoder, encoder_weights=None, in_channels=1, classes=2
    ).eval()
    x1 = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        out1 = model(x1)
    assert _first_encoder_conv(model).in_channels == 1

    _adapt_first_conv_for_channels(model, old_channels=1, new_channels=3)

    # The true input stem (not some deeper conv) must have been adapted.
    assert _first_encoder_conv(model).in_channels == 3
    x3 = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out3 = model(x3)  # raises if the wrong conv was replaced
    assert out3.shape == out1.shape


def test_adapt_three_to_one_channel():
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2
    ).eval()
    assert _first_encoder_conv(model).in_channels == 3
    _adapt_first_conv_for_channels(model, old_channels=3, new_channels=1)
    assert _first_encoder_conv(model).in_channels == 1
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 64, 64))
    assert out.shape[0] == 1


def test_adapt_noop_when_channels_already_match():
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2
    )
    stem_before = _first_encoder_conv(model)
    _adapt_first_conv_for_channels(model, old_channels=3, new_channels=3)
    # No replacement: the exact same module object remains in place.
    assert _first_encoder_conv(model) is stem_before


def test_adapt_one_to_three_preserves_mean_activation_scale():
    """The 1->N expansion divides by N so the summed activation stays comparable."""
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=2
    )
    orig_stem = _first_encoder_conv(model)
    orig_weight = orig_stem.weight.data.clone()
    _adapt_first_conv_for_channels(model, old_channels=1, new_channels=3)
    new_weight = _first_encoder_conv(model).weight.data
    # Each of the 3 new input channels carries the original weight / 3, so a
    # constant 3-channel input reproduces the original single-channel response.
    np.testing.assert_allclose(
        new_weight.sum(dim=1).numpy(),
        orig_weight.sum(dim=1).numpy(),
        rtol=1e-5, atol=1e-6,
    )
