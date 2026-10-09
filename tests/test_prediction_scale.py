"""Prediction at the training scale.

Training resized every slice so its longest side was ``image_size``;
prediction fed slices at their native size, so a 256 px volume trained at
512 was predicted at half the scale it learned. The checkpoint now records
the size and the predictor resizes its slices to it and its outputs back."""
import torch

from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor


def _bare(training_image_size, native_scale=False):
    p = VolSeg2dPredictor.__new__(VolSeg2dPredictor)
    p.training_image_size = training_image_size
    p.native_scale = native_scale
    return p


def test_no_recorded_size_or_matching_size_is_a_no_op():
    batch = torch.rand(2, 1, 64, 64)
    for size in (None, 60):
        out, frame = _bare(size)._to_training_scale(batch, [60, 60])
        assert out is batch and frame is None
        assert VolSeg2dPredictor._from_training_scale(batch, None) is batch


def test_resizes_by_the_longest_side_pads_to_the_divisor_and_restores():
    p = _bare(120)  # training slices were 120 on their longest side
    batch = torch.rand(1, 1, 64, 64)  # a 50 x 60 slice, padded to the network's 32
    scaled, frame = p._to_training_scale(batch, [50, 60])
    assert frame == (64, 64, 128, 128) and tuple(scaled.shape) == (1, 1, 128, 128)  # scale 2, already a multiple of 32
    scaled, frame = _bare(90)._to_training_scale(batch, [50, 60])  # scale 1.5: 96 x 96, padded to 96
    assert frame == (64, 64, 96, 96) and tuple(scaled.shape) == (1, 1, 96, 96)
    scaled, frame = _bare(100)._to_training_scale(batch, [50, 60])  # scale 5/3: 107 x 107, padded to 128
    assert frame == (64, 64, 107, 107) and tuple(scaled.shape) == (1, 1, 128, 128)
    logits = torch.rand(1, 2, 128, 128)
    assert tuple(VolSeg2dPredictor._from_training_scale(logits, frame).shape) == (1, 2, 64, 64)


def test_a_constant_slice_survives_the_round_trip():
    p = _bare(200)
    batch = torch.full((1, 1, 64, 64), 0.7)
    scaled, frame = p._to_training_scale(batch, [64, 64])
    assert torch.allclose(scaled, torch.full_like(scaled, 0.7))
    back = VolSeg2dPredictor._from_training_scale(scaled, frame)
    assert tuple(back.shape) == (1, 1, 64, 64) and torch.allclose(back, batch)


def test_native_scale_predicts_a_large_volume_at_its_own_size():
    """A model trained on 64-cubed crops must not resize a 700 px slice to 64.
    """
    batch = torch.rand(2, 1, 704, 704)
    out, frame = _bare(64, native_scale=True)._to_training_scale(batch, [700, 700])
    assert out is batch and frame is None


def test_native_scale_off_still_resizes_to_the_recorded_size():
    batch = torch.rand(1, 1, 704, 704)
    _, frame = _bare(64)._to_training_scale(batch, [700, 700])
    assert frame == (704, 704, 64, 64)
