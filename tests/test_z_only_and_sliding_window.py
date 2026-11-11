"""``quality: z_only`` and ``use_sliding_window`` operate as their names say.

Before this change, ``z_only`` did the 12-way prediction on all three axes.
Sliding window had no effect for a checkpoint that records its training
``image_size``. Each slice changed to that size first and was not larger than
the window.
"""
from types import SimpleNamespace

import numpy as np
import torch

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model.operations import vol_seg_2d_predictor as predictor_module
from volume_segmantics.model.operations.vol_seg_2d_predictor import VolSeg2dPredictor
from volume_segmantics.utilities.base_data_utils import Axis


def _threshold_predictor(calls, depths):
    predictor = VolSeg2dPredictor.__new__(VolSeg2dPredictor)
    predictor._last_additional_tasks = None

    def predict_single_axis(data_vol, output_probs=True, axis=Axis.Z):
        calls.append(axis)
        depths.append(data_vol.shape[0])
        labels = (data_vol > 127).astype(np.uint8)
        probs = np.stack([1.0 - labels, labels], axis=-1).astype(np.float16)
        return labels, probs, None

    predictor._predict_single_axis = predict_single_axis
    return predictor


def test_z_only_predicts_along_z_in_four_rotations_and_stays_aligned():
    calls = []
    volume = np.random.randint(0, 256, size=(4, 6, 10))
    depths = []
    labels, probs = _threshold_predictor(calls, depths)._predict_Zonly_max_probs(volume)
    assert calls == [Axis.Z] * 4
    assert depths == [4] * 4  # each rotation is around z, thus the slice axis does not change
    assert labels.shape == volume.shape and probs.shape == volume.shape
    assert np.array_equal(labels, (volume > 127).astype(np.uint8))


def test_the_manager_routes_z_only_to_the_z_only_predictor():
    from volume_segmantics.model.operations.vol_seg_prediction_manager import VolSeg2DPredictionManager

    volume = np.random.randint(0, 256, size=(4, 6, 10))
    called = []

    class Predictor:
        def _predict_Zonly_max_probs(self, data_vol):
            called.append("z_only")
            return np.zeros(data_vol.shape, np.uint8), np.ones(data_vol.shape, np.float16)

        def _predict_12_ways_max_probs(self, data_vol):  # pragma: no cover - the old route
            called.append("12_ways")
            return np.zeros(data_vol.shape, np.uint8), np.ones(data_vol.shape, np.float16)

        def get_additional_task_outputs(self):
            return None

    manager = VolSeg2DPredictionManager.__new__(VolSeg2DPredictionManager)
    manager.predictor = Predictor()
    manager.data_vol = volume
    manager.settings = SimpleNamespace(
        quality="z_only", one_hot=False, output_probs=False, output_entropy=False,
        z_smooth=False, prediction_axis="Z",
    )
    manager.predict_volume_to_path(None, quality=utils.Quality.Z_ONLY)
    assert called == ["z_only"]


# Sliding window


def _sliding_window_predictor(monkeypatch, *, training_image_size, roi):
    predictor = VolSeg2dPredictor.__new__(VolSeg2dPredictor)
    predictor.model = torch.nn.Conv2d(1, 2, kernel_size=1)
    predictor.training_image_size = training_image_size
    predictor.native_scale = False
    predictor.use_sliding_window = True
    predictor.sw_roi_size = roi
    predictor.sw_overlap = 0.25
    predictor.sw_batch_size = 2
    predictor.sw_mode = "gaussian"
    predictor.model_device_num = "cpu"
    windows = []
    real = predictor_module.sliding_window_inference

    def spy(*args, **kwargs):
        windows.append(tuple(kwargs["inputs"].shape[-2:]))
        return real(*args, **kwargs)

    monkeypatch.setattr(predictor_module, "sliding_window_inference", spy)
    return predictor, windows


def test_sliding_window_runs_on_a_slice_larger_than_its_window(monkeypatch):
    """Before this change, the slice changed to the recorded training
    image_size first. Then the slice was not larger than the window and the
    setting had no effect."""
    predictor, windows = _sliding_window_predictor(monkeypatch, training_image_size=64, roi=(64, 64))
    with torch.no_grad():
        labels, probs, logits = predictor._predict_slice_sliding_window(torch.rand(1, 1, 128, 160), [128, 160])
    assert windows == [(128, 160)]  # the tiles use the native slice size
    assert labels.shape == (1, 128, 160)


def test_a_slice_that_fits_the_window_is_predicted_whole_at_the_training_scale(monkeypatch):
    predictor, windows = _sliding_window_predictor(monkeypatch, training_image_size=32, roi=(64, 64))
    seen = []
    predictor.model.register_forward_hook(lambda module, args, output: seen.append(tuple(args[0].shape[-2:])))
    with torch.no_grad():
        labels, _, _ = predictor._predict_slice_sliding_window(torch.rand(1, 1, 64, 64), [64, 64])
    assert windows == [] and seen == [(32, 32)]
    assert labels.shape == (1, 64, 64)
