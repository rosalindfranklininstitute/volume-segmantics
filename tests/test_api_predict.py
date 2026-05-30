"""Test ``volume_segmantics.api.predict``.

"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import zarr

import volume_segmantics.model.heads        # noqa: F401
import volume_segmantics.model.loss_registry  # noqa: F401
import volume_segmantics.data.targets       # noqa: F401

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.api import (
    PredictionResult,
    load_data_extra,
    predict,
)
from volume_segmantics.data.pipeline_loader import (
    HeadConfig,
    PipelineConfig,
    PredictionConfig,
)


#  PredictionResult dataclass 


def test_prediction_result_defaults():
    r = PredictionResult()
    assert r.arrays == {}
    assert r.manifest == {}
    assert r.output_zarr is None


def test_prediction_result_with_arrays():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    r = PredictionResult(
        arrays={"teacher_argmax": arr},
        manifest={"inference_mode": "single_axis"},
    )
    assert r.arrays["teacher_argmax"] is arr
    assert r.manifest["inference_mode"] == "single_axis"


#  load_data_extra deferred stub 


def test_load_data_extra_deferred():
    with pytest.raises(NotImplementedError, match="curate-seg"):
        load_data_extra("/some/path")


#  api.predict surface 


def test_predict_requires_settings_or_path():
    with pytest.raises(ValueError, match="settings"):
        predict(model_path="/a", data_vol_path="/b")


def test_predict_unknown_inference_mode_raises(tmp_path):
    """Unknown mode raises at the registry lookup, before doing any work."""
    data_path = tmp_path / "data.h5"
    with h5py.File(data_path, "w") as f:
        f["/data"] = np.zeros((4, 16, 16), dtype=np.uint8)
    settings = _make_settings()
    with pytest.raises(KeyError, match="unknown inference mode"):
        predict(
            model_path=tmp_path / "no_model.pytorch",
            data_vol_path=data_path,
            settings=settings,
            inference_mode="hallucinated",
        )


#  End-to-end: build a tiny model + run predict 


def _make_settings() -> SimpleNamespace:
    """Minimal predict-side settings for the api smoke tests."""
    return SimpleNamespace(
        cuda_device=0,
        quality="medium",
        downsample=False,
        use_2_5d_prediction=False,
        num_slices=1,
        use_imagenet_norm=True,
        clip_data=False,
        st_dev_factor=2.575,
        minmax_norm=False,
        prediction_axis="Z",
        output_size=64,
        data_hdf5_path="/data",
        use_sliding_window=False,
        sw_roi_size=(64, 64),
        sw_overlap=0.25,
        sw_batch_size=2,
        sw_mode="gaussian",
        augmentation_library="albumentations",
        # Predictor expects these:
        output_probs=True,
        output_entropy=False,
        one_hot=False,
    )


@pytest.fixture
def tiny_model_path(tmp_path):
    """Create a tiny vol-seg-format checkpoint for the smoke tests."""
    import volume_segmantics.utilities.base_data_utils as utils
    import segmentation_models_pytorch as smp

    model_struc_dict = {
        "type": utils.ModelType.U_NET,
        "encoder_name": "resnet34",
        "encoder_weights": None,
        "in_channels": 1,
        "classes": 3,
    }
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=3,
    )
    out = tmp_path / "tiny.pytorch"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_struc_dict": model_struc_dict,
        "label_codes": {0: "bg", 1: "fg1", 2: "fg2"},
    }, out)
    return out


@pytest.fixture
def tiny_volume_path(tmp_path):
    """Create a tiny H5 volume for the smoke tests."""
    out = tmp_path / "vol.h5"
    rng = np.random.default_rng(0)
    with h5py.File(out, "w") as f:
        f["/data"] = (rng.random((4, 64, 64)) * 255).astype(np.uint8)
    return out


def test_predict_single_axis_returns_arrays(
    tiny_model_path, tiny_volume_path,
):
    settings = _make_settings()
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=settings,
        inference_mode="single_axis",
        return_arrays=True,
    )
    assert isinstance(result, PredictionResult)
    assert "teacher_argmax" in result.arrays
    assert result.arrays["teacher_argmax"].shape == (4, 64, 64)
    assert result.arrays["teacher_argmax"].dtype == np.uint8
    assert result.manifest["inference_mode"] == "single_axis"
    assert result.manifest["heads_present"] == ["semantic"]


def test_predict_writes_zarr(
    tiny_model_path, tiny_volume_path, tmp_path,
):
    out = tmp_path / "out.zarr"
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=_make_settings(),
        inference_mode="single_axis",
        output_zarr=out,
        return_arrays=False,
    )
    assert result.output_zarr == out
    g = zarr.open(str(out))
    assert "teacher_argmax" in g
    assert g.attrs["schema_version"] == "prediction_v1"
    assert g.attrs["inference_mode"] == "single_axis"


def test_predict_return_arrays_false_drops_arrays(
    tiny_model_path, tiny_volume_path,
):
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=_make_settings(),
        inference_mode="single_axis",
        return_arrays=False,
    )
    assert result.arrays == {}
    # Manifest still populated.
    assert result.manifest["inference_mode"] == "single_axis"


def test_predict_tta_uncertainty_produces_variance_and_entropy(
    tiny_model_path, tiny_volume_path, tmp_path,
):
    """**The B3.F.3 contract**: tta_uncertainty mode emits variance +
    entropy maps alongside the standard merged result."""
    out = tmp_path / "tta.zarr"
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=_make_settings(),
        inference_mode="tta_uncertainty",
        return_arrays=True,
        output_zarr=out,
    )
    assert "teacher_argmax" in result.arrays
    assert "tta_variance_map" in result.arrays
    assert "tta_entropy_map" in result.arrays
    assert result.arrays["tta_variance_map"].shape == (4, 64, 64)
    assert result.arrays["tta_entropy_map"].shape == (4, 64, 64)
    # variance ≥ 0; entropy ≥ 0.
    assert float(result.arrays["tta_variance_map"].min()) >= 0.0
    assert float(result.arrays["tta_entropy_map"].min()) >= 0.0
    # Manifest records the provider.
    assert result.manifest["uncertainty_provider"] == "tta_uncertainty"
    # Zarr written.
    g = zarr.open(str(out))
    assert "tta_variance_map" in g
    assert "tta_entropy_map" in g
    assert g.attrs["uncertainty_provider"] == "tta_uncertainty"


def test_predict_tta_uncertainty_auto_picks_provider(
    tiny_model_path, tiny_volume_path,
):
    """uncertainty_provider=None + inference_mode='tta_uncertainty' ->
    auto-set to 'tta_uncertainty'."""
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=_make_settings(),
        inference_mode="tta_uncertainty",
        uncertainty_provider=None,  # auto-pick
        return_arrays=True,
    )
    assert result.manifest["uncertainty_provider"] == "tta_uncertainty"


def test_predict_uses_pipeline_config_inference_mode_when_unset(
    tiny_model_path, tiny_volume_path,
):
    """When inference_mode kwarg is None, falls back to
    pipeline_config.prediction.inference_mode."""
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
        prediction=PredictionConfig(inference_mode="single_axis"),
    )
    result = predict(
        model_path=tiny_model_path,
        data_vol_path=tiny_volume_path,
        settings=_make_settings(),
        pipeline_config=cfg,
        inference_mode=None,
        return_arrays=True,
    )
    assert result.manifest["inference_mode"] == "single_axis"
