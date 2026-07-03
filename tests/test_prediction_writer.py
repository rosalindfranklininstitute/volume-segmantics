"""Tests PredictionZarrWriter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from volume_segmantics.prediction.writer import (
    PREDICTION_SCHEMA_VERSION,
    PredictionZarrWriter,
    volseg_write_prediction_zarr,
)


# Constants 


def test_schema_version_constant():
    assert PREDICTION_SCHEMA_VERSION == "prediction_v1"


# Construction 


def test_writer_rejects_unknown_inference_mode(tmp_path):
    with pytest.raises(ValueError, match="inference_mode"):
        PredictionZarrWriter(
            output_path=tmp_path / "p.zarr",
            volume_shape=(8, 16, 16),
            inference_mode="hallucinated_mode",
            class_metadata={},
        )


def test_writer_rejects_bad_volume_shape(tmp_path):
    with pytest.raises(ValueError, match="volume_shape"):
        PredictionZarrWriter(
            output_path=tmp_path / "p.zarr",
            volume_shape=(8, 16),
            inference_mode="multi_axis",
            class_metadata={},
        )


def test_writer_creates_empty_store(tmp_path):
    out = tmp_path / "p.zarr"
    PredictionZarrWriter(
        output_path=out,
        volume_shape=(8, 16, 16),
        inference_mode="multi_axis",
        class_metadata={},
    )
    assert out.exists()
    g = zarr.open(str(out))
    # Empty store has only the (lazily-written) root group.
    assert list(g.keys()) == []


# teacher_argmax + finalize 


def test_finalize_raises_without_teacher_argmax(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    # Write some other array but not teacher_argmax.
    w.write_boundary_map(np.zeros((8, 16, 16), dtype=np.float32))
    with pytest.raises(RuntimeError, match="teacher_argmax"):
        w.finalize()


def test_minimal_finalize_with_only_teacher_argmax(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    path = w.finalize()
    assert path == out
    g = zarr.open(str(out))
    assert list(g.keys()) == ["teacher_argmax"]
    assert g.attrs["schema_version"] == "prediction_v1"


def _argmax_writer(tmp_path):
    return PredictionZarrWriter(
        output_path=tmp_path / "p.zarr", volume_shape=(2, 4, 4),
        inference_mode="multi_axis", class_metadata={},
    )


def test_teacher_argmax_accepts_max_uint8_label(tmp_path):
    """Label 255 is the boundary case and must round-trip exactly."""
    w = _argmax_writer(tmp_path)
    data = np.full((2, 4, 4), 255, dtype=np.int32)  # wide input dtype, in range
    w.write_teacher_argmax(data)
    w.finalize()
    g = zarr.open(str(tmp_path / "p.zarr"))
    assert int(np.asarray(g["teacher_argmax"]).max()) == 255


@pytest.mark.parametrize("bad_label", [256, 300, 1000])
def test_teacher_argmax_rejects_labels_above_255(tmp_path, bad_label):
    """Labels > 255 must raise, not silently wrap (e.g. 300 -> 44)."""
    w = _argmax_writer(tmp_path)
    data = np.zeros((2, 4, 4), dtype=np.int32)
    data[0, 0, 0] = bad_label
    with pytest.raises(ValueError, match="at most 256 classes"):
        w.write_teacher_argmax(data)


def test_teacher_argmax_rejects_negative_labels(tmp_path):
    w = _argmax_writer(tmp_path)
    data = np.zeros((2, 4, 4), dtype=np.int32)
    data[0, 0, 0] = -1
    with pytest.raises(ValueError, match=r"\[-1, 0\]"):
        w.write_teacher_argmax(data)


def test_finalize_locks_further_writes(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        w.write_boundary_map(np.zeros((8, 16, 16), dtype=np.float32))


def test_finalize_called_twice_raises(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.finalize()
    with pytest.raises(RuntimeError, match="already called"):
        w.finalize()


# Per-array shape contracts 


def test_teacher_argmax_rejects_float_dtype(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    with pytest.raises(TypeError, match="integer"):
        w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.float32))


def test_teacher_probs_4d_required(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    with pytest.raises(ValueError, match=r"\(C, Z, Y, X\)"):
        w.write_teacher_probs(np.zeros((8, 16, 16), dtype=np.float32))


def test_teacher_probs_class_count_set_on_first_write(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.write_teacher_probs(np.zeros((4, 8, 16, 16), dtype=np.float32))
    path = w.finalize()
    g = zarr.open(str(path))
    assert g["teacher_probs"].shape == (4, 8, 16, 16)


def test_boundary_map_accepts_3d_and_4d(tmp_path):
    """Both ``(Z, Y, X)`` and ``(1, Z, Y, X)`` boundary inputs are valid."""
    out_a = tmp_path / "a.zarr"
    w_a = PredictionZarrWriter(
        output_path=out_a, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w_a.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w_a.write_boundary_map(np.zeros((8, 16, 16), dtype=np.float32))  # 3D
    w_a.finalize()

    out_b = tmp_path / "b.zarr"
    w_b = PredictionZarrWriter(
        output_path=out_b, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w_b.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w_b.write_boundary_map(np.zeros((1, 8, 16, 16), dtype=np.float32))  # 4D
    w_b.finalize()

    g_a = zarr.open(str(out_a))
    g_b = zarr.open(str(out_b))
    assert g_a["boundary_map"].shape == (1, 8, 16, 16)
    assert g_b["boundary_map"].shape == (1, 8, 16, 16)


def test_sdm_map_supports_per_class_channels(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    # K=3 -> per-class SDM with 3 channels.
    w.write_sdm_map(np.zeros((3, 8, 16, 16), dtype=np.float32))
    path = w.finalize()
    g = zarr.open(str(path))
    assert g["sdm_map"].shape == (3, 8, 16, 16)


def test_tta_variance_shape_check(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="tta_uncertainty", class_metadata={},
    )
    with pytest.raises(ValueError, match=r"\(Z, Y, X\)"):
        w.write_tta_variance_map(np.zeros((1, 8, 16, 16), dtype=np.float32))


def test_per_axis_instances_axes_shape_validation(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    with pytest.raises(ValueError, match="must be"):
        w.write_per_axis_instances(
            {"diagonal": np.zeros((8, 16, 16), dtype=np.uint32)},
        )


# Optional-array policy + heads_present manifest 


def test_unwritten_arrays_absent_from_store(tmp_path):
    """**Optional-array policy**: don't zero-fill placeholders."""
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.write_boundary_map(np.zeros((8, 16, 16), dtype=np.float32))
    w.finalize()

    g = zarr.open(str(out))
    keys = sorted(g.keys())
    assert keys == ["boundary_map", "teacher_argmax"]
    # Negative — neither distance_map nor sdm_map nor tta_*_map present.
    assert "distance_map" not in keys
    assert "sdm_map" not in keys


def test_heads_present_auto_derived(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.write_boundary_map(np.zeros((8, 16, 16), dtype=np.float32))
    w.write_distance_map(np.zeros((8, 16, 16), dtype=np.float32))
    w.finalize()
    g = zarr.open(str(out))
    assert sorted(g.attrs["heads_present"]) == ["boundary", "distance", "semantic"]


def test_heads_present_override(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.finalize(heads_present=["semantic", "boundary", "distance", "sdm"])
    g = zarr.open(str(out))
    assert g.attrs["heads_present"] == [
        "semantic", "boundary", "distance", "sdm",
    ]


# Manifest provenance fields 


def test_manifest_carries_provenance_fields(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="tta_uncertainty",
        class_metadata={"classes": [{"id": 0, "name": "bg"}]},
        inference_config={"quality": "medium"},
        source_volume_path="/abs/path/to/data.h5",
        source_volume_hash="abcd1234",
        model_checkpoint_hash="deadbeef",
        voxel_size_nm=(10.0, 4.0, 4.0),
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.finalize(
        instance_assembly_backend="usegment3d",
        uncertainty_provider="tta_uncertainty",
    )
    g = zarr.open(str(out))
    a = dict(g.attrs)
    assert a["schema_version"] == "prediction_v1"
    assert a["inference_mode"] == "tta_uncertainty"
    assert a["inference_config"] == {"quality": "medium"}
    assert a["source_volume_path"] == "/abs/path/to/data.h5"
    assert a["source_volume_hash"] == "abcd1234"
    assert a["model_checkpoint_hash"] == "deadbeef"
    assert a["voxel_size_nm"] == [10.0, 4.0, 4.0]
    assert a["volume_shape"] == [8, 16, 16]
    assert a["uncertainty_provider"] == "tta_uncertainty"
    assert a["instance_assembly_backend"] == "usegment3d"


def test_manifest_extra_attrs_merged(tmp_path):
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(np.zeros((8, 16, 16), dtype=np.uint8))
    w.finalize(extra_attrs={"smoke_run": True, "trial": "alpha"})
    g = zarr.open(str(out))
    assert g.attrs["smoke_run"] is True
    assert g.attrs["trial"] == "alpha"


# Round-trip: write -> reload -> equality 


def test_round_trip_argmax_values(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4, size=(8, 16, 16), dtype=np.uint8)
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="multi_axis", class_metadata={},
    )
    w.write_teacher_argmax(arr)
    w.finalize()
    g = zarr.open(str(out))
    np.testing.assert_array_equal(g["teacher_argmax"][:], arr)


def test_round_trip_full_signal_set(tmp_path):
    rng = np.random.default_rng(1)
    out = tmp_path / "p.zarr"
    w = PredictionZarrWriter(
        output_path=out, volume_shape=(8, 16, 16),
        inference_mode="tta_uncertainty", class_metadata={},
    )
    w.write_teacher_argmax(
        rng.integers(0, 3, size=(8, 16, 16), dtype=np.uint8),
    )
    w.write_teacher_probs(
        rng.random((3, 8, 16, 16)).astype(np.float32),
    )
    w.write_boundary_map(rng.random((8, 16, 16)).astype(np.float32))
    w.write_distance_map(rng.random((8, 16, 16)).astype(np.float32))
    w.write_sdm_map(
        (rng.random((1, 8, 16, 16)).astype(np.float32) * 2 - 1),
    )
    w.write_tta_variance_map(rng.random((8, 16, 16)).astype(np.float32))
    w.write_tta_entropy_map(rng.random((8, 16, 16)).astype(np.float32))
    w.write_per_axis_instances({
        "xy": rng.integers(0, 100, size=(8, 16, 16), dtype=np.uint32),
        "xz": rng.integers(0, 100, size=(16, 8, 16), dtype=np.uint32),
    })
    w.write_instance_labels(
        rng.integers(0, 100, size=(8, 16, 16), dtype=np.uint32),
    )
    w.finalize(
        instance_assembly_backend="usegment3d",
        uncertainty_provider="tta_uncertainty",
    )

    g = zarr.open(str(out))
    assert g["teacher_argmax"].shape == (8, 16, 16)
    assert g["teacher_probs"].shape == (3, 8, 16, 16)
    assert g["boundary_map"].shape == (1, 8, 16, 16)
    assert g["distance_map"].shape == (1, 8, 16, 16)
    assert g["sdm_map"].shape == (1, 8, 16, 16)
    assert g["tta_variance_map"].shape == (8, 16, 16)
    assert g["tta_entropy_map"].shape == (8, 16, 16)
    assert g["per_axis_instances/xy"].shape == (8, 16, 16)
    assert g["per_axis_instances/xz"].shape == (16, 8, 16)
    assert g["instance_labels"].shape == (8, 16, 16)


# volseg_write_prediction_zarr standalone helper 


def test_helper_round_trip(tmp_path):
    out = tmp_path / "p.zarr"
    arrays = {
        "teacher_argmax": np.zeros((8, 16, 16), dtype=np.uint8),
        "teacher_probs": np.zeros((3, 8, 16, 16), dtype=np.float32),
        "boundary_map": np.zeros((8, 16, 16), dtype=np.float32),
        "tta_variance_map": np.zeros((8, 16, 16), dtype=np.float32),
        "per_axis_instances": {
            "xy": np.zeros((8, 16, 16), dtype=np.uint32),
        },
    }
    path = volseg_write_prediction_zarr(
        arrays, output_path=out,
        inference_mode="tta_uncertainty",
        uncertainty_provider="tta_uncertainty",
    )
    assert path == out
    g = zarr.open(str(out))
    assert sorted(g.keys()) == [
        "boundary_map", "per_axis_instances", "teacher_argmax",
        "teacher_probs", "tta_variance_map",
    ]


def test_helper_requires_teacher_argmax(tmp_path):
    with pytest.raises(ValueError, match="teacher_argmax"):
        volseg_write_prediction_zarr({}, output_path=tmp_path / "p.zarr")
