"""Tests for ``api.predict`` instance-assembly dispatch.


"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import zarr

import volume_segmantics.model.heads        # noqa: F401  (registry side effects)
import volume_segmantics.model.loss_registry  # noqa: F401
import volume_segmantics.data.targets       # noqa: F401

from volume_segmantics.api import _run_instance_assembly, predict
from volume_segmantics.data.pipeline_loader import (
    HeadConfig,
    InstanceAssemblyConfig,
    PerAxisInstancesConfig,
    PipelineConfig,
    PredictionConfig,
)
from volume_segmantics.inference.instance_assembly import list_backends


_HAS_USEGMENT3D = "usegment3d" in list_backends()


#  Helpers 


def _structured_semantic_volume() -> np.ndarray:
    """``(Z, Y, X) uint8`` argmax with two well-separated 5x5x5 blobs.

    Designed to make uSegment3D's indirect aggregation succeed: clean
    foreground voxels, no scattered noise, both blobs span all Z slices.
    """
    sem = np.zeros((6, 32, 32), dtype=np.uint8)
    sem[:, 8:13, 8:13] = 1
    sem[:, 8:13, 20:25] = 1
    return sem


def _pipeline(
    *,
    cfg_backend: str | None = "usegment3d",
    producer: str | None = None,
    axes=("xy", "xz", "yz"),
    paxis_params: dict | None = None,
    asm_params: dict | None = None,
) -> PipelineConfig:
    """Build a PipelineConfig.

    Note: ``cfg_backend`` is the name pinned on
    :class:`InstanceAssemblyConfig.backend`, which the loader validates
    against :data:`KNOWN_ASSEMBLY_BACKENDS`. Tests that exercise a
    different backend pass it via ``_run_instance_assembly``'s own
    ``backend_name`` argument, since that's the kwarg the dispatcher
    actually reads.
    """
    if asm_params is None:
        asm_params = {
            "params_overrides": {
                "indirect_method": {"dtform_method": "edt"},
            },
        }
    return PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
        prediction=PredictionConfig(
            inference_mode="single_axis",
            per_axis_instances=PerAxisInstancesConfig(
                enabled=True,
                producer=producer,
                axes=tuple(axes),
                params=paxis_params or {},
            ),
        ),
        instance_assembly=InstanceAssemblyConfig(
            backend=cfg_backend,
            params=asm_params,
        ),
    )


# _run_instance_assembly: producer-only path (no uSegment3D) 
# These tests cover the producer dispatch + error paths without
# relying on the optional uSegment3D extra. The end-to-end flow is in
# the :func:`predict` block below.


class _IdentityAssembler:
    """Drop-in stand-in for uSegment3D — returns the xy producer output.

    Lets the dispatch tests run without the real assembly engine, so
    the test focus stays on argument plumbing rather than upstream
    aggregation correctness.
    """

    name = "identity"
    required_fields = ("per_axis_instances",)

    def __init__(self, **kwargs):  # accept any params
        self.kwargs = kwargs

    def assemble(self, bundle, config):
        per_axis = bundle.per_axis_instances
        # Pick an axis the producer populated.
        for axis in ("xy", "xz", "yz"):
            if axis in per_axis:
                arr = per_axis[axis]
                if axis == "xy":
                    return arr.astype(np.uint32, copy=False)
                if axis == "xz":
                    # (Y, Z, X) -> (Z, Y, X)
                    return np.transpose(arr, (1, 0, 2)).astype(np.uint32)
                # yz: (X, Z, Y) -> (Z, Y, X)
                return np.transpose(arr, (1, 2, 0)).astype(np.uint32)
        raise RuntimeError("identity assembler: no per-axis maps available")


@pytest.fixture
def identity_backend():
    """Register an identity backend for the duration of the test."""
    from volume_segmantics.inference.instance_assembly import (
        register_backend,
    )
    name = "identity_test_backend"
    # Skip dup-register if a previous test left it registered.
    from volume_segmantics.inference.instance_assembly import _BACKENDS
    if name not in _BACKENDS:
        register_backend(name, _IdentityAssembler)
    yield name


def test_run_assembly_adds_arrays(identity_backend):
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(asm_params={})
    _run_instance_assembly(
        arrays=arrays,
        backend_name=identity_backend,
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    assert "per_axis_instances" in arrays
    assert set(arrays["per_axis_instances"].keys()) == {"xy", "xz", "yz"}
    assert "instance_labels" in arrays
    assert arrays["instance_labels"].shape == sem.shape
    assert arrays["instance_labels"].dtype == np.uint32


def test_run_assembly_auto_selects_semantic_cc(identity_backend):
    """Producer=None + only semantic populated -> semantic_cc auto-pick."""
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(producer=None, asm_params={})
    _run_instance_assembly(
        arrays=arrays,
        backend_name=identity_backend,
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    # semantic_cc on two well-separated blobs -> 2 instances per slice.
    paxis = arrays["per_axis_instances"]
    assert int(paxis["xy"][0].max()) >= 2


def test_run_assembly_auto_selects_distance_watershed(identity_backend):
    """Producer=None + distance_map present -> distance_watershed picked."""
    sem = _structured_semantic_volume()
    Z, Y, X = sem.shape
    yy, xx = np.indices((Y, X))
    cone1 = np.maximum(0.0, 4.0 - np.hypot(yy - 10, xx - 10))
    cone2 = np.maximum(0.0, 4.0 - np.hypot(yy - 10, xx - 22))
    plane = np.maximum(cone1, cone2).astype(np.float32)
    distance_map = np.broadcast_to(plane, (1, Z, Y, X)).copy()

    arrays = {"teacher_argmax": sem, "distance_map": distance_map}
    cfg = _pipeline(
        producer=None,
        paxis_params={"peak_min_distance": 3},
        asm_params={},
    )
    _run_instance_assembly(
        arrays=arrays,
        backend_name=identity_backend,
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    paxis = arrays["per_axis_instances"]
    # Two peaks -> two instances per slice.
    assert int(paxis["xy"][0].max()) == 2


def test_run_assembly_explicit_producer(identity_backend):
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(
        producer="semantic_cc",
        paxis_params={"connectivity": 1},
        asm_params={},
    )
    _run_instance_assembly(
        arrays=arrays,
        backend_name=identity_backend,
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    assert int(arrays["per_axis_instances"]["xy"].max()) >= 2


def test_run_assembly_subset_axes(identity_backend):
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(
        axes=("xy", "xz"),
        asm_params={},
    )
    _run_instance_assembly(
        arrays=arrays,
        backend_name=identity_backend,
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    assert set(arrays["per_axis_instances"].keys()) == {"xy", "xz"}


def test_run_assembly_voxel_size_flows_to_config(identity_backend):
    """voxel_size kwarg -> AssemblyConfig.voxel_size on the call.

    Captured via the identity backend's stored kwargs.
    """
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(asm_params={})

    captured = {}

    class _CapturingAssembler(_IdentityAssembler):
        def assemble(self, bundle, config):
            captured["voxel_size"] = config.voxel_size
            captured["fg_class_ids"] = config.foreground_class_ids
            return super().assemble(bundle, config)

    from volume_segmantics.inference.instance_assembly import _BACKENDS
    _BACKENDS["identity_capture_test"] = _CapturingAssembler

    _run_instance_assembly(
        arrays=arrays,
        backend_name="identity_capture_test",
        pipeline_config=cfg,
        voxel_size=(2.0, 4.0, 4.0),
    )
    assert captured["voxel_size"] == (2.0, 4.0, 4.0)
    assert captured["fg_class_ids"] is None


def test_run_assembly_foreground_class_ids_flows(identity_backend):
    sem = _structured_semantic_volume()
    arrays = {"teacher_argmax": sem}
    cfg = _pipeline(
        asm_params={"foreground_class_ids": [1, 2]},
    )

    captured = {}

    class _Capture(_IdentityAssembler):
        def assemble(self, bundle, config):
            captured["fg"] = config.foreground_class_ids
            return super().assemble(bundle, config)

    from volume_segmantics.inference.instance_assembly import _BACKENDS
    _BACKENDS["fg_capture_test"] = _Capture

    _run_instance_assembly(
        arrays=arrays,
        backend_name="fg_capture_test",
        pipeline_config=cfg,
        voxel_size=(1.0, 1.0, 1.0),
    )
    assert captured["fg"] == (1, 2)


def test_run_assembly_missing_teacher_argmax_raises():
    arrays = {}
    cfg = _pipeline(asm_params={})
    with pytest.raises(RuntimeError, match="teacher_argmax"):
        _run_instance_assembly(
            arrays=arrays,
            backend_name="usegment3d",
            pipeline_config=cfg,
            voxel_size=(1.0, 1.0, 1.0),
        )


#  End-to-end through api.predict (uSegment3D-gated) 


pytest_e2e = pytest.mark.skipif(
    not _HAS_USEGMENT3D,
    reason="USegment3DAssembler not registered — install [usegment3d] extra.",
)


def _make_settings() -> SimpleNamespace:
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
        output_probs=True,
        output_entropy=False,
        one_hot=False,
    )


@pytest.fixture
def seeded_tiny_model_path(tmp_path):
    """Tiny vol-seg-format checkpoint with a torch-seeded random init.

    Determinism matters here — uSegment3D's gradient pipeline
    occasionally chokes on degenerate argmax volumes the random
    init can produce. Seeding fixes that for the smoke gates.
    """
    import volume_segmantics.utilities.base_data_utils as utils
    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    np.random.seed(0)
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
def structured_volume_path(tmp_path):
    """An H5 volume with two clearly-distinguished regions.

    Avoids fully random data so the random model still produces
    *some* structured argmax.
    """
    out = tmp_path / "vol.h5"
    rng = np.random.default_rng(0)
    Z, Y, X = 6, 64, 64
    base = (rng.random((Z, Y, X)) * 60).astype(np.uint8)
    base[:, 16:24, 16:24] = 220   # bright square
    base[:, 16:24, 40:48] = 220   # second bright square
    with h5py.File(out, "w") as f:
        f["/data"] = base
    return out


@pytest_e2e
def test_predict_with_assembly_kwarg_writes_zarr(
    seeded_tiny_model_path, structured_volume_path, tmp_path,
):
    """Predict + assembly + zarr write — full e2e roundtrip."""
    cfg = _pipeline()
    out = tmp_path / "assembly.zarr"
    result = predict(
        model_path=seeded_tiny_model_path,
        data_vol_path=structured_volume_path,
        settings=_make_settings(),
        pipeline_config=cfg,
        inference_mode="single_axis",
        instance_assembly_backend="usegment3d",
        output_zarr=out,
        return_arrays=True,
    )
    assert "per_axis_instances" in result.arrays
    assert set(result.arrays["per_axis_instances"].keys()) == {
        "xy", "xz", "yz",
    }
    assert "instance_labels" in result.arrays
    assert result.arrays["instance_labels"].dtype == np.uint32
    assert result.manifest["instance_assembly_backend"] == "usegment3d"

    # Zarr layout.
    g = zarr.open(str(out))
    assert "instance_labels" in g
    assert "per_axis_instances" in g
    assert set(g["per_axis_instances"].array_keys()) == {"xy", "xz", "yz"}
    assert g.attrs["instance_assembly_backend"] == "usegment3d"


@pytest_e2e
def test_predict_assembly_from_pipeline_config_only(
    seeded_tiny_model_path, structured_volume_path,
):
    """No instance_assembly_backend kwarg + cfg.backend set -> assembly runs."""
    cfg = _pipeline()
    result = predict(
        model_path=seeded_tiny_model_path,
        data_vol_path=structured_volume_path,
        settings=_make_settings(),
        pipeline_config=cfg,
        inference_mode="single_axis",
        instance_assembly_backend=None,
        return_arrays=True,
    )
    assert "instance_labels" in result.arrays
    assert result.manifest["instance_assembly_backend"] == "usegment3d"


def test_predict_no_assembly_when_backend_unset(
    seeded_tiny_model_path, structured_volume_path,
):
    """Both kwarg + cfg.backend ``None`` -> no assembly arrays."""
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
        prediction=PredictionConfig(inference_mode="single_axis"),
    )
    result = predict(
        model_path=seeded_tiny_model_path,
        data_vol_path=structured_volume_path,
        settings=_make_settings(),
        pipeline_config=cfg,
        inference_mode="single_axis",
        instance_assembly_backend=None,
        return_arrays=True,
    )
    assert "instance_labels" not in result.arrays
    assert "per_axis_instances" not in result.arrays
    assert result.manifest["instance_assembly_backend"] is None
