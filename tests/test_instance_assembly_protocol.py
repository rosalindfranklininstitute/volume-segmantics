"""Tests for InstanceAssembler Protocol, registry, bundle.

"""

from __future__ import annotations

import numpy as np
import pytest

from volume_segmantics.inference.instance_assembly import (
    AssemblyConfig,
    InstanceAssembler,
    InstanceAssemblerInputError,
    PredictionBundle,
    get_backend,
    list_backends,
    register_backend,
)


#  Protocol conformance 


class _StubAssembler:
    """Bare-minimum assembler that conforms to the Protocol."""

    name = "stub"
    required_fields = ("semantic_argmax",)

    def assemble(self, bundle, config):  # noqa: ANN001
        return np.zeros(bundle.semantic_argmax.shape, dtype=np.uint32)


def test_protocol_runtime_checkable_accepts_stub():
    assert isinstance(_StubAssembler(), InstanceAssembler)


def test_protocol_runtime_checkable_rejects_non_conforming():
    class _NoAssemble:
        name = "no_assemble"
        required_fields = ()

    assert not isinstance(_NoAssemble(), InstanceAssembler)


#  AssemblyConfig 


def test_assembly_config_defaults():
    cfg = AssemblyConfig()
    assert cfg.foreground_class_ids is None
    assert cfg.voxel_size == (1.0, 1.0, 1.0)


def test_assembly_config_carries_voxel_size():
    cfg = AssemblyConfig(voxel_size=(2.0, 4.0, 4.0))
    assert cfg.voxel_size == (2.0, 4.0, 4.0)


def test_assembly_config_foreground_filter():
    cfg = AssemblyConfig(foreground_class_ids=(1, 2))
    assert cfg.foreground_class_ids == (1, 2)


#  PredictionBundle 


def test_bundle_has_recognises_set_fields():
    sem = np.zeros((2, 4, 4), dtype=np.uint8)
    bundle = PredictionBundle(semantic_argmax=sem)
    assert bundle.has("semantic_argmax")
    assert not bundle.has("distance_map")
    assert not bundle.has("boundary_map")


def test_bundle_require_passes_when_fields_present():
    sem = np.zeros((2, 4, 4), dtype=np.uint8)
    dist = np.zeros((1, 2, 4, 4), dtype=np.float32)
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    bundle.require(("semantic_argmax", "distance_map"))


def test_bundle_require_raises_lists_missing_and_present():
    sem = np.zeros((2, 4, 4), dtype=np.uint8)
    bundle = PredictionBundle(semantic_argmax=sem)
    with pytest.raises(InstanceAssemblerInputError) as excinfo:
        bundle.require(("semantic_argmax", "distance_map", "sdm_map"))
    msg = str(excinfo.value)
    assert "distance_map" in msg
    assert "sdm_map" in msg
    assert "semantic_argmax" in msg  # listed as present.


def test_bundle_require_unknown_attr_treated_as_missing():
    """Unknown field names are treated as missing rather than raising
    AttributeError — gives backends a sensible failure mode if they
    declare a typo'd field."""
    bundle = PredictionBundle(semantic_argmax=np.zeros((1, 1, 1), np.uint8))
    with pytest.raises(InstanceAssemblerInputError):
        bundle.require(("bogus_field",))


def test_bundle_per_axis_instances_optional():
    bundle = PredictionBundle()
    assert bundle.per_axis_instances is None
    bundle.per_axis_instances = {"xy": np.zeros((2, 4, 4), dtype=np.uint32)}
    assert bundle.has("per_axis_instances")


#  Registry 


def test_list_backends_returns_sorted_list():
    backends = list_backends()
    assert backends == sorted(backends)


def test_register_backend_duplicate_raises():
    register_backend("stub_dup_test", _StubAssembler)
    with pytest.raises(KeyError, match="already registered"):
        register_backend("stub_dup_test", _StubAssembler)


def test_get_backend_unknown_raises_with_known_list():
    with pytest.raises(KeyError, match="unknown instance-assembly backend"):
        get_backend("hallucinated_backend")


def test_get_backend_returns_class_not_instance():
    register_backend("stub_class_test", _StubAssembler)
    cls = get_backend("stub_class_test")
    assert cls is _StubAssembler


#  InstanceAssemblerInputError 


def test_input_error_is_value_error():
    """Subclass of ValueError so callers' generic ``except ValueError``
    handlers catch it without pulling our hierarchy in directly."""
    assert issubclass(InstanceAssemblerInputError, ValueError)


#  SliceOverlapAssembler (baseline single-axis 3D stitcher)


def _two_column_per_axis(n_slices=5):
    """(Z, Y, X) per-axis map: two separate objects, labels restart per slice."""
    vol = np.zeros((n_slices, 12, 12), dtype=np.uint32)
    for z in range(n_slices):
        vol[z, 1:4, 1:4] = 1   # object A (top-left), same xy every slice
        vol[z, 7:11, 7:11] = 2  # object B (bottom-right)
    return vol


def test_slice_overlap_registered():
    assert "slice_overlap" in list_backends()
    assert isinstance(get_backend("slice_overlap")(), InstanceAssembler)


def test_slice_overlap_links_columns_into_two_instances():
    vol = _two_column_per_axis()
    asm = get_backend("slice_overlap")(axis="xy", min_overlap=1)
    bundle = PredictionBundle(
        semantic_argmax=(vol > 0).astype(np.uint8),
        per_axis_instances={"xy": vol},
    )
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 2  # two 3D instances
    # each object is a single connected 3D label
    assert out[:, 1:4, 1:4].max() == out[:, 1:4, 1:4].min()  # uniform label A
    assert out[0, 1, 1] != out[0, 7, 7]  # A and B distinct


def test_slice_overlap_heals_per_slice_split():
    # One object split into two labels in slice 0, whole in the others ->
    # overlap-linking should unify it into ONE 3D instance.
    vol = np.zeros((3, 10, 10), dtype=np.uint32)
    vol[0, 2:5, 2:8] = 1
    vol[0, 5:8, 2:8] = 2   # same object, split in two in slice 0
    vol[1, 2:8, 2:8] = 1
    vol[2, 2:8, 2:8] = 1
    asm = get_backend("slice_overlap")(axis="xy", min_overlap=1)
    bundle = PredictionBundle(
        semantic_argmax=(vol > 0).astype(np.uint8),
        per_axis_instances={"xy": vol},
    )
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 1


def test_slice_overlap_min_size_drops_small():
    vol = _two_column_per_axis(n_slices=5)
    # object A = 3x3x5 = 45 voxels; object B = 4x4x5 = 80 voxels.
    asm = get_backend("slice_overlap")(axis="xy", min_overlap=1, min_size=50)
    bundle = PredictionBundle(
        semantic_argmax=(vol > 0).astype(np.uint8),
        per_axis_instances={"xy": vol},
    )
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 1  # only B (80 vox) survives


def test_slice_overlap_missing_axis_raises():
    asm = get_backend("slice_overlap")(axis="xy")
    bundle = PredictionBundle(
        semantic_argmax=np.zeros((2, 4, 4), np.uint8),
        per_axis_instances={"xz": np.zeros((4, 2, 4), np.uint32)},
    )
    with pytest.raises(InstanceAssemblerInputError, match="missing axis"):
        asm.assemble(bundle, AssemblyConfig())


#  Watershed3DAssembler (direct 3D h-maxima watershed)


def _two_blob_bundle():
    """Two separated 3D blobs + a foreground-EDT distance map peaking at each."""
    from scipy.ndimage import distance_transform_edt
    sem = np.zeros((6, 32, 32), dtype=np.uint8)
    sem[:, 5:13, 5:13] = 1     # blob A
    sem[:, 19:27, 19:27] = 1   # blob B
    dist = distance_transform_edt(sem > 0).astype(np.float32)
    return PredictionBundle(semantic_argmax=sem, distance_map=dist), sem


def test_watershed_3d_registered():
    assert "watershed_3d" in list_backends()
    assert isinstance(get_backend("watershed_3d")(), InstanceAssembler)


def test_watershed_3d_splits_two_blobs_head():
    bundle, _ = _two_blob_bundle()
    asm = get_backend("watershed_3d")(source="head", h=0.2, smooth_sigma=0.0)
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 2
    # distinct labels for the two blobs, background stays 0
    assert out[0, 8, 8] != out[0, 22, 22]
    assert out[0, 0, 0] == 0


def test_watershed_3d_semantic_edt_needs_no_distance_head():
    _, sem = _two_blob_bundle()
    bundle = PredictionBundle(semantic_argmax=sem)  # no distance_map
    asm = get_backend("watershed_3d")(source="semantic_edt", h=0.2, smooth_sigma=0.0)
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 2


def test_watershed_3d_min_size_drops_small():
    from scipy.ndimage import distance_transform_edt
    sem = np.zeros((4, 30, 30), dtype=np.uint8)
    sem[:, 4:8, 4:8] = 1       # small blob: 4*4*4 = 64 voxels
    sem[:, 15:25, 15:25] = 1   # large blob: 10*10*4 = 400 voxels
    dist = distance_transform_edt(sem > 0).astype(np.float32)
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    asm = get_backend("watershed_3d")(source="head", h=0.2, smooth_sigma=0.0, min_size=100)
    out = asm.assemble(bundle, AssemblyConfig())
    assert int(np.unique(out).size) - 1 == 1  # only the large blob survives
