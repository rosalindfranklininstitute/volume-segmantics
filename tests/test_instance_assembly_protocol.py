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
