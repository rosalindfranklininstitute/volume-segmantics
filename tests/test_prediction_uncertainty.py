"""Tests UncertaintyProvider and MultiAxisTTAProvider.

"""

from __future__ import annotations

import math

import numpy as np
import pytest

from volume_segmantics.prediction.inference_modes import (
    get_inference_mode,
    list_inference_modes,
)
from volume_segmantics.prediction.uncertainty import (
    MultiAxisTTAProvider,
    UncertaintyOutputs,
    UncertaintyProvider,
    get_uncertainty_provider,
    list_uncertainty_providers,
    register_uncertainty_provider,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return (e / e.sum(axis=0, keepdims=True)).astype(np.float32)


@pytest.fixture
def three_pass_stacks():
    np.random.seed(0)
    return {
        "z_rot0": _softmax(np.random.randn(3, 4, 8, 8)),
        "y_rot0": _softmax(np.random.randn(3, 4, 8, 8)),
        "x_rot0": _softmax(np.random.randn(3, 4, 8, 8)),
    }


# Protocol + registry 


def test_provider_protocol_conformance():
    p = MultiAxisTTAProvider()
    assert isinstance(p, UncertaintyProvider)


def test_tta_provider_registered():
    assert "tta_uncertainty" in list_uncertainty_providers()
    assert get_uncertainty_provider("tta_uncertainty") is MultiAxisTTAProvider


def test_register_duplicate_raises():
    class _Stub:
        name = "stub"

        def compute(self, *a, **kw):
            return UncertaintyOutputs(teacher_argmax=np.zeros((1, 1, 1)))

    register_uncertainty_provider("stub_temp", _Stub)
    with pytest.raises(KeyError, match="already registered"):
        register_uncertainty_provider("stub_temp", _Stub)


def test_get_unknown_provider_raises():
    with pytest.raises(KeyError, match="unknown"):
        get_uncertainty_provider("hallucinated")


#  compute_from_stacks: error paths 


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        MultiAxisTTAProvider().compute_from_stacks({})


def test_3d_input_raises():
    with pytest.raises(ValueError, match=r"\(C, Z, Y, X\)"):
        MultiAxisTTAProvider().compute_from_stacks({
            "tag": np.zeros((4, 8, 8), dtype=np.float32),  # missing C
        })


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        MultiAxisTTAProvider().compute_from_stacks({
            "a": np.zeros((3, 4, 8, 8), dtype=np.float32),
            "b": np.zeros((3, 4, 8, 16), dtype=np.float32),  # X mismatch
        })


# Output shapes + dtypes 


def test_output_shape_dtype(three_pass_stacks):
    out = MultiAxisTTAProvider().compute_from_stacks(three_pass_stacks)
    assert out.teacher_argmax.shape == (4, 8, 8)
    assert out.teacher_argmax.dtype == np.uint8
    assert out.teacher_probs.shape == (3, 4, 8, 8)
    assert out.tta_variance_map.shape == (4, 8, 8)
    assert out.tta_entropy_map.shape == (4, 8, 8)


def test_variance_non_negative(three_pass_stacks):
    out = MultiAxisTTAProvider().compute_from_stacks(three_pass_stacks)
    assert float(out.tta_variance_map.min()) >= 0.0


def test_entropy_within_bounds(three_pass_stacks):
    out = MultiAxisTTAProvider().compute_from_stacks(three_pass_stacks)
    n_classes = 3
    log_c = math.log(n_classes)
    assert float(out.tta_entropy_map.min()) >= 0.0
    assert float(out.tta_entropy_map.max()) <= log_c + 1e-5


def test_identical_stacks_zero_variance():
    """Three identical passes -> variance map should be zeros."""
    p = _softmax(np.random.randn(3, 4, 8, 8))
    stacks = {
        "z_rot0": p.copy(),
        "y_rot0": p.copy(),
        "x_rot0": p.copy(),
    }
    out = MultiAxisTTAProvider().compute_from_stacks(stacks)
    assert float(out.tta_variance_map.max()) == pytest.approx(0.0, abs=1e-6)


def test_uniform_probs_entropy_at_max():
    """Uniform 1/C probability -> entropy = log(C) at every voxel."""
    n_classes = 4
    uniform = np.full((n_classes, 4, 8, 8), 1.0 / n_classes, dtype=np.float32)
    stacks = {
        "z_rot0": uniform.copy(),
        "y_rot0": uniform.copy(),
        "x_rot0": uniform.copy(),
    }
    out = MultiAxisTTAProvider().compute_from_stacks(stacks)
    expected = math.log(n_classes)
    assert float(out.tta_entropy_map.mean()) == pytest.approx(expected, abs=1e-4)


def test_per_axis_stash_when_enabled(three_pass_stacks):
    out = MultiAxisTTAProvider(store_per_axis=True).compute_from_stacks(
        three_pass_stacks,
    )
    assert set(out.per_axis_stash) == {"z_rot0", "y_rot0", "x_rot0"}
    for tag, arr in out.per_axis_stash.items():
        assert arr.shape == three_pass_stacks[tag].shape


def test_per_axis_stash_disabled(three_pass_stacks):
    out = MultiAxisTTAProvider(store_per_axis=False).compute_from_stacks(
        three_pass_stacks,
    )
    assert out.per_axis_stash == {}


def test_compute_alias_forwards_to_compute_from_stacks(three_pass_stacks):
    p = MultiAxisTTAProvider()
    out_a = p.compute(three_pass_stacks)
    out_b = p.compute_from_stacks(three_pass_stacks)
    np.testing.assert_array_equal(out_a.teacher_argmax, out_b.teacher_argmax)
    np.testing.assert_allclose(
        out_a.tta_variance_map, out_b.tta_variance_map,
    )


# UncertaintyOutputs.as_dict() 


def test_outputs_as_dict_emits_prediction_v1_keys(three_pass_stacks):
    out = MultiAxisTTAProvider().compute_from_stacks(three_pass_stacks)
    d = out.as_dict()
    assert "teacher_argmax" in d
    assert "teacher_probs" in d
    assert "tta_variance_map" in d
    assert "tta_entropy_map" in d
    # Per-axis stash NOT in as_dict (it's stash-only).
    assert "per_axis_stash" not in d


def test_outputs_as_dict_omits_none_keys():
    """Optional keys with ``None`` values are omitted."""
    out = UncertaintyOutputs(
        teacher_argmax=np.zeros((1, 1, 1), dtype=np.uint8),
    )
    d = out.as_dict()
    assert d == {"teacher_argmax": out.teacher_argmax}


#  12-pass smoke (quality: high) 


def test_12_pass_quality_high():
    """High-quality multi-axis = 3 axes × 4 rotations = 12 passes."""
    np.random.seed(2)
    stacks = {}
    for axis in ("z", "y", "x"):
        for rot in range(4):
            stacks[f"{axis}_rot{rot}"] = _softmax(
                np.random.randn(3, 4, 8, 8),
            )
    out = MultiAxisTTAProvider().compute_from_stacks(stacks)
    assert out.teacher_argmax.shape == (4, 8, 8)
    assert out.tta_variance_map.shape == (4, 8, 8)
    assert float(out.tta_variance_map.min()) >= 0.0


# inference_modes.py descriptors 


def test_all_four_modes_registered():
    assert set(list_inference_modes()) == {
        "single_axis", "multi_axis", "sliding_window", "tta_uncertainty",
    }


def test_tta_mode_requires_uncertainty_provider():
    desc = get_inference_mode("tta_uncertainty")
    assert desc.requires_uncertainty_provider is True


def test_other_modes_dont_require_provider():
    for name in ("single_axis", "multi_axis", "sliding_window"):
        desc = get_inference_mode(name)
        assert desc.requires_uncertainty_provider is False


def test_tta_mode_populates_variance_and_entropy():
    desc = get_inference_mode("tta_uncertainty")
    assert "tta_variance_map" in desc.populates
    assert "tta_entropy_map" in desc.populates


def test_unknown_mode_raises():
    with pytest.raises(KeyError, match="unknown"):
        get_inference_mode("hallucinated_mode")
