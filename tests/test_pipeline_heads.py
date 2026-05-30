"""Tests for head modules and build_head_modules.

"""

from __future__ import annotations

import pytest
import torch

import volume_segmantics.model.heads  # triggers registration
from volume_segmantics.data import pipeline_registry as reg
from volume_segmantics.data.pipeline_loader import HeadConfig
from volume_segmantics.model.heads import (
    BoundaryHead,
    DistanceHead,
    SDMHead,
    SemanticHead,
    build_head_modules,
)
from volume_segmantics.model.heads.base import PredictionHead, TargetKind


# Registration 


def test_all_four_heads_registered():
    assert set(reg.list_heads()) >= {
        "semantic", "boundary", "distance", "sdm",
    }


def test_built_heads_satisfy_protocol():
    for name in ("semantic", "boundary", "distance", "sdm"):
        if name == "semantic":
            h = reg.build_head(name, in_channels=8, out_channels=3,
                               num_classes=3)
        elif name == "sdm":
            h = reg.build_head(name, in_channels=8, num_classes=3,
                               variant="per_class")
        else:
            h = reg.build_head(name, in_channels=8)
        assert isinstance(h, PredictionHead)
        assert h.name == name
        assert isinstance(h.target_kind, TargetKind)


#  Per-head forward 


@pytest.fixture
def feat():
    return torch.randn(2, 16, 32, 32)


def test_semantic_forward_shape(feat):
    h = SemanticHead(in_channels=16, out_channels=4, num_classes=4)
    out = h(feat)
    assert out.shape == (2, 4, 32, 32)
    assert h.out_channels == 4


def test_semantic_zero_classes_raises():
    with pytest.raises(ValueError, match=">= 1"):
        SemanticHead(in_channels=16, out_channels=0)


def test_boundary_forward_shape(feat):
    h = BoundaryHead(in_channels=16)
    out = h(feat)
    assert out.shape == (2, 1, 32, 32)
    assert h.out_channels == 1


def test_boundary_rejects_multi_channel():
    with pytest.raises(ValueError, match="must be 1"):
        BoundaryHead(in_channels=16, out_channels=4)


def test_distance_forward_shape_and_range(feat):
    h = DistanceHead(in_channels=16)
    out = h(feat)
    assert out.shape == (2, 1, 32, 32)
    # Identity output — random conv produces unbounded values; .
    assert torch.isfinite(out).all()


def test_distance_rejects_multi_channel():
    with pytest.raises(ValueError, match="must be 1"):
        DistanceHead(in_channels=16, out_channels=2)


def test_sdm_binary_forward(feat):
    h = SDMHead(in_channels=16, num_classes=2, variant="binary")
    out = h(feat)
    assert out.shape == (2, 1, 32, 32)
    # tanh-bounded.
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= -1.0


def test_sdm_per_class_forward(feat):
    h = SDMHead(in_channels=16, num_classes=4, variant="per_class")
    out = h(feat)
    assert out.shape == (2, 3, 32, 32)  # num_classes - 1 = 3
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= -1.0


def test_sdm_per_class_with_one_class_raises():
    with pytest.raises(ValueError, match="num_classes >= 2"):
        SDMHead(in_channels=16, num_classes=1, variant="per_class")


def test_sdm_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        SDMHead(in_channels=16, num_classes=2, variant="trinary")


def test_sdm_negative_d_clip_raises():
    with pytest.raises(ValueError, match="d_clip"):
        SDMHead(in_channels=16, num_classes=2, d_clip=-1.0)


#  3D rejected  


@pytest.mark.parametrize("cls,kwargs", [
    (SemanticHead, dict(in_channels=16, out_channels=4)),
    (BoundaryHead, dict(in_channels=16)),
    (DistanceHead, dict(in_channels=16)),
    (SDMHead, dict(in_channels=16, num_classes=2)),
])
def test_3d_rejected(cls, kwargs):
    with pytest.raises(ValueError, match="2D"):
        cls(spatial_dims=3, **kwargs)


#  build_head_modules 


def test_build_head_modules_preserves_dict_order():
    cfg = {
        "boundary": HeadConfig(enabled=True, loss="bce"),
        "semantic": HeadConfig(enabled=True, loss="dice_ce"),
        "sdm":      HeadConfig(enabled=True, loss="sdm_l1"),
    }
    heads = build_head_modules(cfg, in_channels=8, num_classes=3)
    assert [h.name for h in heads] == ["boundary", "semantic", "sdm"]


def test_build_head_modules_skips_disabled():
    cfg = {
        "semantic": HeadConfig(enabled=True),
        "boundary": HeadConfig(enabled=False, loss="bce"),
        "distance": HeadConfig(enabled=True, loss="distance_l1"),
    }
    heads = build_head_modules(cfg, in_channels=8, num_classes=2)
    assert [h.name for h in heads] == ["semantic", "distance"]


def test_build_head_modules_semantic_default_out_channels():
    cfg = {"semantic": HeadConfig(enabled=True)}
    heads = build_head_modules(cfg, in_channels=8, num_classes=5)
    assert heads[0].out_channels == 5


def test_build_head_modules_explicit_out_channels_overrides_default():
    cfg = {"semantic": HeadConfig(enabled=True, out_channels=7)}
    heads = build_head_modules(cfg, in_channels=8, num_classes=5)
    assert heads[0].out_channels == 7


def test_build_head_modules_sdm_binary_default_one_channel():
    cfg = {
        "semantic": HeadConfig(enabled=True),
        "sdm": HeadConfig(enabled=True, extra={"variant": "binary"}),
    }
    heads = build_head_modules(cfg, in_channels=8, num_classes=4)
    sdm = next(h for h in heads if h.name == "sdm")
    assert sdm.out_channels == 1


def test_build_head_modules_sdm_per_class_resolves_to_n_minus_1():
    cfg = {
        "semantic": HeadConfig(enabled=True),
        "sdm": HeadConfig(enabled=True, extra={"variant": "per_class"}),
    }
    heads = build_head_modules(cfg, in_channels=8, num_classes=4)
    sdm = next(h for h in heads if h.name == "sdm")
    assert sdm.out_channels == 3


def test_build_head_modules_dim_3_raises():
    cfg = {"semantic": HeadConfig(enabled=True)}
    with pytest.raises(ValueError, match="3D path is deferred"):
        build_head_modules(cfg, in_channels=8, num_classes=2, dim=3)


def test_build_head_modules_empty_when_all_disabled():
    cfg = {"semantic": HeadConfig(enabled=False)}
    heads = build_head_modules(cfg, in_channels=8, num_classes=2)
    assert heads == []
