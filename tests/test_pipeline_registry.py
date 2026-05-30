"""Tests for ``volume_segmantics.data.pipeline_registry``.

"""

from __future__ import annotations

import pytest

from volume_segmantics.data import pipeline_registry as reg


# Test-local fixture: a clean registry per test 


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset all four registries to empty before each test, restore after.

    Production code populates the registries at import time; these tests
    need a clean slate to exercise registration semantics without relying
    on import order. We snapshot + restore so other tests in the suite
    that *do* rely on populated registries are unaffected.
    """
    snapshot = {
        "_HEADS":             dict(reg._HEADS),
        "_TARGET_GENERATORS": dict(reg._TARGET_GENERATORS),
        "_LOSSES":            dict(reg._LOSSES),
        "_TRANSFORMS":        dict(reg._TRANSFORMS),
    }
    reg._clear_all_registries_for_tests()
    try:
        yield
    finally:
        reg._clear_all_registries_for_tests()
        reg._HEADS.update(snapshot["_HEADS"])
        reg._TARGET_GENERATORS.update(snapshot["_TARGET_GENERATORS"])
        reg._LOSSES.update(snapshot["_LOSSES"])
        reg._TRANSFORMS.update(snapshot["_TRANSFORMS"])


# Heads 


class _DummyHead:
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels


def test_register_head_then_build():
    reg.register_head("dummy", _DummyHead)
    inst = reg.build_head("dummy", in_channels=3, out_channels=5)
    assert isinstance(inst, _DummyHead)
    assert inst.in_channels == 3
    assert inst.out_channels == 5


def test_duplicate_head_registration_raises():
    reg.register_head("dummy", _DummyHead)
    with pytest.raises(KeyError, match="already registered"):
        reg.register_head("dummy", _DummyHead)


def test_build_unknown_head_raises_with_known_list():
    reg.register_head("alpha", _DummyHead)
    reg.register_head("beta", _DummyHead)
    with pytest.raises(KeyError) as ei:
        reg.build_head("gamma")
    msg = str(ei.value)
    assert "gamma" in msg
    assert "alpha" in msg and "beta" in msg


def test_list_heads_returns_sorted():
    for name in ("zulu", "alpha", "mike"):
        reg.register_head(name, _DummyHead)
    assert reg.list_heads() == ["alpha", "mike", "zulu"]


# Target generators 


def _gen_factory(**kwargs):
    return lambda label_slice: label_slice  # identity


def test_register_target_generator_and_build():
    reg.register_target_generator("dummy_target", _gen_factory)
    fn = reg.build_target_generator("dummy_target")
    assert callable(fn)


def test_duplicate_target_generator_raises():
    reg.register_target_generator("dt", _gen_factory)
    with pytest.raises(KeyError):
        reg.register_target_generator("dt", _gen_factory)


def test_unknown_target_generator_raises():
    with pytest.raises(KeyError, match="unknown target generator"):
        reg.build_target_generator("missing")


# Losses 


def _loss_factory(**kwargs):
    def _call(pred, target, **extra):
        return pred  # placeholder
    return _call


def test_register_loss_then_build():
    reg.register_loss("custom_loss", _loss_factory)
    loss = reg.build_loss("custom_loss")
    assert callable(loss)


def test_duplicate_loss_registration_raises():
    reg.register_loss("custom_loss", _loss_factory)
    with pytest.raises(KeyError):
        reg.register_loss("custom_loss", _loss_factory)


def test_unknown_loss_raises():
    with pytest.raises(KeyError, match="unknown loss"):
        reg.build_loss("does_not_exist")


# Transforms 


def _identity_transform(**kwargs):
    return lambda image: image


def test_register_transform_and_build():
    reg.register_transform("Identity", _identity_transform)
    t = reg.build_transform("Identity")
    assert callable(t)


def test_duplicate_transform_raises():
    reg.register_transform("Flip", _identity_transform)
    with pytest.raises(KeyError):
        reg.register_transform("Flip", _identity_transform)


def test_unknown_transform_raises():
    with pytest.raises(KeyError, match="unknown transform"):
        reg.build_transform("NoSuchTransform")


# Cross-registry isolation 


def test_registries_are_independent():
    reg.register_head("name", _DummyHead)
    reg.register_loss("name", _loss_factory)
    reg.register_target_generator("name", _gen_factory)
    reg.register_transform("name", _identity_transform)
    # Same name in four registries does not collide.
    assert reg.list_heads() == ["name"]
    assert reg.list_losses() == ["name"]
    assert reg.list_target_generators() == ["name"]
    assert reg.list_transforms() == ["name"]
