"""Tests for per-axis 2D instance producers.

"""

from __future__ import annotations

import numpy as np
import pytest

from volume_segmantics.inference.instance_assembly import (
    InstanceAssemblerInputError,
    PredictionBundle,
)
from volume_segmantics.prediction.per_axis_instances import (
    DistanceWatershedSliceProducer,
    PerAxisInstanceProducer,
    SemanticCcSliceProducer,
    VALID_AXES,
    get_producer,
    list_producers,
    register_producer,
    select_producer_name,
)


#  Protocol conformance 


def test_distance_watershed_conforms():
    p = DistanceWatershedSliceProducer()
    assert isinstance(p, PerAxisInstanceProducer)
    assert p.name == "distance_watershed"
    assert "distance_map" in p.required_bundle_fields


def test_semantic_cc_conforms():
    p = SemanticCcSliceProducer()
    assert isinstance(p, PerAxisInstanceProducer)
    assert p.name == "semantic_cc"
    assert p.required_bundle_fields == ("semantic_argmax",)


#  Registry 


def test_distance_watershed_registered():
    assert "distance_watershed" in list_producers()


def test_semantic_cc_registered():
    assert "semantic_cc" in list_producers()


def test_register_producer_dup_raises():
    with pytest.raises(KeyError, match="already registered"):
        register_producer("semantic_cc", SemanticCcSliceProducer)


def test_get_producer_unknown_raises():
    with pytest.raises(KeyError, match="unknown per-axis producer"):
        get_producer("hallucinated_producer")


#  select_producer_name 


def test_select_returns_explicit_when_set():
    assert (
        select_producer_name("semantic_cc", enabled_heads=["semantic"])
        == "semantic_cc"
    )


def test_select_explicit_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        select_producer_name("hallucinated", enabled_heads=["semantic"])


def test_select_auto_distance_when_distance_enabled():
    name = select_producer_name(
        None, enabled_heads=["semantic", "distance"],
    )
    assert name == "distance_watershed"


def test_select_auto_semantic_cc_when_no_distance():
    name = select_producer_name(None, enabled_heads=["semantic"])
    assert name == "semantic_cc"


def test_select_auto_semantic_cc_when_only_boundary():
    name = select_producer_name(
        None, enabled_heads=["semantic", "boundary"],
    )
    assert name == "semantic_cc"


#  SemanticCcSliceProducer 


def _semantic_volume_two_blobs(
    xy_layout: bool = True,
) -> np.ndarray:
    """(Z, Y, X) uint8 with two well-separated foreground blobs.

    Two 3x3 blobs in plane y=2..4,x=2..4 and y=2..4,x=10..12, both
    repeated across all Z slices.
    """
    sem = np.zeros((4, 8, 16), dtype=np.uint8)
    sem[:, 2:5, 2:5] = 1
    sem[:, 2:5, 10:13] = 1
    return sem


def test_semantic_cc_basic_two_blobs_xy():
    sem = _semantic_volume_two_blobs()
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(bundle, ("xy",), {})
    assert "xy" in out
    assert out["xy"].shape == sem.shape
    assert out["xy"].dtype == np.uint32
    # Each Z slice should have exactly 2 instances + background.
    for z in range(sem.shape[0]):
        unique = np.unique(out["xy"][z])
        # densely labelled: {0, 1, 2}
        assert set(unique.tolist()) == {0, 1, 2}


def test_semantic_cc_4_vs_8_connectivity():
    """Two diagonally-adjacent pixels: 4-conn -> 2 components,
    8-conn -> 1 component."""
    sem = np.zeros((1, 4, 4), dtype=np.uint8)
    sem[0, 0, 0] = 1
    sem[0, 1, 1] = 1
    bundle = PredictionBundle(semantic_argmax=sem)

    out_4 = SemanticCcSliceProducer().produce(
        bundle, ("xy",), {"connectivity": 1},
    )
    assert int(out_4["xy"][0].max()) == 2

    out_8 = SemanticCcSliceProducer().produce(
        bundle, ("xy",), {"connectivity": 2},
    )
    assert int(out_8["xy"][0].max()) == 1


def test_semantic_cc_invalid_connectivity_raises():
    sem = np.zeros((1, 4, 4), dtype=np.uint8)
    bundle = PredictionBundle(semantic_argmax=sem)
    with pytest.raises(ValueError, match="connectivity"):
        SemanticCcSliceProducer().produce(
            bundle, ("xy",), {"connectivity": 3},
        )


def test_semantic_cc_foreground_class_filter():
    """Class-id filter: only class 2 voxels become foreground."""
    sem = np.zeros((1, 4, 4), dtype=np.uint8)
    sem[0, 0, 0] = 1   # class 1 — filtered out.
    sem[0, 2, 2] = 2   # class 2 — kept.
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(
        bundle, ("xy",), {"foreground_class_ids": [2]},
    )
    # Only the (2,2) pixel survives.
    assert int(out["xy"].sum()) == 1


def test_semantic_cc_empty_foreground_returns_zeros():
    sem = np.zeros((2, 4, 4), dtype=np.uint8)
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(bundle, ("xy",), {})
    assert int(out["xy"].max()) == 0


def test_semantic_cc_axis_xz_shape():
    """xz axis: slice along Y -> output stack shape (Y, Z, X)."""
    sem = _semantic_volume_two_blobs()
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(bundle, ("xz",), {})
    Z, Y, X = sem.shape
    assert out["xz"].shape == (Y, Z, X)


def test_semantic_cc_axis_yz_shape():
    """yz axis: slice along X -> output stack shape (X, Z, Y)."""
    sem = _semantic_volume_two_blobs()
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(bundle, ("yz",), {})
    Z, Y, X = sem.shape
    assert out["yz"].shape == (X, Z, Y)


def test_semantic_cc_three_axes_all_present():
    sem = _semantic_volume_two_blobs()
    bundle = PredictionBundle(semantic_argmax=sem)
    out = SemanticCcSliceProducer().produce(bundle, VALID_AXES, {})
    assert set(out.keys()) == set(VALID_AXES)


def test_semantic_cc_unknown_axis_raises():
    sem = _semantic_volume_two_blobs()
    bundle = PredictionBundle(semantic_argmax=sem)
    with pytest.raises(InstanceAssemblerInputError, match="unknown axis"):
        SemanticCcSliceProducer().produce(bundle, ("bogus",), {})


def test_semantic_cc_missing_bundle_field_raises():
    bundle = PredictionBundle()  # no semantic_argmax
    with pytest.raises(InstanceAssemblerInputError):
        SemanticCcSliceProducer().produce(bundle, ("xy",), {})


#  DistanceWatershedSliceProducer 


def _distance_volume_two_peaks() -> np.ndarray:
    """Distance map (Z, Y, X) with two clearly-separated peaks per slice.

    Peak at (y=3, x=3) and (y=3, x=11), both replicated across Z.
    Built from a synthetic FG mask (3x3 squares) + Euclidean distance
    transform-style cone.
    """
    Z, Y, X = 2, 8, 16
    dist = np.zeros((Z, Y, X), dtype=np.float32)
    yy, xx = np.indices((Y, X))
    # Cone-style distance to (3, 3): max where r=0.
    cone1 = np.maximum(0.0, 4.0 - np.hypot(yy - 3, xx - 3))
    cone2 = np.maximum(0.0, 4.0 - np.hypot(yy - 3, xx - 11))
    plane = np.maximum(cone1, cone2)
    for z in range(Z):
        dist[z] = plane
    return dist


def _semantic_two_peaks_match_distance() -> np.ndarray:
    """FG mask (Z, Y, X) covering both 3x3 blobs the distance peaks live in."""
    Z, Y, X = 2, 8, 16
    sem = np.zeros((Z, Y, X), dtype=np.uint8)
    sem[:, 1:6, 1:6] = 1   # blob covering (3,3) peak.
    sem[:, 1:6, 9:14] = 1  # blob covering (3,11) peak.
    return sem


def test_distance_watershed_two_instances_per_slice():
    sem = _semantic_two_peaks_match_distance()
    dist = _distance_volume_two_peaks()
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    out = DistanceWatershedSliceProducer().produce(
        bundle, ("xy",), {"peak_min_distance": 3},
    )
    assert out["xy"].shape == sem.shape
    assert out["xy"].dtype == np.uint32
    for z in range(sem.shape[0]):
        unique = set(np.unique(out["xy"][z]).tolist())
        # densely labelled: {0, 1, 2}.
        assert unique == {0, 1, 2}


def test_distance_watershed_accepts_4d_distance_map():
    """``(1, Z, Y, X)`` distance map (the writer's canonical layout) is
    accepted by the producer just as ``(Z, Y, X)`` is."""
    sem = _semantic_two_peaks_match_distance()
    dist3 = _distance_volume_two_peaks()
    dist4 = dist3[np.newaxis, ...]  # (1, Z, Y, X)
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist4)
    out = DistanceWatershedSliceProducer().produce(
        bundle, ("xy",), {"peak_min_distance": 3},
    )
    assert out["xy"].shape == sem.shape


def test_distance_watershed_shape_mismatch_raises():
    sem = np.zeros((2, 8, 16), dtype=np.uint8)
    dist = np.zeros((2, 8, 32), dtype=np.float32)  # X mismatch
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    with pytest.raises(ValueError, match="does not match"):
        DistanceWatershedSliceProducer().produce(bundle, ("xy",), {})


def test_distance_watershed_invalid_peak_min_distance_raises():
    sem = _semantic_two_peaks_match_distance()
    dist = _distance_volume_two_peaks()
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    with pytest.raises(ValueError, match="peak_min_distance"):
        DistanceWatershedSliceProducer().produce(
            bundle, ("xy",), {"peak_min_distance": 0},
        )


def test_distance_watershed_empty_foreground_returns_zeros():
    sem = np.zeros((2, 8, 16), dtype=np.uint8)
    dist = np.zeros((2, 8, 16), dtype=np.float32)
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    out = DistanceWatershedSliceProducer().produce(bundle, ("xy",), {})
    assert int(out["xy"].max()) == 0


def test_distance_watershed_no_peaks_falls_back_to_centroid():
    """Tiny FG with no clear peak still produces a single-instance slice."""
    Z, Y, X = 1, 6, 6
    sem = np.zeros((Z, Y, X), dtype=np.uint8)
    sem[0, 2:4, 2:4] = 1
    # Constant distance inside FG -> no local max under high min_distance.
    dist = np.zeros((Z, Y, X), dtype=np.float32)
    dist[0, 2:4, 2:4] = 1.0
    bundle = PredictionBundle(semantic_argmax=sem, distance_map=dist)
    out = DistanceWatershedSliceProducer().produce(
        bundle, ("xy",), {"peak_min_distance": 5},
    )
    # Single instance (label = 1) over the FG square.
    unique = set(np.unique(out["xy"][0]).tolist())
    assert unique == {0, 1}


def test_distance_watershed_missing_distance_map_raises():
    sem = np.zeros((2, 8, 16), dtype=np.uint8)
    bundle = PredictionBundle(semantic_argmax=sem)  # no distance_map
    with pytest.raises(InstanceAssemblerInputError):
        DistanceWatershedSliceProducer().produce(bundle, ("xy",), {})
