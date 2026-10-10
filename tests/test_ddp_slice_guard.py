"""Tests for the DDP slice-once guard in train_2d_model.

Under multi-GPU DDP, Lightning's subprocess launcher re-runs the whole
training script for each non-zero rank. These tests cover the guard that
stops worker ranks from redundantly re-slicing the data (expensive,
non-idempotent) and instead has them wait for rank 0's slices. Pure
filesystem/env logic -- no GPU, model, or real slicing required.
"""

import os
import tempfile
from pathlib import Path

import pytest

import volume_segmantics.scripts.train_2d_model as m


@pytest.fixture()
def restore_local_rank():
    """Save/restore LOCAL_RANK so tests don't leak env state."""
    old = os.environ.get("LOCAL_RANK")
    yield
    if old is None:
        os.environ.pop("LOCAL_RANK", None)
    else:
        os.environ["LOCAL_RANK"] = old


def _set_rank(value):
    if value is None:
        os.environ.pop("LOCAL_RANK", None)
    else:
        os.environ["LOCAL_RANK"] = value


class TestRankDetection:
    def test_unset_is_rank_zero(self, restore_local_rank):
        _set_rank(None)
        assert m._local_rank() == 0
        assert not m._is_ddp_relaunched_worker()

    def test_explicit_zero_is_rank_zero(self, restore_local_rank):
        _set_rank("0")
        assert m._local_rank() == 0
        assert not m._is_ddp_relaunched_worker()

    def test_nonzero_is_worker(self, restore_local_rank):
        _set_rank("1")
        assert m._local_rank() == 1
        assert m._is_ddp_relaunched_worker()

    def test_malformed_defaults_to_zero(self, restore_local_rank):
        _set_rank("not-an-int")
        assert m._local_rank() == 0
        assert not m._is_ddp_relaunched_worker()


class TestSlicingSentinel:
    def test_mark_then_wait_returns_immediately(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            assert not m._slicing_sentinel_path(root).exists()
            m._mark_slicing_complete(root)
            assert m._slicing_sentinel_path(root).exists()
            # Present already -> fast path, no raise.
            m._wait_for_slicing_complete(root, timeout_s=1.0, poll_s=0.1)

    def test_clear_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            m._clear_slicing_sentinel(root)
            assert not m._slicing_sentinel_path(root).exists()
            m._clear_slicing_sentinel(root)  # missing file must not raise

    def test_wait_times_out_when_never_signalled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with pytest.raises(TimeoutError, match="waiting for rank 0"):
                m._wait_for_slicing_complete(root, timeout_s=0.2, poll_s=0.1)


class _FakeSlicer:
    def __init__(self):
        self.cleaned = False

    def clean_up_slices(self):
        self.cleaned = True


class TestRankZeroCleanup:
    def test_worker_rank_does_not_clean_or_clear(self, restore_local_rank):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            _set_rank("1")
            slicer = _FakeSlicer()
            m._rank0_cleanup_slices(slicer, root)
            assert slicer.cleaned is False
            assert m._slicing_sentinel_path(root).exists()

    def test_rank_zero_cleans_and_clears_sentinel(self, restore_local_rank):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            _set_rank("0")
            slicer = _FakeSlicer()
            m._rank0_cleanup_slices(slicer, root)
            assert slicer.cleaned is True
            assert not m._slicing_sentinel_path(root).exists()

    def test_rank_zero_with_no_slicer_does_not_crash(self, restore_local_rank):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            _set_rank("0")
            m._rank0_cleanup_slices(None, root)  # worker-path slicer is None
            assert not m._slicing_sentinel_path(root).exists()

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_rank_zero_keeps_slices_when_requested(self, restore_local_rank, monkeypatch, value):
        monkeypatch.setenv("VOLSEG_KEEP_SLICES", value)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            _set_rank("0")
            slicer = _FakeSlicer()
            m._rank0_cleanup_slices(slicer, root)
            assert slicer.cleaned is False
            assert not m._slicing_sentinel_path(root).exists()

    @pytest.mark.parametrize("value", ["", "0", "no"])
    def test_rank_zero_cleans_when_keep_not_requested(self, restore_local_rank, monkeypatch, value):
        monkeypatch.setenv("VOLSEG_KEEP_SLICES", value)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m._mark_slicing_complete(root)
            _set_rank("0")
            slicer = _FakeSlicer()
            m._rank0_cleanup_slices(slicer, root)
            assert slicer.cleaned is True
