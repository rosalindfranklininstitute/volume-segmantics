"""Tests for atomic_torch_save (PLAN.md Step 38).

A direct torch.save truncates the target before writing, so an interrupted save
corrupts the last good checkpoint. atomic_torch_save publishes via os.replace so
the target is never left partial.
"""

import os

import pytest
import torch

from volume_segmantics.utilities.atomic_io import (
    atomic_output_path,
    atomic_torch_save,
)


def test_atomic_save_round_trips(tmp_path):
    obj = {"a": torch.arange(5), "b": "meta"}
    path = tmp_path / "ckpt.pytorch"
    atomic_torch_save(obj, path)
    loaded = torch.load(path, weights_only=False)
    assert torch.equal(loaded["a"], obj["a"])
    assert loaded["b"] == "meta"


def test_atomic_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "ckpt.pytorch"
    atomic_torch_save({"x": torch.zeros(2)}, path)
    assert path.is_file()


def test_atomic_save_leaves_no_temp_files(tmp_path):
    path = tmp_path / "ckpt.pytorch"
    atomic_torch_save({"x": torch.zeros(2)}, path)
    # Only the final file remains; no leftover *.tmp from the temp-then-rename.
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pytorch"]


def test_failed_save_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pytorch"
    atomic_torch_save({"v": torch.tensor([1])}, path)  # known-good checkpoint
    good_bytes = path.read_bytes()

    # Simulate a crash mid-write (after the temp file is created).
    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        atomic_torch_save({"v": torch.tensor([2])}, path)

    # The previous good checkpoint is untouched...
    assert path.read_bytes() == good_bytes
    # ...and no temp file was left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pytorch"]


def test_atomic_save_overwrites_existing(tmp_path):
    path = tmp_path / "ckpt.pytorch"
    atomic_torch_save({"v": torch.tensor([1])}, path)
    atomic_torch_save({"v": torch.tensor([2])}, path)
    assert torch.load(path, weights_only=False)["v"].item() == 2


# --- atomic_output_path (path-based writers: h5py/tifffile/mrcfile) -------- #

def test_atomic_output_path_publishes_on_success(tmp_path):
    path = tmp_path / "out.bin"
    with atomic_output_path(path) as tmp:
        assert tmp != path
        tmp.write_bytes(b"final")
    assert path.read_bytes() == b"final"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]  # no temp left


def test_atomic_output_path_preserves_existing_on_failure(tmp_path):
    path = tmp_path / "out.bin"
    path.write_bytes(b"good")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_output_path(path) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("boom")
    assert path.read_bytes() == b"good"  # existing output untouched
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]  # temp cleaned up


def test_save_data_functions_are_atomic(tmp_path):
    """The wired save_data_to_* helpers leave no temp files after a clean write."""
    import numpy as np
    import volume_segmantics.utilities.base_data_utils as utils

    vol = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    utils.save_data_to_hdf5(vol, tmp_path / "a.h5")
    utils.save_data_to_tif(vol, tmp_path / "b.tiff")
    utils.save_data_to_mrc(vol, tmp_path / "c.mrc")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["a.h5", "b.tiff", "c.mrc"]  # no .tmp leftovers
