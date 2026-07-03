"""Atomic file writes for durable artifacts.

A direct ``torch.save(obj, path)`` truncates the target before writing, so a
crash, full disk, or interrupt mid-write leaves a corrupt file -- and if that
target was the last known-good checkpoint, it is now unrecoverable. Writing to a
temporary file in the same directory and then ``os.replace``-ing it onto the
target makes the publish atomic: the target is either the old file or the fully
written new one, never a partial.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Union

import torch

__all__ = ["atomic_torch_save", "atomic_output_path"]


@contextmanager
def atomic_output_path(path: Union[str, Path]) -> Iterator[Path]:
    """Context manager yielding a temp path that is atomically published on exit.

    Use for writers that take a *path* (h5py, tifffile, mrcfile) rather than a
    file object: write to the yielded temp path, and on a clean exit it is
    ``os.replace``-d onto ``path`` (a same-directory, same-filesystem atomic
    rename). On any exception the temp file is removed and ``path`` is left as it
    was, so an interrupted write never leaves a partial output in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    """``torch.save`` ``obj`` to ``path`` atomically.

    Writes to a temp file in the same directory (so the final ``os.replace`` is a
    same-filesystem atomic rename), fsyncs it, then replaces the target. On any
    failure the temp file is removed and the existing target is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on the same filesystem
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
