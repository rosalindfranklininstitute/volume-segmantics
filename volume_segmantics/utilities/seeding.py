"""Reproducibility helpers.

Centralises the seeding of Python's ``random``, NumPy, and PyTorch (CPU + CUDA)
so that training and prediction runs can be made deterministic when a seed is
supplied. No seeding happens implicitly at import time -- callers opt in by
calling :func:`set_seed`.

* **Tier A -- CPU bitwise.** ``set_seed(seed)`` makes ``random``/NumPy/torch RNG
  state reproducible; CPU ops are then bitwise-reproducible run-to-run.
* **Tier B -- GPU tolerance.** ``set_seed(seed, deterministic=True)`` additionally
  requests deterministic cuDNN/algorithm selection. Some CUDA kernels have no
  deterministic implementation; with ``warn_only=True`` (the default) those warn
  rather than raise, so results are reproducible to a numeric tolerance, not
  necessarily bit-for-bit.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed", "seed_worker", "make_generator"]


def set_seed(seed: Optional[int], *, deterministic: bool = False,
             warn_only: bool = True) -> Optional[int]:
    """Seed all RNGs used by the library.

    Args:
        seed: The seed to apply. ``None`` is a no-op (preserves the existing,
            non-deterministic behaviour) and returns ``None``.
        deterministic: If ``True``, also request deterministic algorithm
            selection (Tier B): set ``CUBLAS_WORKSPACE_CONFIG``, disable cuDNN
            benchmarking, and call :func:`torch.use_deterministic_algorithms`.
        warn_only: Passed to :func:`torch.use_deterministic_algorithms`. When
            ``True`` (default), ops lacking a deterministic kernel warn instead
            of raising -- appropriate for the GPU tolerance tier.

    Returns:
        The seed that was applied, or ``None`` if seeding was skipped.
    """
    if seed is None:
        return None
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Covers all current and future CUDA devices; safe to call without CUDA.
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS requires this to be set *before* the first CUDA work for
        # deterministic GEMMs; setting it here is harmless on CPU-only runs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only avoids hard failures on ops without deterministic kernels
        # (e.g. some upsampling/scatter paths) while still pinning the rest.
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`.

    Module-level (hence picklable, so it survives the Windows ``spawn`` start
    method) and derives each worker's ``random``/NumPy seed from the per-worker
    ``torch.initial_seed()`` that the DataLoader sets up. This makes augmentation
    randomness reproducible across ``num_workers`` values.
    """
    # torch sets a distinct initial_seed per worker derived from the loader's
    # base seed; fold it into the 32-bit range NumPy/random accept.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Return a CPU :class:`torch.Generator` seeded with ``seed``.

    Pass the result as the ``generator=`` argument to a DataLoader (and use it
    for the train/val split) so shuffling order is reproducible. Returns
    ``None`` when ``seed`` is ``None`` so callers fall back to global RNG.
    """
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
