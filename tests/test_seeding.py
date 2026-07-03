"""Tests for the reproducibility helpers.

"""

import random

import numpy as np
import pytest
import torch

from volume_segmantics.utilities.seeding import (
    make_generator,
    seed_worker,
    set_seed,
)


def _draw():
    """Draw from every RNG set_seed is responsible for."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4).tolist(),
    )


def test_set_seed_makes_draws_reproducible():
    set_seed(1234)
    first = _draw()
    set_seed(1234)
    second = _draw()
    assert first == second


def test_different_seeds_differ():
    set_seed(1)
    a = _draw()
    set_seed(2)
    b = _draw()
    assert a != b


def test_set_seed_none_is_noop_and_returns_none():
    # None must not seed anything: two consecutive None calls leave the RNG
    # advancing (i.e. draws differ), preserving current non-deterministic flow.
    assert set_seed(None) is None
    a = _draw()
    assert set_seed(None) is None
    b = _draw()
    assert a != b


def test_set_seed_returns_applied_seed():
    assert set_seed(7) == 7
    # Accepts seed-like values and normalises to int.
    assert set_seed("7") == 7


def test_make_generator_is_reproducible_and_independent():
    g1 = make_generator(99)
    g2 = make_generator(99)
    # Same seed -> identical shuffle order; generator is independent of global RNG.
    assert torch.randperm(10, generator=g1).tolist() == torch.randperm(
        10, generator=g2
    ).tolist()


def test_make_generator_none_returns_none():
    assert make_generator(None) is None


def test_seed_worker_is_deterministic_per_initial_seed():
    # Two workers that share an initial_seed must seed identically. We emulate
    # the DataLoader contract by pinning torch.initial_seed via manual_seed.
    torch.manual_seed(555)
    seed_worker(0)
    a = (random.random(), float(np.random.rand()))
    torch.manual_seed(555)
    seed_worker(0)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_seed_worker_is_picklable():
    # Must survive the Windows 'spawn' start method -> picklable (module-level).
    import pickle

    restored = pickle.loads(pickle.dumps(seed_worker))
    assert restored is seed_worker


def test_deterministic_flag_sets_cudnn_state():
    set_seed(3, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # Reset to library defaults so we don't leak deterministic mode into other
    # tests (use_deterministic_algorithms can slow/restrict unrelated ops).
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False


@pytest.mark.parametrize("seed", [0, 1, 42, 2**31 - 1])
def test_a_range_of_seeds_round_trip(seed):
    set_seed(seed)
    first = _draw()
    set_seed(seed)
    assert first == _draw()
