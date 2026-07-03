
from __future__ import annotations

from typing import Any, Callable, Dict, List


#  Registries 
# Module-level dicts. Keys are canonical names; values are the callable
# (class for heads, factory function for everything else). 

_HEADS: Dict[str, type] = {}
_TARGET_GENERATORS: Dict[str, Callable[..., Any]] = {}
_LOSSES: Dict[str, Callable[..., Any]] = {}
_TRANSFORMS: Dict[str, Callable[..., Any]] = {}

_BOOTSTRAPPED: bool = False


def _bootstrap_registrations() -> None:
    """One-shot import of the modules that self-register heads/losses/targets.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    try:
        import volume_segmantics.model.heads  # noqa: F401 — registers heads.
        import volume_segmantics.model.loss_registry  # noqa: F401 — losses.
        import volume_segmantics.data.targets  # noqa: F401 — target generators.
    except ImportError:
        pass


#  Heads 


def register_head(name: str, cls: type) -> None:
    """Register a :class:`PredictionHead` subclass.

    Raises :class:`KeyError` on duplicate registration so import-order
    accidents surface rather than silently shadowing.
    """
    if name in _HEADS:
        raise KeyError(
            f"head {name!r} already registered "
            f"(existing: {_HEADS[name].__name__})"
        )
    _HEADS[name] = cls


def build_head(name: str, **kwargs: Any) -> Any:
    """Instantiate the head registered under ``name``.

    Raises :class:`KeyError` listing known head names if ``name`` is
    not registered.
    """
    _bootstrap_registrations()
    if name not in _HEADS:
        raise KeyError(
            f"unknown head {name!r}; known: {sorted(_HEADS)}"
        )
    return _HEADS[name](**kwargs)


def list_heads() -> List[str]:
    """Return registered head names in alphabetical order."""
    _bootstrap_registrations()
    return sorted(_HEADS)


#  Target generators 


def register_target_generator(name: str, factory: Callable[..., Any]) -> None:
    """Register a target-generator factory under ``name``.

    Per the design, ``name`` is normally a head name — the generator
    produces that head's target tensor from a label volume slice.
    """
    if name in _TARGET_GENERATORS:
        raise KeyError(
            f"target generator {name!r} already registered"
        )
    _TARGET_GENERATORS[name] = factory


def build_target_generator(name: str, **kwargs: Any) -> Any:
    _bootstrap_registrations()
    if name not in _TARGET_GENERATORS:
        raise KeyError(
            f"unknown target generator {name!r}; "
            f"known: {sorted(_TARGET_GENERATORS)}"
        )
    return _TARGET_GENERATORS[name](**kwargs)


def list_target_generators() -> List[str]:
    _bootstrap_registrations()
    return sorted(_TARGET_GENERATORS)


#  Losses 


def register_loss(name: str, factory: Callable[..., Any]) -> None:
    """Register a loss factory under ``name``.

    Factories return a callable with the uniform
    ``(pred, target, **extra) -> Tensor`` signature; per-head extras
    (``pos_weight``, ``ignore_index``, etc.) flow via ``**extra``.
    """
    if name in _LOSSES:
        raise KeyError(f"loss {name!r} already registered")
    _LOSSES[name] = factory


def build_loss(name: str, **kwargs: Any) -> Any:
    _bootstrap_registrations()
    if name not in _LOSSES:
        raise KeyError(
            f"unknown loss {name!r}; known: {sorted(_LOSSES)}"
        )
    return _LOSSES[name](**kwargs)


def list_losses() -> List[str]:
    _bootstrap_registrations()
    return sorted(_LOSSES)


#  Transforms 


def register_transform(name: str, factory: Callable[..., Any]) -> None:
    """Register an augmentation transform factory under ``name``."""
    if name in _TRANSFORMS:
        raise KeyError(f"transform {name!r} already registered")
    _TRANSFORMS[name] = factory


def build_transform(name: str, **kwargs: Any) -> Any:
    if name not in _TRANSFORMS:
        raise KeyError(
            f"unknown transform {name!r}; "
            f"known: {sorted(_TRANSFORMS)}"
        )
    return _TRANSFORMS[name](**kwargs)


def list_transforms() -> List[str]:
    return sorted(_TRANSFORMS)


#  Test-only helper 



def _clear_all_registries_for_tests() -> None:
    """Empty every registry. **Tests only.**"""
    _HEADS.clear()
    _TARGET_GENERATORS.clear()
    _LOSSES.clear()
    _TRANSFORMS.clear()


__all__ = [
    "build_head",
    "build_loss",
    "build_target_generator",
    "build_transform",
    "list_heads",
    "list_losses",
    "list_target_generators",
    "list_transforms",
    "register_head",
    "register_loss",
    "register_target_generator",
    "register_transform",
]
