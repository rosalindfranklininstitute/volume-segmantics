"""Optuna hyperparameter optimisation (optional).

Importing this package never hard-fails when optuna is missing: the core
install works without ``[optuna]``. ``OPTUNA_AVAILABLE`` reports whether
the optimiser could be imported.
"""

try:
    from .optuna_optimizer import OptunaOptimizer
    OPTUNA_AVAILABLE = True
except ImportError:  # optuna (or the extra) not installed
    OptunaOptimizer = None
    OPTUNA_AVAILABLE = False

__all__ = ["OptunaOptimizer", "OPTUNA_AVAILABLE"]
