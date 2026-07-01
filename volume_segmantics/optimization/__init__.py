try:
    from .optuna_optimizer import OptunaOptimizer
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None

__all__ = ['OptunaOptimizer', 'OPTUNA_AVAILABLE']

# Version info
__version__ = '1.0.0'
