import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from datetime import date

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from volume_segmantics.data import TrainingDataSlicer, get_settings_data
from volume_segmantics.model import VolSeg2dTrainer

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna.
    
    Integrates Optuna Bayesian optimization with the Volume Segmantics
    training pipeline to automatically find optimal hyperparameters.
    All study settings are read from the optuna config YAML file.
    """

    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        base_config: str,
        optuna_config: str,
        root_path: Optional[str] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            data_paths: List of paths to training data volumes
            label_paths: List of paths to label volumes
            base_config: Path to base training configuration YAML
                         (the volseg-settings/2d_model_train_settings.yaml)
            optuna_config: Path to Optuna search configuration YAML
            root_path: Path to project root directory. Trial models and
                       best config will be saved here. Defaults to cwd.

        Raises:
            ImportError: If Optuna is not installed
        """
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.base_config = Path(base_config)
        self.root_path = Path(root_path) if root_path else Path.cwd()

        # Load the raw base config dict for parameter overriding per trial
        with open(self.base_config, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load optuna config  this is the single source of truth for all
        # study settings (n_trials, direction, storage, search_space, etc.)
        with open(optuna_config, 'r') as f:
            self.optuna_config = yaml.safe_load(f)

        self.search_space = self.optuna_config.get('search_space', {})

        # Study identity
        self.study_name = self.optuna_config.get('study_name', 'optuna_study')
        self.storage    = self.optuna_config.get('storage', None)

        logger.info(
            f"Initialized optimizer with {len(self.search_space)} "
            f"hyperparameters in search space"
        )

    def _suggest_hyperparameter(
        self,
        trial: 'optuna.Trial',
        param_name: str,
        param_config: Dict[str, Any],
    ) -> Any:
        """Suggest a hyperparameter value based on its config block."""
        param_type = param_config['type']

        if param_type == 'loguniform':
            return trial.suggest_float(
                param_name,
                float(param_config['low']),
                float(param_config['high']),
                log=True,
            )
        elif param_type == 'uniform':
            return trial.suggest_float(
                param_name,
                float(param_config['low']),
                float(param_config['high']),
            )
        elif param_type == 'int':
            return trial.suggest_int(
                param_name,
                int(param_config['low']),
                int(param_config['high']),
                step=int(param_config.get('step', 1)),
            )
        elif param_type == 'categorical':
            return trial.suggest_categorical(
                param_name,
                param_config['choices'],
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _update_config_with_params(
        self,
        config: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update a config dict with suggested trial parameters."""
        for param_name, value in params.items():
            # Nested model parameters
            if param_name == 'encoder_name':
                config['model']['encoder_name'] = value
            elif param_name == 'encoder_depth':
                config['model']['encoder_depth'] = value
            # Known top-level parameters
            elif param_name in [
                'starting_lr', 'end_lr', 'loss_criterion',
                'num_cyc_frozen', 'num_cyc_unfrozen', 'patience',
                'ce_weight', 'dice_weight', 'pct_lr_inc',
                'encoder_lr_multiplier', 'augmentation_library',
                'image_size', 'training_set_proportion',
            ]:
                config[param_name] = value
            else:
                config[param_name] = value

        return config

    def _train_trial(
        self,
        trial_config: Dict[str, Any],
        trial: 'optuna.Trial',
    ) -> float:
        """
        Slice volumes, train a model with the trial config, and return
        the best validation metric achieved.

        Args:
            trial_config: Full config dict with trial hyperparameters applied
            trial: Optuna trial object (used for pruning)

        Returns:
            Best validation metric (higher is better for Dice)
        """
        settings = type('Settings', (), trial_config)()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_im_dir = temp_path / trial_config.get('data_im_dirname', 'data')
            seg_im_dir  = temp_path / trial_config.get('seg_im_out_dirname', 'seg')

            # Slice all volumes for this trial
            logger.info(f"Trial {trial.number}: slicing volumes...")
            max_label_no = 0

            for count, (data_path, label_path) in enumerate(
                zip(self.data_paths, self.label_paths)
            ):
                slicer = TrainingDataSlicer(data_path, label_path, settings)
                slicer.output_data_slices(data_im_dir, f"data{count}")
                slicer.output_label_slices(seg_im_dir, f"seg{count}")
                if slicer.num_seg_classes > max_label_no:
                    max_label_no = slicer.num_seg_classes

            logger.info(
                f"Trial {trial.number}: training with {max_label_no} classes..."
            )
            trainer = VolSeg2dTrainer(
                data_im_dir, seg_im_dir, max_label_no, settings
            )

            # Save trial model to root_path so it persists if needed
            import volume_segmantics.utilities.base_data_utils as utils
            if isinstance(settings.model["type"], str):
                model_type_enum = utils.get_model_type(settings)
                model_type_name = model_type_enum.name
            else:
                model_type_name = settings.model["type"].name

            
            model_out = self.root_path / f"{date.today()}_{model_type_name}_trial_{trial.number}_model.pytorch"

            num_cyc_frozen   = trial_config.get('num_cyc_frozen', 0)
            num_cyc_unfrozen = trial_config.get('num_cyc_unfrozen', 0)
            patience = trial_config.get('patience', 4)
            results = None

            if num_cyc_frozen > 0:
                logger.debug(
                    f"Trial {trial.number}: frozen training "
                    f"({num_cyc_frozen} epochs)"
                )
                results = trainer.train_model(
                    model_out, num_cyc_frozen, patience,
                    create=True, frozen=True, trial=trial,
                )

            if num_cyc_unfrozen > 0:
                logger.debug(
                    f"Trial {trial.number}: unfrozen training "
                    f"({num_cyc_unfrozen} epochs)"
                )
                results = trainer.train_model(
                    model_out, num_cyc_unfrozen, patience,
                    create=(num_cyc_frozen == 0), frozen=False, trial=trial,
                )

            if results is None:
                logger.warning(
                    f"Trial {trial.number}: no training cycles completed"
                )
                return 0.0

            return results.get('best_val_metric', 0.0)

    def objective(self, trial: 'optuna.Trial') -> float:
        """
        Optuna objective function suggests params, trains, returns metric.

        Args:
            trial: Optuna trial

        Returns:
            Validation metric value for this trial
        """
        # Suggest all hyperparameters defined in search_space
        suggested_params = {}
        for param_name, param_config in self.search_space.items():
            suggested_params[param_name] = self._suggest_hyperparameter(
                trial, param_name, param_config
            )

        logger.info(f"Trial {trial.number}: {suggested_params}")

        # Apply suggested params on top of the base config
        trial_config = self.config.copy()
        trial_config = self._update_config_with_params(
            trial_config, suggested_params
        )

        try:
            metric = self._train_trial(trial_config, trial)
            logger.info(f"Trial {trial.number} completed: {metric:.4f}")
            return metric

        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise

    def optimize(self) -> 'optuna.Study':
        """
        Run hyperparameter optimization.
        All settings (n_trials, direction, timeout, etc.) are read from
        the optuna config YAML no arguments needed.

        Returns:
            Completed Optuna study
        """
        n_trials  = self.optuna_config.get('n_trials', 50)
        direction = self.optuna_config.get('direction', 'maximize')
        timeout   = self.optuna_config.get('timeout', None)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=TPESampler(seed=self.optuna_config.get('seed', 42)),
            pruner=MedianPruner(
                n_startup_trials=self.optuna_config.get('n_startup_trials', 5),
                n_warmup_steps=self.optuna_config.get('n_warmup_steps', 3),
            ),
            load_if_exists=True,
        )

        logger.info("="*60)
        logger.info(f"Starting optimization: {self.study_name}")
        logger.info(f"Trials: {n_trials}  |  Direction: {direction}")
        if timeout:
            logger.info(f"Timeout: {timeout}s")
        logger.info("="*60)

        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        logger.info("="*60)
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info("Best parameters:")
        for param, value in study.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("="*60)

        self._save_best_config(study)
        return study

    def _save_best_config(self, study: 'optuna.Study'):
        """Save the best trial's config as a YAML file in root_path."""
        best_config = self.config.copy()
        best_config = self._update_config_with_params(
            best_config, study.best_params
        )

        output_path = self.root_path / f"best_config_{self.study_name}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(best_config, f)

        logger.info(f"Best config saved to: {output_path}")

    def visualize(self, study: 'optuna.Study', output_dir: str = "optuna_plots"):
        """
        Generate and save visualization plots as PNG images.
        Args:
            study: Completed Optuna study
            output_dir: Directory to save plots (relative to root_path)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend, no display needed
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
        except ImportError:
            logger.warning(
                "Matplotlib not installed - skipping visualization. "
                "Install with: pip install matplotlib"
            )
            return

        output_path = self.root_path / output_dir
        output_path.mkdir(exist_ok=True)

        plots = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "parallel_coordinate": plot_parallel_coordinate,
        }

        for name, plot_fn in plots.items():
            try:
                ax = plot_fn(study)
                fig = ax.figure
                fig.savefig(output_path / f"{name}.png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                logger.info(f"Saved {name}.png")
            except Exception as e:
                logger.warning(f"Could not generate {name} plot: {e}")

        logger.info(f"Visualization plots saved to: {output_path}")