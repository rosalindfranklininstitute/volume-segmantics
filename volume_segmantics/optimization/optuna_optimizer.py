"""Optuna hyperparameter optimisation for the legacy ``VolSeg2dTrainer``.

"""

from __future__ import annotations

import copy
import gc
import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from volume_segmantics.data import TrainingDataSlicer, get_settings_data
from volume_segmantics.model import VolSeg2dTrainer
from volume_segmantics.model.operations.vol_seg_2d_trainer import (
    NumericalTrainingError,
)
from volume_segmantics.utilities.seeding import set_seed

logger = logging.getLogger(__name__)

# --- Search-space allow-list -------------------------------------------------.
_ALLOWED_TOP_LEVEL = {
    "starting_lr", "end_lr", "loss_criterion", "num_cyc_frozen",
    "num_cyc_unfrozen", "patience", "ce_weight", "dice_weight", "pct_lr_inc",
    "encoder_lr_multiplier", "augmentation_library", "training_set_proportion",
}
# param name -> key inside the settings.model dict
_ALLOWED_MODEL = {
    "model_type": "type",
    "encoder_name": "encoder_name",
    "encoder_depth": "encoder_depth",
}
_FORBIDDEN = {
    "image_size": "slice-affecting; this variant slices once and reuses "
                  "(searching image_size would require per-trial re-slicing).",
}


class OptunaOptimizer:
    """Hyperparameter optimiser for the legacy trainer (ephemeral study)."""

    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        settings_path: str,
        optuna_config: str,
        root_path: Optional[str] = None,
    ):
        self.data_paths = list(data_paths)
        self.label_paths = list(label_paths)
        self.root_path = Path(root_path) if root_path else Path.cwd()

        if not self.label_paths or len(self.label_paths) != len(self.data_paths):
            raise ValueError(
                "OptunaOptimizer needs one label volume per data volume; got "
                f"{len(self.data_paths)} data and {len(self.label_paths)} label paths."
            )

        # Base settings, built the same way main() builds them for the legacy
        # path, so the trainer sees an identical settings object.
        self.base_settings = get_settings_data(Path(settings_path))
        self.base_settings.use_legacy_trainer = True
        self._resolve_pipeline_config()

        with open(optuna_config, "r") as f:
            self.optuna_config = yaml.safe_load(f) or {}

        self.search_space: Dict[str, Any] = self.optuna_config.get("search_space", {})
        self.study_name = self.optuna_config.get("study_name", "optuna_study")
        self.direction = self.optuna_config.get("direction", "maximize")
        self.base_seed = int(self.optuna_config.get("seed", 42))

        self._validate_config()

        # Ephemeral, study-owned working root for slices + per-trial models.
        self.work_dir = self.root_path / "tmp" / f"optuna_{self.study_name}"
        self.slice_data_dir = self.work_dir / "data"
        self.slice_seg_dir = self.work_dir / "seg"
        self.max_label_no = 0
        self._sliced = False

        logger.info(
            "OptunaOptimizer ready: %d tunable params, direction=%s, work_dir=%s",
            len(self.search_space), self.direction, self.work_dir,
        )

    # -- setup / validation ---------------------------------------------------
    def _resolve_pipeline_config(self) -> None:
        """Attach a pipeline config to the base settings (legacy dataloaders
        read from it), mirroring train_2d_model.main()."""
        from volume_segmantics.scripts.train_2d_model import _resolve_pipeline_config
        self.base_settings.pipeline_config = _resolve_pipeline_config(
            self.base_settings, self.root_path
        )

    def _validate_config(self) -> None:
        """Reject unknown/forbidden search-space keys, degenerate ranges, and a
        metric-direction mismatch — fail loud before doing any expensive work."""
        # Objective is validation dice → must be maximised.
        if self.direction != "maximize":
            raise ValueError(
                "This variant optimises validation dice, which must be "
                f"maximised; got direction={self.direction!r}. Set "
                "direction: maximize in the optuna config."
            )
        if not self.search_space:
            raise ValueError("optuna config has an empty search_space.")

        for name, spec in self.search_space.items():
            if name in _FORBIDDEN:
                raise ValueError(f"Param {name!r} is not tunable: {_FORBIDDEN[name]}")
            if name not in _ALLOWED_TOP_LEVEL and name not in _ALLOWED_MODEL:
                allowed = sorted(_ALLOWED_TOP_LEVEL | set(_ALLOWED_MODEL))
                raise ValueError(
                    f"Unknown tunable param {name!r}. Allowed: {allowed}"
                )
            self._validate_param_spec(name, spec)

    @staticmethod
    def _validate_param_spec(name: str, spec: Dict[str, Any]) -> None:
        ptype = spec.get("type")
        if ptype in ("loguniform", "uniform"):
            low, high = float(spec["low"]), float(spec["high"])
            if not (low < high):
                raise ValueError(f"{name}: low ({low}) must be < high ({high}).")
            if ptype == "loguniform" and low <= 0:
                raise ValueError(f"{name}: loguniform requires low > 0 (got {low}).")
        elif ptype == "int":
            low, high = int(spec["low"]), int(spec["high"])
            if not (low < high):
                raise ValueError(f"{name}: low ({low}) must be < high ({high}).")
        elif ptype == "categorical":
            choices = spec.get("choices") or []
            if len(choices) < 2:
                raise ValueError(f"{name}: categorical needs >= 2 choices.")
        else:
            raise ValueError(f"{name}: unknown param type {ptype!r}.")

    # -- slicing (once) -------------------------------------------------------
    def _slice_once(self) -> None:
        """Slice every volume a single time into the study-owned dir and reuse
        across all trials (slicing is invariant to the tuned params)."""
        if self._sliced:
            return
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.slice_data_dir.mkdir(parents=True, exist_ok=True)
        self.slice_seg_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Slicing %d volume(s) once for the study...", len(self.data_paths))
        max_label_no = 0
        for i, (data_path, label_path) in enumerate(
            zip(self.data_paths, self.label_paths)
        ):
            slicer = TrainingDataSlicer(data_path, label_path, self.base_settings)
            slicer.output_data_slices(self.slice_data_dir, f"data{i}")
            slicer.output_label_slices(self.slice_seg_dir, f"seg{i}")
            max_label_no = max(max_label_no, slicer.num_seg_classes)
        if max_label_no < 2:
            raise ValueError(
                f"Slicing produced max_label_no={max_label_no}; need >= 2 classes."
            )
        self.max_label_no = max_label_no
        self._sliced = True
        logger.info("Slicing complete: max_label_no=%d", self.max_label_no)

    # -- per-trial config -----------------------------------------------------
    def _suggest(self, trial: "optuna.Trial", name: str, spec: Dict[str, Any]) -> Any:
        ptype = spec["type"]
        if ptype == "loguniform":
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        if ptype == "uniform":
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        if ptype == "int":
            return trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        if ptype == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"Unknown param type {ptype!r}")  # unreachable post-validation

    def _build_trial_settings(self, params: Dict[str, Any]):
        """Deep-copy the immutable base settings and apply suggestions only to
        the copy, so no param or model field leaks between trials."""
        settings = copy.deepcopy(self.base_settings)
        if not isinstance(getattr(settings, "model", None), dict):
            raise ValueError("base settings has no 'model' dict to tune.")
        for name, value in params.items():
            if name in _ALLOWED_MODEL:
                settings.model[_ALLOWED_MODEL[name]] = value
            else:  # allow-list guarantees top-level
                setattr(settings, name, value)
        return settings

    # -- objective ------------------------------------------------------------
    def objective(self, trial: "optuna.Trial") -> float:
        params = {n: self._suggest(trial, n, s) for n, s in self.search_space.items()}
        logger.info("Trial %d params: %s", trial.number, params)

        # Reproducible per-trial seed, recorded so a trial can be replayed from
        # its stored seed (not from trial.number, which shifts if ordering does).
        seed = self.base_seed + trial.number
        trial.set_user_attr("seed", seed)
        set_seed(seed)

        settings = self._build_trial_settings(params)
        num_cyc_frozen = int(getattr(settings, "num_cyc_frozen", 0) or 0)
        num_cyc_unfrozen = int(getattr(settings, "num_cyc_unfrozen", 0) or 0)
        patience = int(getattr(settings, "patience", 4))
        if num_cyc_frozen <= 0 and num_cyc_unfrozen <= 0:
            raise ValueError(
                f"Trial {trial.number}: num_cyc_frozen + num_cyc_unfrozen must be "
                "> 0 (a trial that trains zero epochs is a config error)."
            )

        import volume_segmantics.utilities.base_data_utils as bd_utils
        model_type_name = (
            bd_utils.get_model_type(settings).name
            if isinstance(settings.model["type"], str)
            else settings.model["type"].name
        )
        model_out = self.work_dir / (
            f"{date.today()}_{model_type_name}_trial_{trial.number}.pytorch"
        )

        trainer = None
        try:
            trainer = VolSeg2dTrainer(
                self.slice_data_dir, self.slice_seg_dir, self.max_label_no, settings
            )
            best = 0.0
            if num_cyc_frozen > 0:
                best = trainer.train_model(
                    model_out, num_cyc_frozen, patience,
                    create=True, frozen=True, trial=trial,
                )
            if num_cyc_unfrozen > 0:
                best = trainer.train_model(
                    model_out, num_cyc_unfrozen, patience,
                    create=(num_cyc_frozen == 0), frozen=False, trial=trial,
                )
            logger.info("Trial %d done: dice=%.4f", trial.number, best)
            return float(best)
        finally:
            # finally-based teardown: drop the trainer/model and reclaim VRAM
            # even on prune/OOM/exception (gc + empty_cache alone are not enough).
            del trainer
            self._free_gpu()
            # Search uses a disable-then-retrain-best policy: the winning
            # *hyperparameters* (saved as best_config YAML) are what matter, so
            # per-trial model weights are throwaway. Delete them to keep disk
            # bounded on a chronically-full volume.
            try:
                if model_out.exists():
                    model_out.unlink()
            except OSError:
                pass

    @staticmethod
    def _free_gpu() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # torch always present here, but never fail teardown
            pass

    # -- run ------------------------------------------------------------------
    def optimize(self) -> "optuna.Study":
        self._disk_preflight()
        self._slice_once()

        n_trials = int(self.optuna_config.get("n_trials", 50))
        timeout = self.optuna_config.get("timeout", None)
        n_startup = int(self.optuna_config.get("n_startup_trials", 5))
        n_warmup = int(self.optuna_config.get("n_warmup_steps", 3))
        # Ephemeral, in-process, single-worker: no persistent storage/resume.
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=TPESampler(seed=self.base_seed),
            pruner=MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup),
        )
        logger.info("Starting study %s: %d trials, direction=%s",
                    self.study_name, n_trials, self.direction)

        import torch
        # NaN → NumericalTrainingError, OOM → cuda OOM: mark that trial FAILED
        # and continue; any other error propagates and stops the study loudly.
        catch = (NumericalTrainingError, torch.cuda.OutOfMemoryError)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, catch=catch)

        if study.best_trial is not None:
            logger.info("Best trial %d: dice=%.4f params=%s",
                        study.best_trial.number, study.best_value, study.best_params)
            self._save_best_config(study)
            if self.optuna_config.get("visualize", False):
                self.visualize(study)
        return study

    def _disk_preflight(self) -> None:
        """Refuse to start if free space is below a reserve — this host's C:
        drive is chronically full, and an unbounded study would wedge it."""
        reserve_gb = float(self.optuna_config.get("min_free_gb", 5.0))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.work_dir).free / 1e9
        logger.info("Disk preflight: %.1f GB free at %s (reserve %.1f GB)",
                    free_gb, self.work_dir, reserve_gb)
        if free_gb < reserve_gb:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free at {self.work_dir}; need >= "
                f"{reserve_gb:.1f} GB. Free space or set a --study-root / "
                "min_free_gb on a bounded volume."
            )

    def _save_best_config(self, study: "optuna.Study") -> None:
        """Write the winning hyperparameters (plain, YAML-safe) so the final
        model can be retrained with them. Not the full settings object, which may
        hold non-serialisable fields (e.g. a resolved pipeline_config)."""
        payload = {
            "study_name": self.study_name,
            "best_value": float(study.best_value),
            "best_trial": study.best_trial.number,
            "best_params": dict(study.best_params),
            "seed": study.best_trial.user_attrs.get("seed"),
        }
        out = self.root_path / f"best_config_{self.study_name}.yaml"
        with open(out, "w") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
        logger.info("Best params written to %s", out)

    def visualize(self, study: "optuna.Study", output_dir: str = "optuna_plots") -> Path:
        """Save study plots as PNGs under ``root_path/output_dir``.

        Produces optimization_history.png, param_importances.png (optuna's
        matplotlib backend) and a custom, readable parallel_coordinate.png.
        matplotlib is a core dependency; pandas (for the parallel-coordinate
        plot) ships with the ``[optuna]`` extra. Each plot is guarded so one
        failing plot never loses the others or the study result.
        """
        output_path = self.root_path / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history, plot_param_importances,
            )
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            logger.warning("Skipping optuna plots (matplotlib unavailable): %s", e)
            return output_path

        for name, plot_fn in (
            ("optimization_history", plot_optimization_history),
            ("param_importances", plot_param_importances),
        ):
            try:
                ax = plot_fn(study)
                fig = ax.figure
                fig.savefig(output_path / f"{name}.png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                logger.info("Saved %s.png", name)
            except Exception as e:  # a single plot must not sink the rest
                logger.warning("Could not generate %s plot: %s", name, e)

        try:
            from volume_segmantics.optimization.plot_parallel_coordinates import (
                plot_parallel_coordinates,
            )
            plot_parallel_coordinates(study, output_path / "parallel_coordinate.png")
            logger.info("Saved parallel_coordinate.png")
        except Exception as e:  # e.g. pandas missing, or <1 completed trial
            logger.warning("Could not generate parallel_coordinate plot: %s", e)

        logger.info("Visualisation plots saved to: %s", output_path)
        return output_path

    def cleanup(self) -> None:
        """Remove the ephemeral study working dir (slices + trial models)."""
        shutil.rmtree(self.work_dir, ignore_errors=True)
