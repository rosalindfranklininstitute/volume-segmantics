"""Tests for the optuna hyperparameter optimiser.


"""

from types import SimpleNamespace

import pytest

from volume_segmantics.model.operations.vol_seg_2d_trainer import (
    NumericalTrainingError,
)


def _call_hook(dice_history, trial):
    """Invoke the trainer's report/prune hook against a minimal self.

    Only ``self.epoch_history['seg_dice']`` is used by the hook, so a
    SimpleNamespace avoids constructing a full (GPU-bound) trainer. ``trial`` is
    a real ``optuna.Trial``.
    """
    from volume_segmantics.model.operations.vol_seg_2d_trainer import VolSeg2dTrainer
    fake_self = SimpleNamespace(epoch_history={"seg_dice": list(dice_history)})
    VolSeg2dTrainer._optuna_report_and_prune(fake_self, trial)


# --- always-on: core stays importable without optuna
def test_optimization_package_imports_without_optuna():
    import volume_segmantics.optimization as opt
    assert isinstance(opt.OPTUNA_AVAILABLE, bool)
    if not opt.OPTUNA_AVAILABLE:
        assert opt.OptunaOptimizer is None


# --- trainer hook
def test_nonfinite_dice_raises_numerical_error():
    optuna = pytest.importorskip("optuna")
    trial = optuna.create_study().ask()
    with pytest.raises(NumericalTrainingError):
        _call_hook([0.5, float("nan")], trial)


def test_hook_reports_monotonic_steps_across_phases():
    optuna = pytest.importorskip("optuna")
    # NopPruner so should_prune() never interferes with the reporting check.
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    trial = study.ask()
    # Phase 1 (frozen): 3 epochs accumulate into seg_dice.
    for hist in ([0.1], [0.1, 0.2], [0.1, 0.2, 0.3]):
        _call_hook(hist, trial)
    # Phase 2 (unfrozen): history keeps growing on the same trainer instance.
    for hist in ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5]):
        _call_hook(hist, trial)
    # Read the steps optuna actually recorded for this trial.
    reported = study.get_trials(deepcopy=False)[0].intermediate_values
    steps = sorted(reported.keys())
    assert steps == [0, 1, 2, 3, 4]          # strictly increasing across phases
    assert len(steps) == len(set(steps))     # no duplicates optuna would collapse


def test_hook_prunes_when_pruner_says_so():
    optuna = pytest.importorskip("optuna")
    # ThresholdPruner prunes deterministically once a reported value drops below
    # `lower`, so we can force the prune path through the real should_prune().
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.ThresholdPruner(lower=0.5),
    )
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned):
        _call_hook([0.1], trial)              # 0.1 < 0.5 threshold -> prune


def test_hook_does_not_prune_above_threshold():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.ThresholdPruner(lower=0.5),
    )
    trial = study.ask()
    _call_hook([0.9], trial)                   # 0.9 >= 0.5 -> no prune, no raise


#  config validation (needs optuna: optimizer module imports it) -----------
@pytest.fixture
def _make_optimizer(tmp_path):
    pytest.importorskip("optuna")
    import yaml
    from volume_segmantics.optimization.optuna_optimizer import OptunaOptimizer

    # A minimal base settings YAML with a model dict.
    settings = {
        "model": {"type": "U_Net", "encoder_name": "resnet34"},
        "starting_lr": 1e-4, "end_lr": 1e-3,
        "num_cyc_frozen": 2, "num_cyc_unfrozen": 1, "patience": 2,
        "eval_metric": "dice", "image_size": 64,
    }
    settings_path = tmp_path / "train_settings.yaml"
    settings_path.write_text(yaml.dump(settings))

    def _build(search_space, direction="maximize"):
        cfg = {"study_name": "t", "n_trials": 1, "direction": direction,
               "search_space": search_space}
        cfg_path = tmp_path / "optuna.yaml"
        cfg_path.write_text(yaml.dump(cfg))
        # Avoid pipeline-config resolution / real slicing: construct then poke.
        return OptunaOptimizer, str(settings_path), str(cfg_path), tmp_path

    return _build


def _try_construct(build, search_space, direction="maximize", monkeypatch=None):
    OptunaOptimizer, sp, cp, root = build(search_space, direction)
    # Stub pipeline-config resolution so we test validation in isolation.
    import volume_segmantics.optimization.optuna_optimizer as mod
    orig = OptunaOptimizer._resolve_pipeline_config
    OptunaOptimizer._resolve_pipeline_config = lambda self: None
    try:
        return OptunaOptimizer(["d.h5"], ["l.h5"], sp, cp, str(root))
    finally:
        OptunaOptimizer._resolve_pipeline_config = orig


def test_validate_rejects_unknown_param(_make_optimizer):
    with pytest.raises(ValueError, match="Unknown tunable param"):
        _try_construct(_make_optimizer, {"bogus_param": {"type": "uniform", "low": 0, "high": 1}})


def test_validate_rejects_image_size(_make_optimizer):
    with pytest.raises(ValueError, match="not tunable"):
        _try_construct(_make_optimizer, {"image_size": {"type": "int", "low": 32, "high": 128}})


def test_validate_rejects_minimize_direction(_make_optimizer):
    with pytest.raises(ValueError, match="maximised"):
        _try_construct(
            _make_optimizer,
            {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}},
            direction="minimize",
        )


def test_validate_rejects_degenerate_range(_make_optimizer):
    with pytest.raises(ValueError, match="must be < high"):
        _try_construct(
            _make_optimizer,
            {"starting_lr": {"type": "loguniform", "low": 1e-3, "high": 1e-5}},
        )


def test_validate_accepts_good_space(_make_optimizer):
    opt = _try_construct(
        _make_optimizer,
        {
            "starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            "encoder_name": {"type": "categorical", "choices": ["resnet34", "resnet50"]},
            "num_cyc_frozen": {"type": "int", "low": 2, "high": 6},
        },
    )
    assert opt.direction == "maximize"
    assert "starting_lr" in opt.search_space


def test_visualize_creates_plots(_make_optimizer):
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    opt = _try_construct(
        _make_optimizer,
        {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}},
    )
    study = optuna.create_study(direction="maximize")

    def _obj(t):
        lr = t.suggest_float("starting_lr", 1e-5, 1e-3, log=True)
        frozen = t.suggest_int("num_cyc_frozen", 2, 6)
        return lr * frozen

    study.optimize(_obj, n_trials=4)
    out = opt.visualize(study, output_dir="plots")
    assert (out / "optimization_history.png").exists()
    assert (out / "param_importances.png").exists()
    assert (out / "parallel_coordinate.png").exists()


# ---------------------------------------------------------------------------
# Ported (and adapted) from the original optuna test suite. The originals
# targeted the old API (_suggest_hyperparameter / _update_config_with_params /
# a dict `.config`); these exercise the equivalent hardened API
# (_suggest / _build_trial_settings / _save_best_config payload).
# ---------------------------------------------------------------------------

@pytest.fixture
def _trial():
    optuna = pytest.importorskip("optuna")
    return optuna.create_study().ask()


@pytest.fixture
def _completed_study():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"starting_lr": 1e-4},
            distributions={
                "starting_lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True)
            },
            value=0.87,
            user_attrs={"seed": 123},
        )
    )
    return study


# -- _suggest -----------------------------------------------------------------
def test_suggest_loguniform_in_range(_make_optimizer, _trial):
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    v = opt._suggest(_trial, "lr", {"type": "loguniform", "low": 1e-5, "high": 1e-2})
    assert 1e-5 <= v <= 1e-2


def test_suggest_int_in_range(_make_optimizer, _trial):
    opt = _try_construct(_make_optimizer, {"num_cyc_frozen": {"type": "int", "low": 1, "high": 10}})
    v = opt._suggest(_trial, "epochs", {"type": "int", "low": 1, "high": 10})
    assert 1 <= v <= 10


def test_suggest_categorical_valid_choice(_make_optimizer, _trial):
    choices = ["resnet18", "resnet34"]
    opt = _try_construct(_make_optimizer, {"encoder_name": {"type": "categorical", "choices": choices}})
    v = opt._suggest(_trial, "encoder_name", {"type": "categorical", "choices": choices})
    assert v in choices


def test_suggest_bad_type_raises(_make_optimizer, _trial):
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    with pytest.raises(ValueError):
        opt._suggest(_trial, "x", {"type": "magic", "low": 0, "high": 1})


# -- _build_trial_settings (deep-copied, isolated per trial) -------------------
def test_build_trial_settings_top_level(_make_optimizer):
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    s = opt._build_trial_settings({"starting_lr": 1e-4})
    assert s.starting_lr == pytest.approx(1e-4)


def test_build_trial_settings_nested_encoder(_make_optimizer):
    opt = _try_construct(_make_optimizer, {"encoder_name": {"type": "categorical", "choices": ["resnet34", "resnet18"]}})
    s = opt._build_trial_settings({"encoder_name": "resnet18"})
    assert s.model["encoder_name"] == "resnet18"


def test_build_trial_settings_model_type(_make_optimizer):
    opt = _try_construct(_make_optimizer, {"model_type": {"type": "categorical", "choices": ["U_Net", "U_Net_Plus_Plus"]}})
    s = opt._build_trial_settings({"model_type": "U_Net_Plus_Plus"})
    assert s.model["type"] == "U_Net_Plus_Plus"


def test_build_trial_settings_does_not_mutate_base(_make_optimizer):
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    base_before = opt.base_settings.starting_lr
    opt._build_trial_settings({"starting_lr": 9.9e-4})
    assert opt.base_settings.starting_lr == base_before  # isolation preserved


# -- _save_best_config --------------------------------------------------------
def test_save_best_config_creates_file(_make_optimizer, _completed_study):
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    opt._save_best_config(_completed_study)
    assert (opt.root_path / f"best_config_{opt.study_name}.yaml").is_file()


def test_save_best_config_contains_params_and_seed(_make_optimizer, _completed_study):
    import yaml
    opt = _try_construct(_make_optimizer, {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})
    opt._save_best_config(_completed_study)
    saved = yaml.safe_load((opt.root_path / f"best_config_{opt.study_name}.yaml").read_text())
    assert saved["best_params"]["starting_lr"] == pytest.approx(1e-4)
    assert saved["best_value"] == pytest.approx(0.87)
    assert saved["seed"] == 123


# ---------------------------------------------------------------------------
# Real-data tests (use the shared random-hdf5 fixtures + real training
# settings). Construction/validation is CPU-safe; a full study needs a GPU.
# ---------------------------------------------------------------------------

@pytest.fixture
def real_optimizer(tmp_path, training_settings_path, rand_int_hdf5_path, rand_label_hdf5_path):
    optuna = pytest.importorskip("optuna")
    import yaml
    from volume_segmantics.optimization import OptunaOptimizer

    cfg = {
        "study_name": "itest", "n_trials": 2, "seed": 1, "direction": "maximize",
        "n_startup_trials": 1, "n_warmup_steps": 1, "min_free_gb": 0.0,
        "search_space": {"starting_lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}},
    }
    cfg_path = tmp_path / "opt_itest.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    opt = OptunaOptimizer(
        data_paths=[str(rand_int_hdf5_path)],
        label_paths=[str(rand_label_hdf5_path)],
        settings_path=str(training_settings_path),
        optuna_config=str(cfg_path),
        root_path=str(tmp_path),
    )
    # Keep any GPU study fast: 1 frozen epoch, no unfrozen phase.
    opt.base_settings.num_cyc_frozen = 1
    opt.base_settings.num_cyc_unfrozen = 0
    return opt


def test_optimizer_init_real_settings(real_optimizer, tmp_path):
    # Exercises the real _resolve_pipeline_config path on CPU.
    from volume_segmantics.optimization import OptunaOptimizer
    assert isinstance(real_optimizer, OptunaOptimizer)
    assert real_optimizer.root_path == tmp_path
    assert isinstance(real_optimizer.search_space, dict)
    assert real_optimizer.direction == "maximize"


@pytest.mark.gpu
@pytest.mark.slow
def test_optimize_returns_study(real_optimizer):
    optuna = pytest.importorskip("optuna")
    study = real_optimizer.optimize()
    assert isinstance(study, optuna.Study)


@pytest.mark.gpu
@pytest.mark.slow
def test_optimize_best_value_in_range(real_optimizer):
    study = real_optimizer.optimize()
    assert 0.0 <= study.best_value <= 1.0


@pytest.mark.gpu
@pytest.mark.slow
def test_optimize_saves_best_config(real_optimizer, tmp_path):
    real_optimizer.optimize()
    assert (tmp_path / f"best_config_{real_optimizer.study_name}.yaml").is_file()
