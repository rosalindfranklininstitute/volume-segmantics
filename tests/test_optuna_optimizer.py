from volume_segmantics.optimization import OptunaOptimizer
import pytest
import yaml
import optuna
from pathlib import Path

import volume_segmantics.utilities.config as cfg


@pytest.fixture()
def optuna_config_path(cwd):
    return Path(cwd.parent, "volseg-settings", "optuna_config.yaml")


@pytest.fixture()
def optuna_optimizer(tmp_path, training_settings_path, optuna_config_path,
                     rand_int_hdf5_path, rand_label_hdf5_path):
    return OptunaOptimizer(
        data_paths=[str(rand_int_hdf5_path)],
        label_paths=[str(rand_label_hdf5_path)],
        base_config=str(training_settings_path),
        optuna_config=str(optuna_config_path),
        root_path=str(tmp_path),
    )


@pytest.fixture()
def trial():
    study = optuna.create_study()
    return study.ask()


@pytest.fixture()
def completed_study(optuna_optimizer):
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"starting_lr": 1e-4},
            distributions={"starting_lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True)},
            value=0.87,
        )
    )
    return study

class TestOptunaOptimizer:

    def test_optimizer_init(self, optuna_optimizer):
        assert isinstance(optuna_optimizer, OptunaOptimizer)
        assert isinstance(optuna_optimizer.study_name, str)
        assert isinstance(optuna_optimizer.search_space, dict)
        assert isinstance(optuna_optimizer.config, dict)

    def test_optimizer_root_path_set(self, optuna_optimizer, tmp_path):
        assert optuna_optimizer.root_path == tmp_path

    # _suggest_hyperparameter 

    @pytest.mark.parametrize("param_name,param_config", [
        ("lr",      {"type": "loguniform", "low": 1e-5, "high": 1e-2}),
        ("dropout", {"type": "uniform",    "low": 0.0,  "high": 0.5}),
        ("epochs",  {"type": "int",        "low": 1,    "high": 10}),
        ("encoder_name", {"type": "categorical", "choices": ["resnet18", "resnet34"]}),
    ])
    def test_suggest_hyperparameter(self, optuna_optimizer, trial, param_name, param_config):
        value = optuna_optimizer._suggest_hyperparameter(trial, param_name, param_config)
        assert value is not None

    def test_suggest_hyperparameter_loguniform_in_range(self, optuna_optimizer, trial):
        value = optuna_optimizer._suggest_hyperparameter(
            trial, "lr", {"type": "loguniform", "low": 1e-5, "high": 1e-2}
        )
        assert 1e-5 <= value <= 1e-2

    def test_suggest_hyperparameter_int_in_range(self, optuna_optimizer, trial):
        value = optuna_optimizer._suggest_hyperparameter(
            trial, "epochs", {"type": "int", "low": 1, "high": 10}
        )
        assert 1 <= value <= 10

    def test_suggest_hyperparameter_categorical_valid_choice(self, optuna_optimizer, trial):
        choices = ["resnet18", "resnet34"]
        value = optuna_optimizer._suggest_hyperparameter(
            trial, "encoder_name", {"type": "categorical", "choices": choices}
        )
        assert value in choices

    def test_suggest_hyperparameter_bad_type(self, optuna_optimizer, trial):
        with pytest.raises(ValueError):
            optuna_optimizer._suggest_hyperparameter(
                trial, "x", {"type": "magic", "low": 0, "high": 1}
            )

    # _update_config_with_params 

    def test_update_config_top_level_param(self, optuna_optimizer):
        config = {"starting_lr": 1e-3, "model": {"encoder_name": "resnet34"}}
        updated = optuna_optimizer._update_config_with_params(config, {"starting_lr": 1e-4})
        assert updated["starting_lr"] == pytest.approx(1e-4)

    def test_update_config_encoder_name_nested(self, optuna_optimizer):
        config = {"model": {"encoder_name": "resnet34", "encoder_depth": 5}}
        updated = optuna_optimizer._update_config_with_params(config, {"encoder_name": "resnet18"})
        assert updated["model"]["encoder_name"] == "resnet18"

    def test_update_config_encoder_depth_nested(self, optuna_optimizer):
        config = {"model": {"encoder_name": "resnet34", "encoder_depth": 5}}
        updated = optuna_optimizer._update_config_with_params(config, {"encoder_depth": 3})
        assert updated["model"]["encoder_depth"] == 3

    # _save_best_config 

    def test_save_best_config_creates_file(self, optuna_optimizer, tmp_path, completed_study):
        optuna_optimizer._save_best_config(completed_study)
        expected = tmp_path / f"best_config_{optuna_optimizer.study_name}.yaml"
        assert expected.is_file()

    def test_save_best_config_yaml_contains_params(self, optuna_optimizer, tmp_path, completed_study):
        optuna_optimizer._save_best_config(completed_study)
        output = tmp_path / f"best_config_{optuna_optimizer.study_name}.yaml"
        saved = yaml.safe_load(output.read_text())
        assert saved["starting_lr"] == pytest.approx(1e-4)


    @pytest.mark.gpu
    @pytest.mark.slow
    def test_train_trial_returns_metric(self, optuna_optimizer):
        trial_config = optuna_optimizer.config.copy()
        trial_config["num_cyc_frozen"] = 0
        trial_config["num_cyc_unfrozen"] = 1
        study = optuna.create_study()
        real_trial = study.ask()
        result = optuna_optimizer._train_trial(trial_config, real_trial)
        assert isinstance(result, float)
        assert result >= 0.0

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_optimize_returns_study(self, optuna_optimizer):
        study = optuna_optimizer.optimize()
        assert isinstance(study, optuna.Study)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_optimize_best_value_in_range(self, optuna_optimizer):
        study = optuna_optimizer.optimize()
        assert 0.0 <= study.best_value <= 1.0

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_optimize_creates_model_file(self, optuna_optimizer, tmp_path):
        optuna_optimizer.optimize()
        models = list(tmp_path.glob("*_trial_*_model.pytorch"))
        assert len(models) >= 1

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_optimize_saves_best_config(self, optuna_optimizer, tmp_path):
        optuna_optimizer.optimize()
        expected = tmp_path / f"best_config_{optuna_optimizer.study_name}.yaml"
        assert expected.is_file()