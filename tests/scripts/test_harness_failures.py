"""Quality assertions are collected; infrastructure errors stop the run.

"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_data_harness as H  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "volseg-settings").mkdir()
    (tmp_path / "volseg-settings" / "2d_model_train_settings.yaml").write_text("image_size: 64\n")
    (tmp_path / "volseg-settings" / "2d_model_predict_settings.yaml").write_text("quality: low\n")
    (tmp_path / "training_data").mkdir()
    for name in ("vessels_256cube_DATA.h5", "vessels_256cube_LABELS.h5"):
        (tmp_path / "training_data" / name).write_bytes(b"")
    monkeypatch.setattr(H, "_repo_root", lambda: tmp_path)
    return tmp_path


def _config(repo: Path, steps) -> Path:
    path = repo / "cfg.yaml"
    path.write_text(yaml.safe_dump({"tests": steps}))
    return path


def _dispatch(monkeypatch, **handlers):
    monkeypatch.setattr(H, "STEP_DISPATCH", {**H.STEP_DISPATCH, **handlers})


def test_a_failing_quality_assertion_is_collected_and_the_run_continues(repo, monkeypatch):
    seen = []

    def failing_assert(ctx, step):
        raise RuntimeError("Command failed with exit code 1: pipeline_assertions ...")

    _dispatch(monkeypatch, pipeline_assert=failing_assert, predict=lambda ctx, step: seen.append(step["name"]))
    cfg = _config(repo, [
        {"name": "check", "type": "pipeline_assert", "subcommand": "zarr-keys"},
        {"name": "after", "type": "predict", "model_path": "x"},
    ])
    failures = H.run_harness(cfg)
    assert seen == ["after"]
    assert failures == ["check: Command failed with exit code 1: pipeline_assertions ..."]


def test_an_assert_outputs_failure_is_a_quality_failure_too(repo, monkeypatch):
    def predict_with_missing_output(ctx, step):
        raise AssertionError("Expected files matching '*_vol_pred.*' in run, found none.")

    _dispatch(monkeypatch, predict=predict_with_missing_output)
    failures = H.run_harness(_config(repo, [{"name": "p", "type": "predict", "model_path": "x"}]))
    assert failures == ["p: Expected files matching '*_vol_pred.*' in run, found none."]


def test_an_infrastructure_error_stops_the_run(repo, monkeypatch):
    seen = []

    def failing_train(ctx, step):
        raise RuntimeError("Command failed with exit code 2: train")

    _dispatch(monkeypatch, train=failing_train, predict=lambda ctx, step: seen.append(step["name"]))
    cfg = _config(repo, [
        {"name": "t", "type": "train"},
        {"name": "after", "type": "predict", "model_path": "x"},
    ])
    with pytest.raises(RuntimeError, match="exit code 2"):
        H.run_harness(cfg)
    assert seen == []


def test_a_clean_run_reports_no_failures(repo, monkeypatch):
    _dispatch(monkeypatch, predict=lambda ctx, step: None)
    assert H.run_harness(_config(repo, [{"name": "p", "type": "predict", "model_path": "x"}])) == []
