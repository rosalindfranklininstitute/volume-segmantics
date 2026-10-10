"""A data profile must move every path an assertion reads.

Before this, five configs spelled the default output directory in their Dice
gates and four spelled a label volume, so ``--data-profile multiclass`` scored
the default run's predictions, or the wrong labels, and reported 26 false
failures and one false pass.
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
    for name in ("vessels_256cube_DATA.h5", "vessels_256cube_LABELS.h5", "cells_INSTANCE.tif"):
        (tmp_path / "training_data" / name).write_bytes(b"")
    (tmp_path / "model.pytorch").write_bytes(b"")
    monkeypatch.setattr(H, "_repo_root", lambda: tmp_path)
    return tmp_path


def _config(repo: Path, steps, paths=None, output_root="tmp/smoke") -> Path:
    cfg = {"tests": steps, "output_root": output_root}
    if paths is not None:
        cfg["paths"] = paths
    path = repo / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _dispatch(monkeypatch, **handlers):
    monkeypatch.setattr(H, "STEP_DISPATCH", {**H.STEP_DISPATCH, **handlers})


def test_a_predict_step_registers_its_prediction_glob_under_the_profile_output_root(repo, monkeypatch):
    commands = []
    monkeypatch.setattr(H, "_run_command", lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(H, "_assert_globs", lambda **kw: None)
    seen = {}

    def record(ctx, step):
        seen["pred"] = H._resolve_token("${artifact:p_pred}", ctx, ctx.output_root)
        seen["root"] = ctx.output_root

    _dispatch(monkeypatch, pipeline_assert=record)
    cfg = _config(repo, [
        {"name": "p", "type": "predict", "model_path": str(repo / "model.pytorch")},
        {"name": "gate", "type": "pipeline_assert", "subcommand": "segmentation-dice",
         "args": ["${artifact:p_pred}"]},
    ])
    H.run_harness(cfg, path_overrides={"image": "training_data/vessels_256cube_DATA.h5",
                                       "label": "training_data/vessels_256cube_LABELS.h5"},
                  output_suffix="other")
    assert seen["root"] == (repo / "tmp/smoke_other").resolve()
    assert seen["pred"] == str(seen["root"] / "p" / H.PREDICTION_GLOB)


def test_pred_key_renames_the_registered_glob(repo, monkeypatch):
    monkeypatch.setattr(H, "_run_command", lambda cmd, **kw: None)
    monkeypatch.setattr(H, "_assert_globs", lambda **kw: None)
    keys = {}
    _dispatch(monkeypatch, pipeline_assert=lambda ctx, step: keys.update(ctx.artifacts))
    H.run_harness(_config(repo, [
        {"name": "p", "type": "predict", "model_path": str(repo / "model.pytorch"), "pred_key": "mine"},
        {"name": "gate", "type": "pipeline_assert", "subcommand": "zarr-keys"},
    ]))
    assert "mine" in keys and "p_pred" not in keys


def test_a_profile_replaces_the_paths_block_instead_of_merging_into_it():
    cfg = {"paths": {"image": "a.h5", "label": "b.h5", "instance_label": "cells.tif"}, "output_root": "tmp/x"}
    H._apply_profile(cfg, {"image": "c.tif", "label": "d.tif"}, "multiclass")
    assert cfg["paths"] == {"image": "c.tif", "label": "d.tif"}
    assert cfg["output_root"] == "tmp/x_multiclass"


def test_instance_label_resolves_when_the_data_set_has_one(repo, monkeypatch):
    seen = {}
    _dispatch(monkeypatch, pipeline_assert=lambda ctx, step: seen.update(
        gt=H._resolve_token("${path:instance_label}", ctx, ctx.output_root)))
    H.run_harness(_config(repo, [
        {"name": "gate", "type": "pipeline_assert", "subcommand": "instance-detection",
         "args": ["--gt-label", "${path:instance_label}"]},
    ], paths={"image": "training_data/vessels_256cube_DATA.h5",
              "label": "training_data/vessels_256cube_LABELS.h5",
              "instance_label": "training_data/cells_INSTANCE.tif"}))
    assert seen["gt"] == str((repo / "training_data/cells_INSTANCE.tif").resolve())


def test_a_gate_needing_instance_labels_is_skipped_under_a_profile_without_them(repo, monkeypatch, capsys):
    ran = []
    _dispatch(monkeypatch, pipeline_assert=lambda ctx, step: ran.append(step["name"]))
    cfg = _config(repo, [
        {"name": "gate", "type": "pipeline_assert", "subcommand": "instance-detection",
         "args": ["--gt-label", "${path:instance_label}"]},
        {"name": "other", "type": "pipeline_assert", "subcommand": "zarr-keys", "args": []},
    ], paths={"image": "training_data/vessels_256cube_DATA.h5",
              "label": "training_data/vessels_256cube_LABELS.h5",
              "instance_label": "training_data/cells_INSTANCE.tif"})
    failures = H.run_harness(cfg, path_overrides={"image": "training_data/vessels_256cube_DATA.h5",
                                                  "label": "training_data/vessels_256cube_LABELS.h5"},
                             output_suffix="multiclass")
    assert ran == ["other"]
    assert failures == []
    assert "[SKIP] gate: no 'instance_label' path" in capsys.readouterr().out


def test_a_missing_instance_label_file_is_an_infrastructure_error(repo):
    cfg = _config(repo, [{"name": "gate", "type": "pipeline_assert", "subcommand": "zarr-keys"}],
                  paths={"image": "training_data/vessels_256cube_DATA.h5",
                         "label": "training_data/vessels_256cube_LABELS.h5",
                         "instance_label": "training_data/absent.tif"})
    with pytest.raises(FileNotFoundError, match="Instance label path not found"):
        H.run_harness(cfg)
