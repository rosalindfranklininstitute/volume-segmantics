"""End-to-end CLI smoke tests.

Runs the installed entry points (`model-train-2d`, `model-predict-2d`) as
subprocesses against a tiny synthetic volume with fast settings, asserting
exit 0, that the expected artifacts are written, and -- crucially -- that
training actually updated the weights (BatchNorm num_batches_tracked >= 1),
guarding the "checkpoint saved before training ran" failure mode.

GPU + slow: a real (short) training + prediction run.
"""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "scripts"))
from b3_assertions import assert_checkpoint_trained  # noqa: E402


def _fast_train_settings():
    with open(REPO_ROOT / "volseg-settings" / "2d_model_train_settings.yaml") as f:
        s = yaml.safe_load(f)
    s["image_size"] = 64
    s["num_cyc_frozen"] = 1
    s["num_cyc_unfrozen"] = 0
    s["patience"] = 10
    s["downsample"] = False
    # Small, randomly-initialised encoder so the smoke test is fast and needs
    # no pretrained-weight download.
    s["model"]["encoder_name"] = "resnet18"
    s["model"]["encoder_weights"] = None
    s["model"]["type"] = "U_Net"
    return s


def _make_synthetic_volume(path, label=False, seed=0):
    rng = np.random.default_rng(seed)
    shape = (40, 96, 96)
    if label:
        # Two classes with spatial structure so training has signal.
        z, y, x = np.indices(shape)
        data = ((x + y) > 96).astype(np.uint8)
    else:
        data = rng.integers(0, 256, size=shape, dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("/data", data=data)


@pytest.fixture()
def cli_data_dir(tmp_path):
    settings_dir = tmp_path / "volseg-settings"
    settings_dir.mkdir()
    with open(settings_dir / "2d_model_train_settings.yaml", "w") as f:
        yaml.safe_dump(_fast_train_settings(), f)
    # Predict settings: copy the repo default unchanged.
    with open(REPO_ROOT / "volseg-settings" / "2d_model_predict_settings.yaml") as f:
        predict_settings = yaml.safe_load(f)
    with open(settings_dir / "2d_model_predict_settings.yaml", "w") as f:
        yaml.safe_dump(predict_settings, f)

    img = tmp_path / "image.h5"
    lbl = tmp_path / "labels.h5"
    _make_synthetic_volume(img, label=False)
    _make_synthetic_volume(lbl, label=True)
    return tmp_path, img, lbl


def _run(module, args, cwd, extra_env=None):
    import os

    env = dict(os.environ)
    env["VOLSEG_NUM_WORKERS"] = "0"  # deterministic, no worker spawn overhead
    env["VOLSEG_SUPPRESS_RAW_TRAINER_DEPRECATION"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_cli_train_then_predict_smoke(cli_data_dir):
    data_dir, img, lbl = cli_data_dir

    # --- train ---
    res = _run(
        "volume_segmantics.scripts.train_2d_model",
        ["--data", str(img), "--labels", str(lbl), "--data_dir", str(data_dir)],
        cwd=data_dir,
    )
    assert res.returncode == 0, f"train failed:\nSTDOUT{res.stdout[-2000:]}\nSTDERR{res.stderr[-2000:]}"
    models = list(data_dir.glob("*.pytorch"))
    assert models, "no model checkpoint written by model-train-2d"
    model_path = models[0]
    # Training actually ran (not just a checkpoint of an untrained model).
    assert_checkpoint_trained(model_path)

    # --- predict --- (positional: <model> <data>, then --data_dir)
    res = _run(
        "volume_segmantics.scripts.predict_2d_model",
        [str(model_path), str(img), "--data_dir", str(data_dir)],
        cwd=data_dir,
    )
    assert res.returncode == 0, f"predict failed:\nSTDOUT{res.stdout[-2000:]}\nSTDERR{res.stderr[-2000:]}"
    # Prediction output is "{date}_{stem}_2d_model_vol_pred.{ext}".
    outputs = list(data_dir.glob("*vol_pred*"))
    assert outputs, f"no prediction output written; dir contents: {[p.name for p in data_dir.iterdir()]}"
