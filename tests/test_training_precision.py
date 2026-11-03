"""Mixed-precision training is a setting of the default trainer.

The train settings' ``precision`` and the ``--precision`` option select Lightning's
precision; full precision stays the default.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import volume_segmantics.model.heads  # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets  # noqa: F401 — register targets
import volume_segmantics.utilities.base_data_utils as bdu
from volume_segmantics.utilities.arg_parsing import get_2d_training_parser

pl = pytest.importorskip("pytorch_lightning")

from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule  # noqa: E402
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    VolSeg2dLightningModule,
    resolve_precision,
)


@pytest.mark.parametrize("raw, expected", [
    (None, "32-true"), ("", "32-true"), (32, "32-true"), ("32", "32-true"), ("32-true", "32-true"),
    (16, "16-mixed"), ("16", "16-mixed"), ("16-mixed", "16-mixed"),
    ("bf16", "bf16-mixed"), ("bf16-mixed", "bf16-mixed"), ("BF16-Mixed", "bf16-mixed"),
    (64, "64-true"),
])
def test_precision_settings_resolve_to_lightning_values(raw, expected):
    assert resolve_precision(SimpleNamespace(precision=raw)) == expected


def test_full_precision_is_the_default_when_the_setting_is_absent():
    assert resolve_precision(SimpleNamespace()) == "32-true"


def test_the_command_line_wins_over_the_settings():
    assert resolve_precision(SimpleNamespace(precision="16-mixed"), override="bf16-mixed") == "bf16-mixed"
    assert resolve_precision(SimpleNamespace(precision="16-mixed"), override=None) == "16-mixed"


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision must be one of"):
        resolve_precision(SimpleNamespace(precision="half"))


def test_the_train_parser_takes_precision():
    args = get_2d_training_parser().parse_args(["--data", "a.h5", "--labels", "b.h5", "--precision", "bf16-mixed"])
    assert args.precision == "bf16-mixed"
    assert get_2d_training_parser().parse_args(["--data", "a.h5"]).precision is None


@pytest.fixture
def fixture_data(tmp_path):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        cv2.imwrite(str(img_dir / f"data_{i:03d}.png"), (rng.random((64, 64)) * 255).astype(np.uint8))
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


def test_a_mixed_precision_fit_runs(fixture_data, tmp_path, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = SimpleNamespace(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=64, training_set_proportion=0.85,
        cuda_device=0, use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=0, patience=1, lr_policy="constant", precision="bf16-mixed",
    )
    config = PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})
    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir, settings=settings, pipeline_config=config, num_classes=2,
    )
    data_module.setup()
    module = VolSeg2dLightningModule.from_pipeline_config(
        config, settings=settings, num_classes=2, total_steps=len(data_module.train_dataloader()),
    )
    trainer = pl.Trainer(
        max_epochs=1, precision=resolve_precision(settings), accelerator="cpu", devices=1,
        logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    assert trainer.precision == "bf16-mixed"
    trainer.fit(module, datamodule=data_module)
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert torch.isfinite(trainer.callback_metrics["val_loss"])
    # The weights stay in full precision; autocast is per operation.
    assert all(p.dtype == torch.float32 for p in module.parameters())
