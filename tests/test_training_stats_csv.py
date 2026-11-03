"""The training stats CSV carries every epoch's own metrics.


"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import volume_segmantics.model.heads  # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets  # noqa: F401 — register targets
import volume_segmantics.utilities.base_data_utils as bdu

pl = pytest.importorskip("pytorch_lightning")

from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule  # noqa: E402
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    EpochHistoryCallback,
    LegacyEpochSummaryCallback,
    UnfreezeEncoderCallback,
    VolSeg2dLightningModule,
    VolSegCheckpointCallback,
    VolSegDiagnosticsCallback,
    model_struct_dict_for_checkpoint,
)


@pytest.fixture
def fixture_data(tmp_path):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(20):
        cv2.imwrite(str(img_dir / f"data_{i:03d}.png"), (rng.random((64, 64)) * 255).astype(np.uint8))
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


def test_every_row_has_its_own_losses_and_per_class_dice(fixture_data, tmp_path, monkeypatch):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    settings = SimpleNamespace(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=64, training_set_proportion=0.85,
        cuda_device=0, use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=1, patience=4, lr_policy="constant",
        eval_metric="DiceCoefficient",
    )
    config = PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})
    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir, settings=settings, pipeline_config=config, num_classes=2,
    )
    data_module.setup()
    module = VolSeg2dLightningModule.from_pipeline_config(
        config, settings=settings, num_classes=2, total_steps=len(data_module.train_dataloader()) * 2,
    )
    out = tmp_path / "run.pytorch"
    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[
            UnfreezeEncoderCallback(num_frozen_epochs=1),
            EpochHistoryCallback(),
            LegacyEpochSummaryCallback(),
            VolSegCheckpointCallback(output_path=out, model_struc_dict=model_struct_dict_for_checkpoint(settings, module, 2)),
            VolSegDiagnosticsCallback(output_path=out, num_classes=2, head_names=("semantic",)),
        ],
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=data_module)

    with open(tmp_path / "run_train_stats.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["Epoch"] for row in rows] == ["1", "2"]
    assert list(rows[0]) == ["Epoch", "Train_Loss", "Valid_Loss", "Seg_Dice", "Dice_Class_0", "Dice_Class_1"]
    for row in rows:
        assert float(row["Train_Loss"]) > 0 and float(row["Valid_Loss"]) > 0
        assert 0 < float(row["Seg_Dice"]) <= 1
        # The background covers most of every slice, so its Dice is never 0.
        assert float(row["Dice_Class_0"]) > 0
    # The last row is the last epoch's own numbers, not a zero-filled pad.
    assert float(rows[-1]["Train_Loss"]) == pytest.approx(float(trainer.callback_metrics["train_loss"]), rel=1e-5)
    assert float(rows[-1]["Dice_Class_0"]) == pytest.approx(float(trainer.callback_metrics["val_dice_class_0"]), abs=1e-6)
    assert (tmp_path / "run_loss_plot.png").exists()
