"""``eval_metric`` reaches the default trainer.

Dice is still always reported, and selects the checkpoint; ``MeanIoU`` adds 
the mean and per-class IoU to the epoch summary, the stats CSV and the loss plot. 
``dice_averaging`` and ``exclude_background_from_dice`` shape the reported mean.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

import volume_segmantics.model.heads  # noqa: F401 — register heads
import volume_segmantics.model.loss_registry  # noqa: F401 — register losses
import volume_segmantics.data.targets  # noqa: F401 — register targets
import volume_segmantics.utilities.base_data_utils as bdu

pl = pytest.importorskip("pytorch_lightning")

from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule  # noqa: E402
from volume_segmantics.data.pipeline_loader import HeadConfig, PipelineConfig  # noqa: E402
from volume_segmantics.model.operations import lightning2d  # noqa: E402
from volume_segmantics.model.operations.lightning2d import (  # noqa: E402
    EpochHistoryCallback,
    LegacyEpochSummaryCallback,
    VolSeg2dLightningModule,
    VolSegDiagnosticsCallback,
    _SemanticDiceAccumulator,
    eval_metric_from_settings,
)


def _counts(tp, fp, fn):
    acc = _SemanticDiceAccumulator(len(tp))
    acc.tp, acc.fp, acc.fn = list(tp), list(fp), list(fn)
    return acc


def test_iou_comes_from_the_same_counts_as_dice():
    acc = _counts(tp=[8, 2], fp=[1, 2], fn=[1, 2])
    assert acc.compute() == pytest.approx([16 / 18, 4 / 8])
    assert acc.compute_iou() == pytest.approx([8 / 10, 2 / 6])


def test_the_macro_mean_skips_absent_classes_and_the_background_when_asked():
    acc = _counts(tp=[8, 2, 0], fp=[1, 2, 0], fn=[1, 2, 0])  # class 2 never seen
    per_class = acc.compute()
    assert acc.average(per_class) == pytest.approx((16 / 18 + 0.5) / 2)
    assert acc.average(per_class, exclude_background=True) == pytest.approx(0.5)
    assert _counts([0], [0], [0]).average([0.0]) == 0.0


def test_the_weighted_mean_weights_each_class_by_its_target_count():
    acc = _counts(tp=[90, 5], fp=[0, 0], fn=[10, 5])  # targets: 100 and 10 pixels
    per_class = acc.compute()
    freq = {0: 100 / 110, 1: 10 / 110}
    inverse = {c: 1 / f for c, f in freq.items()}
    expected = sum(per_class[c] * inverse[c] for c in (0, 1)) / sum(inverse.values())
    assert acc.average(per_class, averaging="weighted", weight_mode="inverse_freq") == pytest.approx(expected)
    by_pixels = (per_class[0] * 100 + per_class[1] * 10) / 110
    assert acc.average(per_class, averaging="weighted", weight_mode="pixel_count") == pytest.approx(by_pixels)
    assert acc.average(per_class, averaging="weighted", weight_mode="uniform") == pytest.approx(sum(per_class) / 2)


@pytest.mark.parametrize("raw, expected", [
    ("MeanIoU", "MeanIoU"), ("meaniou", "MeanIoU"), ("DiceCoefficient", "DiceCoefficient"),
    ("dicecoefficient", "DiceCoefficient"), (None, "DiceCoefficient"), ("", "DiceCoefficient"),
])
def test_eval_metric_names_are_normalised(raw, expected):
    assert eval_metric_from_settings(SimpleNamespace(eval_metric=raw)) == expected


def _settings(**overrides):
    base = dict(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=1.0, image_size=64, training_set_proportion=0.85,
        cuda_device=0, use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        num_cyc_frozen=1, num_cyc_unfrozen=0, patience=1, lr_policy="constant",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _semantic():
    return PipelineConfig(heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")})


def test_an_unknown_eval_metric_is_refused_when_the_module_is_built():
    with pytest.raises(ValueError, match="eval_metric"):
        VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=_settings(eval_metric="Jaccard"), num_classes=2)
    with pytest.raises(ValueError, match="dice_averaging"):
        VolSeg2dLightningModule.from_pipeline_config(_semantic(), settings=_settings(dice_averaging="micro"), num_classes=2)


def test_the_epoch_summary_line_prints_the_iou_when_it_was_logged(monkeypatch):
    lines = []
    monkeypatch.setattr(lightning2d, "_emit_via_progress_console", lambda trainer, line: lines.append(line))
    trainer = MagicMock()
    trainer.global_rank, trainer.sanity_checking, trainer.current_epoch = 0, False, 0
    trainer.callback_metrics = {
        "train_loss": torch.tensor(0.5), "val_loss": torch.tensor(0.4), "val_seg_dice": torch.tensor(0.8),
        "val_seg_iou": torch.tensor(0.7), "val_dice_class_0": torch.tensor(0.9), "val_dice_class_1": torch.tensor(0.7),
        "val_iou_class_0": torch.tensor(0.85), "val_iou_class_1": torch.tensor(0.55),
    }
    pl_module = SimpleNamespace(head_names=("semantic",), num_classes=2, label_codes={})
    callback = LegacyEpochSummaryCallback()
    # Nothing at validation end: the epoch's training loss is aggregated only
    # once the training epoch ends, which is where the line is printed.
    callback.on_validation_end(trainer, pl_module)
    assert lines == []
    callback.on_train_epoch_end(trainer, pl_module)
    assert "Epoch 1 | Train Loss: 0.5000 | Val Loss: 0.4000 | Seg Dice: 0.8000 | Seg IoU: 0.7000" in lines[0]
    assert lines[1].strip() == "Per-class Dice: C0: 0.900 | C1: 0.700"
    assert lines[2].strip() == "Per-class IoU: C0: 0.850 | C1: 0.550"


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
        msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


def _fit(fixture_data, tmp_path, monkeypatch, settings, epochs=1):
    img_dir, msk_dir = fixture_data
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)
    config = _semantic()
    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir, settings=settings, pipeline_config=config, num_classes=3,
    )
    data_module.setup()
    module = VolSeg2dLightningModule.from_pipeline_config(
        config, settings=settings, num_classes=3, total_steps=len(data_module.train_dataloader()) * epochs,
    )
    out = tmp_path / "metric_run.pytorch"
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[EpochHistoryCallback(), VolSegDiagnosticsCallback(output_path=out, num_classes=3, head_names=("semantic",))],
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=data_module)
    with open(tmp_path / "metric_run_train_stats.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return trainer, module, rows


def test_mean_iou_is_logged_and_written_to_the_csv(fixture_data, tmp_path, monkeypatch):
    trainer, module, rows = _fit(fixture_data, tmp_path, monkeypatch, _settings(eval_metric="MeanIoU"))
    metrics = trainer.callback_metrics
    assert {"val_seg_dice", "val_seg_iou", "val_iou_class_0", "val_iou_class_1", "val_iou_class_2"} <= set(metrics)
    # IoU is never above Dice for the same class.
    for c in range(3):
        assert float(metrics[f"val_iou_class_{c}"]) <= float(metrics[f"val_dice_class_{c}"]) + 1e-6
    assert all(f"val_iou_class_{c}" in row for c in range(3) for row in module.epoch_history)
    assert list(rows[0]) == [
        "Epoch", "Train_Loss", "Valid_Loss", "Seg_Dice", "Dice_Class_0", "Dice_Class_1", "Dice_Class_2",
        "Seg_IoU", "IoU_Class_0", "IoU_Class_1", "IoU_Class_2",
    ]
    assert float(rows[-1]["Seg_IoU"]) == pytest.approx(float(metrics["val_seg_iou"]), abs=1e-6)
    assert float(rows[-1]["IoU_Class_0"]) > 0
    assert (tmp_path / "metric_run_loss_plot.png").exists()


def test_dice_coefficient_reports_no_iou(fixture_data, tmp_path, monkeypatch):
    trainer, _, rows = _fit(fixture_data, tmp_path, monkeypatch, _settings(eval_metric="DiceCoefficient"))
    assert "val_seg_iou" not in trainer.callback_metrics
    assert list(rows[0]) == ["Epoch", "Train_Loss", "Valid_Loss", "Seg_Dice", "Dice_Class_0", "Dice_Class_1", "Dice_Class_2"]
