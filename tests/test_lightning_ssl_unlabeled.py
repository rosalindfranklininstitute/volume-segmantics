"""The default (Lightning) trainer's semi-supervised path.

"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import volume_segmantics.model.heads  # noqa: F401 - register heads
import volume_segmantics.model.loss_registry  # noqa: F401 - register losses
import volume_segmantics.data.targets  # noqa: F401 - register targets
import volume_segmantics.utilities.base_data_utils as bdu

pl = pytest.importorskip("pytorch_lightning")

from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule
from volume_segmantics.data.pipeline_dataset import UnlabeledPipelineDataset
from volume_segmantics.data.pipeline_loader import (
    ConsistencyRampConfig,
    PipelineConfigError,
    legacy_settings_to_pipeline_config,
    parse_pipeline_dict,
)
from volume_segmantics.model.operations.lightning2d import VolSeg2dLightningModule
from volume_segmantics.model.training.mean_teacher import MeanTeacherModel
from volume_segmantics.scripts.train_2d_model import _resolve_unlabeled_dir


def _settings(**over):
    base = dict(
        model={"type": "U_Net", "encoder_name": "resnet18", "encoder_weights": None, "encoder_depth": 5},
        starting_lr=5e-5, encoder_lr_multiplier=0.1, image_size=32,
        training_set_proportion=0.75, cuda_device=0, use_imagenet_norm=True,
        use_2_5d_slicing=False, num_slices=1, num_cyc_frozen=1, num_cyc_unfrozen=0,
        patience=4, max_label_no=2, loss_criterion="CombinedCEDiceLoss",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _write_slices(directory: Path, n: int, *, shape=(32, 32), labels=False, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        if labels:
            arr = np.zeros(shape, dtype=np.uint8)
            arr[shape[0] // 4: 3 * shape[0] // 4, shape[1] // 4: 3 * shape[1] // 4] = 1
        else:
            arr = (rng.random(shape) * 255).astype(np.uint8)
        cv2.imwrite(str(directory / f"slice_{i:03d}.png"), arr)
    return directory


# --- the ramp and the legacy mapping


def test_ramp_ratio_sigmoid_linear_and_no_ramp():
    sigmoid = ConsistencyRampConfig(start_step=0, end_step=100, type="sigmoid")
    assert sigmoid.ratio(0) == pytest.approx(math.exp(-5.0))
    assert sigmoid.ratio(50) == pytest.approx(math.exp(-5.0 * 0.25))
    assert sigmoid.ratio(100) == 1.0 and sigmoid.ratio(10_000) == 1.0
    linear = ConsistencyRampConfig(start_step=20, end_step=120, type="linear")
    assert linear.ratio(10) == 0.0 and linear.ratio(45) == pytest.approx(0.25)
    assert ConsistencyRampConfig().ratio(0) == 1.0  # end_step 0: full weight from step 0


def test_ramp_config_refuses_bad_values():
    with pytest.raises(PipelineConfigError, match="sigmoid"):
        ConsistencyRampConfig(type="cosine")
    with pytest.raises(PipelineConfigError, match="non-negative"):
        ConsistencyRampConfig(start_step=-1)


def test_pipeline_yaml_carries_the_ramp():
    cfg = parse_pipeline_dict({
        "heads": {"semantic": {"enabled": True}},
        "ema_teacher": {
            "enabled": True,
            "consistency_weights": {"semantic": 0.2},
            "consistency_rampup": {"start_step": 10, "end_step": 20, "type": "linear"},
        },
    })
    assert cfg.ema_teacher.consistency_rampup.ratio(15) == pytest.approx(0.5)
    bare = parse_pipeline_dict({"heads": {"semantic": {"enabled": True}}})
    assert bare.ema_teacher.consistency_rampup.ratio(0) == 1.0


def test_legacy_mapping_honours_the_documented_weight_and_ramp():
    cfg = legacy_settings_to_pipeline_config(_settings(
        use_semi_supervised=True, consistency_weight=0.1, rampup_start=0, rampup_end=100, ema_decay=0.99,
    ))
    ema = cfg.ema_teacher
    assert ema.enabled and ema.consistency_weights == {"semantic": 0.1}
    assert ema.consistency_rampup.end_step == 100 and ema.consistency_rampup.ratio(0) < 0.01
    assert ema.schedule.alpha_end == 0.99 and ema.schedule.warmup_steps == 0


def test_legacy_mapping_defaults_pseudo_only_and_off():
    default = legacy_settings_to_pipeline_config(_settings(use_semi_supervised=True))
    assert default.ema_teacher.consistency_weights == {"semantic": 0.1}
    assert default.ema_teacher.consistency_rampup.end_step == 10000
    pseudo_only = legacy_settings_to_pipeline_config(_settings(use_pseudo_labeling=True))
    assert pseudo_only.ema_teacher.enabled and pseudo_only.ema_teacher.consistency_weights == {"semantic": 0.0}
    off = legacy_settings_to_pipeline_config(_settings())
    assert not off.ema_teacher.enabled and off.ema_teacher.consistency_weights == {}


# --- unlabeled slices into the data module


def test_unlabeled_dataset_yields_two_views_in_the_labeled_layout(tmp_path):
    unl = _write_slices(tmp_path / "unl", 3, shape=(40, 24))
    cfg = legacy_settings_to_pipeline_config(_settings(use_semi_supervised=True))
    ds = UnlabeledPipelineDataset(unl, pipeline_config=cfg, img_size=32)
    assert len(ds) == 3
    item = ds[0]
    assert set(item) == {"image", "image_teacher"}
    assert item["image"].shape == (1, 32, 32) and item["image"].dtype == torch.float32
    assert item["image_teacher"].shape == (1, 32, 32)
    assert torch.equal(ds[0]["image_teacher"], ds[0]["image_teacher"])  # the teacher view is deterministic


def test_data_module_pairs_an_unlabeled_batch_with_every_labeled_batch(tmp_path, monkeypatch):
    img = _write_slices(tmp_path / "data", 12)
    msk = _write_slices(tmp_path / "seg", 12, labels=True)
    unl = _write_slices(tmp_path / "unl", 5, seed=1)
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **k: 4)
    settings = _settings(use_semi_supervised=True, unlabeled_batch_size=3)
    cfg = legacy_settings_to_pipeline_config(settings)
    dm = VolSeg2dDataModule(image_dir=img, label_dir=msk, settings=settings, pipeline_config=cfg, num_classes=2, unlabeled_dir=unl)
    dm.setup()
    steps = dm.steps_per_epoch()
    assert steps == len(dm.labeled_train_dataloader()) >= 1
    assert dm.unlabeled_train_dataloader() is not None
    batches = list(dm.train_dataloader())
    assert len(batches) == steps
    batch = batches[0][0] if isinstance(batches[0], tuple) else batches[0]
    assert set(batch) == {"labeled", "unlabeled"}
    assert batch["labeled"]["image"].shape[0] == 4 and "semantic" in batch["labeled"]
    assert batch["unlabeled"]["image"].shape == (3, 1, 32, 32)
    assert batch["unlabeled"]["image_teacher"].shape == (3, 1, 32, 32)
    # Without an unlabeled directory the plain labeled loader is returned.
    plain = VolSeg2dDataModule(image_dir=img, label_dir=msk, settings=settings, pipeline_config=cfg, num_classes=2)
    assert len(plain.train_dataloader()) == steps


def test_resolve_unlabeled_dir(tmp_path):
    assert _resolve_unlabeled_dir(_settings()) is None
    assert _resolve_unlabeled_dir(_settings(unlabeled_data_dir=str(tmp_path))) is None  # no SSL flag: ignored
    with pytest.raises(ValueError, match="unlabeled_data_dir must be provided"):
        _resolve_unlabeled_dir(_settings(use_semi_supervised=True, unlabeled_data_dir=""))
    with pytest.raises(FileNotFoundError):
        _resolve_unlabeled_dir(_settings(use_pseudo_labeling=True, unlabeled_data_dir=str(tmp_path / "missing")))
    assert _resolve_unlabeled_dir(_settings(use_semi_supervised=True, unlabeled_data_dir=str(tmp_path))) == tmp_path


# --- the fit uses them


def test_fit_trains_on_unlabeled_slices_with_ramped_consistency_and_pseudo_labels(tmp_path, monkeypatch):
    img = _write_slices(tmp_path / "data", 12)
    msk = _write_slices(tmp_path / "seg", 12, labels=True)
    unl = _write_slices(tmp_path / "unl", 6, seed=2)
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **k: 4)
    settings = _settings(
        use_semi_supervised=True, consistency_weight=0.1, rampup_start=0, rampup_end=1000,
        use_pseudo_labeling=True, pseudo_label_confidence_threshold=0.0,
        pseudo_label_min_pixels_per_class=0, pseudo_label_weight=0.5,
        pseudo_label_rampup_start=0, pseudo_label_rampup_end=1, unlabeled_batch_size=2,
        mean_teacher_vis_epoch_interval=0, pseudo_labeling_vis_epoch_interval=0,
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    dm = VolSeg2dDataModule(image_dir=img, label_dir=msk, settings=settings, pipeline_config=cfg, num_classes=2, unlabeled_dir=unl)
    dm.setup()
    steps = dm.steps_per_epoch()
    module = VolSeg2dLightningModule.from_pipeline_config(cfg, settings=settings, num_classes=2, in_channels=1, total_steps=steps)
    assert isinstance(module.model, MeanTeacherModel)
    assert module._pseudo_labelling is not None and module._pseudo_labelling.use_teacher
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    metrics = trainer.callback_metrics
    for key in ("train_loss", "train_supervised_loss", "train_semantic_consistency",
                "train_consistency_ramp", "train_pseudo_label_loss", "train_pseudo_label_acceptance"):
        assert key in metrics, f"{key} missing from {sorted(metrics)}"
    assert 0.0 < float(metrics["train_consistency_ramp"]) < 1.0  # ramp end 1000 > steps taken
    assert float(metrics["train_pseudo_label_acceptance"]) == pytest.approx(1.0)  # threshold 0 accepts every pixel
    assert float(metrics["train_pseudo_label_loss"]) > 0.0
    assert module._pseudo_labelling.total_pixels == steps * 2 * 32 * 32
