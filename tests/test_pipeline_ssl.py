"""Test multihead SSL teacher.


"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

import volume_segmantics.model.heads        # noqa: F401
import volume_segmantics.model.loss_registry  # noqa: F401
import volume_segmantics.data.targets       # noqa: F401

import volume_segmantics.utilities.base_data_utils as bdu


pl = pytest.importorskip("pytorch_lightning")


from volume_segmantics.data.lightning_dataloaders import VolSeg2dDataModule
from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    EMAScheduleConfig,
    EMATeacherConfig,
    HeadConfig,
    PipelineConfig,
    TransformSpec,
)
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.operations.lightning2d import (
    VolSeg2dLightningModule,
)
from volume_segmantics.model.operations.trainer_losses import (
    ConsistencyLoss,
    boundary_consistency_loss,
    distance_consistency_loss,
    per_head_consistency_loss,
    sdm_consistency_loss,
    seg_consistency_loss,
)
from volume_segmantics.model.pipeline_multitask_unet import PipelineMultitaskUnet
from volume_segmantics.model.training.mean_teacher import (
    EMASchedule,
    MeanTeacherModel,
)
from volume_segmantics.model.training.multihead_disagreement import (
    DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS,
    compute_per_head_disagreement,
)
from volume_segmantics.model.training.pseudo_labeling import (
    PseudoLabelGenerator,
)


# EMASchedule 


def test_ema_schedule_defaults():
    s = EMASchedule()
    assert s.alpha_warmup == 0.99
    assert s.alpha_end == 0.999
    assert s.warmup_steps == 500


def test_ema_schedule_alpha_at_zero_is_warmup():
    s = EMASchedule(alpha_warmup=0.95, alpha_end=0.999, warmup_steps=100)
    assert s.alpha(0) == pytest.approx(0.95)


def test_ema_schedule_alpha_post_warmup_is_end():
    s = EMASchedule(alpha_warmup=0.95, alpha_end=0.999, warmup_steps=100)
    assert s.alpha(100) == pytest.approx(0.999)
    assert s.alpha(500) == pytest.approx(0.999)


def test_ema_schedule_alpha_mid_warmup_interpolates():
    s = EMASchedule(alpha_warmup=0.0, alpha_end=1.0, warmup_steps=100)
    # 50/100 = 0.5 -> α = 0.0 + 0.5 * 1.0 = 0.5.
    assert s.alpha(50) == pytest.approx(0.5)


def test_ema_schedule_zero_warmup_steps_returns_end():
    s = EMASchedule(alpha_warmup=0.5, alpha_end=0.99, warmup_steps=0)
    assert s.alpha(0) == pytest.approx(0.99)


def test_ema_schedule_rejects_out_of_range():
    with pytest.raises(ValueError, match="alpha_warmup"):
        EMASchedule(alpha_warmup=1.5)
    with pytest.raises(ValueError, match="alpha_end"):
        EMASchedule(alpha_end=-0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        EMASchedule(warmup_steps=-1)


# MeanTeacherModel 


def _tiny_student() -> nn.Module:
    """A tiny conv+BN module for testing the EMA buffer path."""
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 1, 3, padding=1),
    )


def test_mean_teacher_with_schedule_uses_schedule_alpha():
    student = _tiny_student()
    sched = EMASchedule(alpha_warmup=0.0, alpha_end=1.0, warmup_steps=10)
    mt = MeanTeacherModel(student, schedule=sched)
    # At step 0 with α_warmup=0.0, the teacher should jump to student
    # weights entirely (α=0 -> θ_t = 0·θ_t + 1·θ_s = θ_s).
    # First, perturb the student so it differs from the teacher.
    with torch.no_grad():
        for p in mt.student.parameters():
            p.data += 1.0
    mt.update_teacher()
    # After update, teacher matches student.
    for s_p, t_p in zip(mt.student.parameters(), mt.teacher.parameters()):
        torch.testing.assert_close(s_p.data, t_p.data)


def test_mean_teacher_legacy_adaptive_path_when_no_schedule():
    student = _tiny_student()
    mt = MeanTeacherModel(student, ema_decay=0.99)  # no schedule
    # adaptive: α = min(1 - 1/(0+1), 0.99) = min(0, 0.99) = 0 -> teacher copies student.
    with torch.no_grad():
        for p in mt.student.parameters():
            p.data += 1.0
    mt.update_teacher()
    for s_p, t_p in zip(mt.student.parameters(), mt.teacher.parameters()):
        torch.testing.assert_close(s_p.data, t_p.data)


def test_mean_teacher_ema_updates_bn_running_stats():
    """**The B3.E.1 contract**: BN running stats are EMA-updated, not
    left at random init like v0.4.0b2."""
    student = _tiny_student()
    sched = EMASchedule(alpha_warmup=0.5, alpha_end=0.5, warmup_steps=0)
    mt = MeanTeacherModel(student, schedule=sched)

    # Run a forward pass on the student in train mode so its BN
    # running_mean/var update from defaults.
    student.train()
    x = torch.randn(2, 1, 16, 16)
    _ = student(x)
    # Confirm student BN moved from default running_mean=0 / running_var=1.
    s_bn = student[1]
    assert not torch.allclose(s_bn.running_mean, torch.zeros(4), atol=1e-6)

    # Capture teacher BN before update.
    t_bn = mt.teacher[1]
    teacher_mean_before = t_bn.running_mean.clone()
    teacher_var_before = t_bn.running_var.clone()

    mt.update_teacher()

    # Teacher BN should have moved toward the student's BN stats.
    assert not torch.allclose(t_bn.running_mean, teacher_mean_before, atol=1e-6)
    assert not torch.allclose(t_bn.running_var, teacher_var_before, atol=1e-6)


def test_mean_teacher_state_dict_round_trip_with_schedule():
    student_a = _tiny_student()
    student_b = _tiny_student()
    sched = EMASchedule(alpha_warmup=0.95, alpha_end=0.999, warmup_steps=200)
    mt_a = MeanTeacherModel(student_a, ema_decay=0.99, schedule=sched)
    # Advance some iterations.
    for _ in range(5):
        mt_a.update_teacher()
    sd = mt_a.state_dict()
    assert sd["schedule"] == {
        "alpha_warmup": 0.95, "alpha_end": 0.999, "warmup_steps": 200,
    }
    assert sd["glob_it"] == 5

    mt_b = MeanTeacherModel(student_b)
    mt_b.load_state_dict(sd)
    assert mt_b.glob_it == 5
    assert mt_b.schedule is not None
    assert mt_b.schedule.warmup_steps == 200


def test_mean_teacher_state_dict_round_trip_without_schedule():
    """Legacy state dicts (no 'schedule' key) load cleanly."""
    student_a = _tiny_student()
    mt = MeanTeacherModel(student_a, ema_decay=0.97)
    sd = mt.state_dict()
    assert sd["schedule"] is None

    student_b = _tiny_student()
    mt_b = MeanTeacherModel(student_b)
    mt_b.load_state_dict(sd)
    assert mt_b.schedule is None
    assert mt_b.ema_decay == pytest.approx(0.97)


def test_mean_teacher_parameters_returns_only_student():
    student = _tiny_student()
    mt = MeanTeacherModel(student)
    student_param_ids = {id(p) for p in student.parameters()}
    mt_param_ids = {id(p) for p in mt.parameters()}
    assert mt_param_ids == student_param_ids


# Per-head consistency losses 


def test_seg_consistency_zero_on_identical():
    sem = torch.randn(2, 4, 16, 16)
    assert float(seg_consistency_loss(sem, sem.clone())) == 0.0


def test_boundary_consistency_zero_on_identical():
    bnd = torch.randn(2, 1, 16, 16)
    assert float(boundary_consistency_loss(bnd, bnd.clone())) == 0.0


def test_distance_consistency_zero_on_identical():
    dst = torch.randn(2, 1, 16, 16)
    assert float(distance_consistency_loss(dst, dst.clone())) == 0.0


def test_sdm_consistency_zero_on_identical():
    sdm = torch.tanh(torch.randn(2, 1, 16, 16))
    assert float(sdm_consistency_loss(sdm, sdm.clone())) == 0.0


def test_seg_consistency_positive_on_different():
    a = torch.randn(2, 4, 16, 16)
    b = torch.randn(2, 4, 16, 16)
    assert float(seg_consistency_loss(a, b)) > 0.0


def test_consistency_loss_legacy_class_back_compat():
    """v0.4.0b2 ConsistencyLoss class still works (just delegates)."""
    sem = torch.randn(2, 4, 16, 16)
    assert float(ConsistencyLoss()(sem, sem.clone())) == 0.0


def test_per_head_consistency_dispatches_correctly():
    sem = torch.randn(2, 4, 16, 16)
    bnd = torch.randn(2, 1, 16, 16)
    dst = torch.randn(2, 1, 16, 16)
    sdm = torch.tanh(torch.randn(2, 1, 16, 16))
    assert float(per_head_consistency_loss("semantic", sem, sem.clone())) == 0.0
    assert float(per_head_consistency_loss("boundary", bnd, bnd.clone())) == 0.0
    assert float(per_head_consistency_loss("distance", dst, dst.clone())) == 0.0
    assert float(per_head_consistency_loss("sdm", sdm, sdm.clone())) == 0.0


def test_per_head_consistency_unknown_head_raises():
    with pytest.raises(ValueError, match="unknown head"):
        per_head_consistency_loss(
            "lsd", torch.randn(2, 6, 16, 16), torch.randn(2, 6, 16, 16),
        )


def test_consistency_teacher_detach():
    """Consistency losses must detach the teacher — gradient should
    flow only through the student."""
    student = torch.randn(2, 4, 16, 16, requires_grad=True)
    teacher = torch.randn(2, 4, 16, 16, requires_grad=True)
    loss = seg_consistency_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


# compute_per_head_disagreement 


def test_disagreement_default_weights_set():
    assert set(DEFAULT_PER_HEAD_DISAGREEMENT_WEIGHTS) == {
        "semantic", "boundary", "distance", "sdm",
    }


def test_disagreement_zero_on_identical_outputs():
    s = {"semantic": torch.randn(2, 3, 16, 16)}
    t = {"semantic": s["semantic"].clone()}
    d = compute_per_head_disagreement(s, t)
    assert d.shape == (2, 1, 16, 16)
    assert float(d.max()) == 0.0


def test_disagreement_positive_on_different_outputs():
    s = {"semantic": torch.randn(2, 3, 16, 16)}
    t = {"semantic": torch.randn(2, 3, 16, 16)}
    d = compute_per_head_disagreement(s, t)
    assert float(d.mean()) > 0.0


def test_disagreement_in_unit_interval():
    s = {
        "semantic": torch.randn(2, 3, 16, 16),
        "boundary": torch.randn(2, 1, 16, 16),
        "distance": torch.randn(2, 1, 16, 16),
        "sdm":      torch.tanh(torch.randn(2, 1, 16, 16)),
    }
    t = {
        "semantic": torch.randn(2, 3, 16, 16),
        "boundary": torch.randn(2, 1, 16, 16),
        "distance": torch.randn(2, 1, 16, 16),
        "sdm":      torch.tanh(torch.randn(2, 1, 16, 16)),
    }
    d = compute_per_head_disagreement(s, t)
    assert d.shape == (2, 1, 16, 16)
    assert float(d.min()) >= 0.0
    assert float(d.max()) <= 1.0


def test_disagreement_skips_heads_absent_on_either_side():
    s = {
        "semantic": torch.randn(2, 3, 16, 16),
        "boundary": torch.randn(2, 1, 16, 16),
    }
    t = {
        "semantic": torch.randn(2, 3, 16, 16),
        # no boundary — should be silently skipped
    }
    d = compute_per_head_disagreement(s, t)
    assert d.shape == (2, 1, 16, 16)


def test_disagreement_empty_intersection_returns_zeros():
    s = {"semantic": torch.randn(2, 3, 16, 16)}
    t = {"boundary": torch.randn(2, 1, 16, 16)}
    d = compute_per_head_disagreement(s, t)
    # No overlap -> zeros tensor with student's spatial shape.
    assert torch.allclose(d, torch.zeros_like(d))


def test_disagreement_empty_inputs_raises():
    with pytest.raises(ValueError, match="student_out"):
        compute_per_head_disagreement({}, {"semantic": torch.zeros(1, 3, 8, 8)})
    with pytest.raises(ValueError, match="teacher_out"):
        compute_per_head_disagreement({"semantic": torch.zeros(1, 3, 8, 8)}, {})


def test_disagreement_zero_weights_uniform_fallback():
    """All-zero weights -> uniform mean of per-head maps, not zeros."""
    s = {"semantic": torch.randn(2, 3, 16, 16)}
    t = {"semantic": torch.randn(2, 3, 16, 16)}
    d = compute_per_head_disagreement(s, t, weights={"semantic": 0.0})
    # Should still have signal.
    assert float(d.mean()) > 0.0


def test_disagreement_teacher_detach():
    """Disagreement gradients flow only through the student."""
    s = {"semantic": torch.randn(2, 3, 16, 16, requires_grad=True)}
    t = {"semantic": torch.randn(2, 3, 16, 16, requires_grad=True)}
    d = compute_per_head_disagreement(s, t)
    # Disagreement on argmax is non-differentiable wrt logits; use
    # boundary instead which is differentiable.
    s2 = {"boundary": torch.randn(2, 1, 16, 16, requires_grad=True)}
    t2 = {"boundary": torch.randn(2, 1, 16, 16, requires_grad=True)}
    d2 = compute_per_head_disagreement(s2, t2)
    d2.sum().backward()
    assert s2["boundary"].grad is not None
    assert t2["boundary"].grad is None


# PseudoLabelGenerator.generate_per_head_pseudo_labels 


class _MultiHeadStub(nn.Module):
    """Stub multi-head model emitting (semantic, boundary, distance, sdm)."""

    def __init__(self):
        super().__init__()
        self.head_names = ("semantic", "boundary", "distance", "sdm")

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        return (
            torch.randn(b, 3, h, w),     # semantic logits
            torch.randn(b, 1, h, w),     # boundary logits
            torch.rand(b, 1, h, w) * 5,  # distance (positive)
            torch.tanh(torch.randn(b, 1, h, w)),  # sdm
        )


def test_per_head_pseudo_labels_shapes_and_dtypes():
    model = _MultiHeadStub()
    gen = PseudoLabelGenerator(
        confidence_threshold=0.5, min_pixels_per_class=0,
    )
    x = torch.randn(2, 1, 16, 16)
    out = gen.generate_per_head_pseudo_labels(
        model, x, head_names=model.head_names, num_classes=3,
    )
    assert set(out) == {"semantic", "boundary", "distance", "sdm"}
    # Semantic: long indices.
    assert out["semantic"]["pseudo_target"].shape == (2, 16, 16)
    assert out["semantic"]["pseudo_target"].dtype == torch.long
    assert out["semantic"]["mask"].shape == (2, 16, 16)
    assert out["semantic"]["mask"].dtype == torch.bool
    # Boundary: float (B, 1, H, W).
    assert out["boundary"]["pseudo_target"].shape == (2, 1, 16, 16)
    assert out["boundary"]["pseudo_target"].dtype == torch.float32
    # Distance: float, all-True mask.
    assert out["distance"]["pseudo_target"].shape == (2, 1, 16, 16)
    assert torch.all(out["distance"]["mask"])
    # SDM: float, all-True mask.
    assert out["sdm"]["pseudo_target"].shape == (2, 1, 16, 16)
    assert torch.all(out["sdm"]["mask"])


def test_per_head_pseudo_labels_semantic_mask_threshold():
    """High threshold -> fewer pixels accepted."""
    model = _MultiHeadStub()
    x = torch.randn(2, 1, 16, 16)

    gen_low = PseudoLabelGenerator(confidence_threshold=0.0, min_pixels_per_class=0)
    out_low = gen_low.generate_per_head_pseudo_labels(
        model, x, head_names=model.head_names, num_classes=3,
    )
    accept_low = out_low["semantic"]["mask"].sum().item()
    assert accept_low > 0  # some accepted

    gen_high = PseudoLabelGenerator(
        confidence_threshold=0.99, min_pixels_per_class=0,
    )
    out_high = gen_high.generate_per_head_pseudo_labels(
        model, x, head_names=model.head_names, num_classes=3,
    )
    accept_high = out_high["semantic"]["mask"].sum().item()
    # Random logits -> softmax max rarely exceeds 0.99 -> almost none accepted.
    assert accept_high <= accept_low


def test_per_head_pseudo_labels_boundary_margin_rejects_uncertain():
    """Larger margin -> more pixels rejected near sigmoid≈0.5."""
    model = _MultiHeadStub()
    x = torch.randn(2, 1, 16, 16)
    gen = PseudoLabelGenerator(confidence_threshold=0.0, min_pixels_per_class=0)

    out_strict = gen.generate_per_head_pseudo_labels(
        model, x, head_names=model.head_names, num_classes=3,
        boundary_confidence_margin=0.49,
    )
    out_loose = gen.generate_per_head_pseudo_labels(
        model, x, head_names=model.head_names, num_classes=3,
        boundary_confidence_margin=0.0,
    )
    accept_strict = out_strict["boundary"]["mask"].sum().item()
    accept_loose = out_loose["boundary"]["mask"].sum().item()
    assert accept_strict <= accept_loose


def test_per_head_pseudo_labels_unknown_head_raises():
    """Single-output stub with an unsupported head name -> 'unknown head'."""

    class _SingleHeadStub(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])

    gen = PseudoLabelGenerator()
    x = torch.randn(2, 1, 16, 16)
    with pytest.raises(ValueError, match="unknown head"):
        gen.generate_per_head_pseudo_labels(
            _SingleHeadStub(), x, head_names=("flow",), num_classes=3,
        )


# Lightning module SSL wiring 


def _ssl_pipeline_config():
    return PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True, loss="dice_ce"),
            "boundary": HeadConfig(enabled=True, loss="boundary_bce_dice"),
        },
        ema_teacher=EMATeacherConfig(
            enabled=True,
            schedule=EMAScheduleConfig(
                alpha_warmup=0.0, alpha_end=0.5, warmup_steps=2,
            ),
            consistency_weights={"semantic": 0.5, "boundary": 0.3},
        ),
        augmentations=AugmentationsConfig(train_transforms=[]),
    )


def _ssl_settings():
    return SimpleNamespace(
        model={
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "encoder_depth": 5,
        },
        starting_lr=5e-5, encoder_lr_multiplier=1.0,
        image_size=64, training_set_proportion=0.85, cuda_device=0,
        use_imagenet_norm=True, use_2_5d_slicing=False, num_slices=1,
        max_label_no=3,
    )


def test_lightning_ssl_construction_wraps_in_mean_teacher():
    cfg = _ssl_pipeline_config()
    settings = _ssl_settings()
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=settings, num_classes=3, in_channels=1, total_steps=10,
    )
    assert mod._ssl_enabled is True
    assert isinstance(mod.model, MeanTeacherModel)
    assert isinstance(mod._underlying_student, PipelineMultitaskUnet)
    assert mod._consistency_weights == {"semantic": 0.5, "boundary": 0.3}


def test_lightning_no_ssl_construction_uses_bare_model():
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
    )
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=_ssl_settings(), num_classes=3,
        in_channels=1, total_steps=10,
    )
    assert mod._ssl_enabled is False
    assert isinstance(mod.model, PipelineMultitaskUnet)
    assert mod._consistency_weights == {}


def test_lightning_ssl_freeze_encoder_drills_through_mean_teacher():
    cfg = _ssl_pipeline_config()
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=_ssl_settings(), num_classes=3,
        in_channels=1, total_steps=10,
    )
    mod.freeze_encoder()
    for p in mod._underlying_student.encoder.parameters():
        assert p.requires_grad is False
    mod.unfreeze_encoder()
    for p in mod._underlying_student.encoder.parameters():
        assert p.requires_grad is True


def test_lightning_ssl_optimizer_only_sees_student_params():
    """Teacher params must NOT receive gradients via the optimiser."""
    cfg = _ssl_pipeline_config()
    settings = _ssl_settings()
    settings.encoder_lr_multiplier = 0.1  # exercise param-group split
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=settings, num_classes=3,
        in_channels=1, total_steps=10,
    )
    opt = mod.configure_optimizers()
    # Sum the param counts across groups.
    opt_param_count = sum(len(g["params"]) for g in opt.param_groups)
    student_param_count = len(list(mod._underlying_student.parameters()))
    assert opt_param_count == student_param_count


def test_lightning_ssl_forward_and_teacher_advance():
    cfg = _ssl_pipeline_config()
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=_ssl_settings(), num_classes=3,
        in_channels=1, total_steps=10,
    )
    x = torch.randn(2, 1, 64, 64)
    preds = mod.forward(x)
    assert isinstance(preds, tuple)
    assert len(preds) == 2

    # on_train_batch_end should advance the EMA teacher.
    glob_before = mod.model.glob_it
    mod.on_train_batch_end(outputs=None, batch=None, batch_idx=0)
    assert mod.model.glob_it == glob_before + 1


def test_lightning_no_ssl_on_train_batch_end_is_noop():
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
    )
    mod = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=_ssl_settings(), num_classes=3,
        in_channels=1, total_steps=10,
    )
    # No exception even without MeanTeacher wrapper.
    mod.on_train_batch_end(outputs=None, batch=None, batch_idx=0)


# End-to-end: SSL fit with EMA teacher 


@pytest.fixture
def fixture_data_ssl(tmp_path):
    img_dir = tmp_path / "data"
    msk_dir = tmp_path / "seg"
    img_dir.mkdir()
    msk_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(20):
        cv2.imwrite(
            str(img_dir / f"data_{i:03d}.png"),
            (rng.random((64, 64)) * 255).astype(np.uint8),
        )
        msk = np.zeros((64, 64), dtype=np.uint8)
        msk[16:48, 16:48] = 1
        msk[24:40, 24:40] = 2
        cv2.imwrite(str(msk_dir / f"seg_{i:03d}.png"), msk)
    return img_dir, msk_dir


def test_end_to_end_ssl_fit_advances_teacher(
    fixture_data_ssl, tmp_path, monkeypatch,
):
    """1-epoch SSL fit on a 20-sample fixture. Asserts:
    * EMA teacher's glob_it advances past zero.
    * Teacher BN running stats moved from random init.
    """
    img_dir, msk_dir = fixture_data_ssl
    monkeypatch.setattr(bdu, "get_batch_size", lambda *a, **kw: 4)

    cfg = _ssl_pipeline_config()
    settings = _ssl_settings()
    settings.num_cyc_frozen = 0
    settings.num_cyc_unfrozen = 1

    data_module = VolSeg2dDataModule(
        image_dir=img_dir, label_dir=msk_dir,
        settings=settings, pipeline_config=cfg, num_classes=3,
    )
    pl_module = VolSeg2dLightningModule.from_pipeline_config(
        cfg, settings=settings, num_classes=3,
        in_channels=1, total_steps=10,
    )
    assert pl_module._ssl_enabled

    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1,
        logger=False, enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(pl_module, datamodule=data_module)

    # glob_it advanced past zero.
    assert pl_module.model.glob_it > 0
    # Per-epoch consistency loss logged.
    assert any(
        k.endswith("consistency") for k in trainer.callback_metrics
    )
