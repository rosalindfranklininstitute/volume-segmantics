from pathlib import Path
from types import SimpleNamespace

import csv
import numpy as np
import pytest
import torch
import torch.nn as nn

from volume_segmantics.model.operations import trainer_visualization as tv
from volume_segmantics.model.operations.trainer_visualization import TrainingVisualizer


@pytest.fixture(autouse=True)
def _patch_matplotlib_save(monkeypatch, tmp_path):

    def _write_dummy_png(fname, *args, **kwargs):
        # fname can be a Path, str, or a file-like object.
        if isinstance(fname, (str, Path)):
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            Path(fname).write_bytes(b"")

    # plt.savefig used by plot_predictions/plot_mean_teacher/pseudo-labeling
    monkeypatch.setattr(tv.plt, "savefig", _write_dummy_png, raising=False)

    # fig.savefig used by plot_loss_history
    try:
        import matplotlib.figure

        monkeypatch.setattr(
            matplotlib.figure.Figure,
            "savefig",
            lambda self, fname, *a, **k: _write_dummy_png(fname, *a, **k),
            raising=False,
        )
    except Exception:
        # If matplotlib internals differ, tests still run since we patched plt.savefig.
        pass


def _make_inputs_and_onehot_targets(B: int, C: int, H: int, W: int):
    # Inputs: (B, 1, H, W) so plotting treats it as grayscale (num_channels=1)
    inputs = torch.linspace(0, 1, steps=B * 1 * H * W, dtype=torch.float32).reshape(
        B, 1, H, W
    )

    # seg_target one-hot (B, C, H, W): make class index depend on pixel parity
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    class_idx = ((yy + xx) % C).to(torch.int64)  # (H,W)
    onehot = torch.nn.functional.one_hot(class_idx, num_classes=C).permute(2, 0, 1)
    onehot = onehot.unsqueeze(0).repeat(B, 1, 1, 1).float()
    return inputs, {"seg": onehot}


def test_plot_loss_history_single_task_creates_png_and_valid_csv(tmp_path, monkeypatch):
    num_classes = 3
    vis = TrainingVisualizer(num_classes=num_classes, label_codes={0: "bg", 1: "fg"}, use_multitask=False)

    epoch_history = {
        "train_total": [2.0, 1.0],
        "valid_total": [2.2, 1.1],
        "seg_dice": [0.1, 0.5],
        # Provide curves for only class 0 and 1; omit class 2 to hit missing-key behavior.
        "dice_class_0": [0.05, 0.6],
        "dice_class_1": [0.1, 0.4],
        # dice_class_2 intentionally missing
    }

    output_path = tmp_path / "run1.pytorch"
    vis.plot_loss_history(epoch_history=epoch_history, output_path=output_path)

    loss_png = tmp_path / "run1_loss_plot.png"
    assert loss_png.exists()

    csv_path = tmp_path / "run1_train_stats.csv"
    assert csv_path.exists()

    with csv_path.open("r", newline="") as f:
        reader = list(csv.reader(f))

    header = reader[0]
    # Single-task header must include loss columns and per-class dice columns.
    assert "Train_Loss" in header
    assert "Valid_Loss" in header
    assert "Seg_Dice" in header
    assert any(col.startswith("Dice_") for col in header)
    assert len([h for h in header if h.startswith("Dice_")]) == num_classes


def test_plot_loss_history_multitask_creates_png_and_csv_headers(tmp_path):
    num_classes = 3
    vis = TrainingVisualizer(
        num_classes=num_classes,
        label_codes={0: "bg", 1: "fg", 2: "c2"},
        use_multitask=True,
    )

    # Boundary arrays are all zeros: any(...) checks should skip boundary curve plotting
    epoch_history = {
        "train_total": [2.0, 1.0],
        "valid_total": [2.2, 1.1],
        "train_seg": [2.0, 1.0],
        "valid_seg": [2.2, 1.1],
        "train_boundary": [0.0, 0.0],
        "valid_boundary": [0.0, 0.0],
        "train_task3": [0.3, 0.2],
        "valid_task3": [0.35, 0.25],
        "seg_dice": [0.2, 0.4],
        "boundary_dice": [0.0, 0.0],  # should skip boundary dice curve
        "dice_class_0": [0.1, 0.2],
        "dice_class_1": [0.2, 0.3],
        "dice_class_2": [0.3, 0.4],
    }

    output_path = tmp_path / "run_mt.pytorch"
    vis.plot_loss_history(epoch_history=epoch_history, output_path=output_path)

    csv_path = tmp_path / "run_mt_train_stats.csv"
    assert csv_path.exists()

    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))

    # Multitask header should include boundary/task columns and Boundary_Dice.
    assert "Train_Boundary" in header
    assert "Valid_Boundary" in header
    assert "Train_Task3" in header
    assert "Valid_Task3" in header
    assert "Boundary_Dice" in header


def test_plot_predictions_single_task_saves_png(monkeypatch, tmp_path):
    # Prepare a deterministic batch and monkeypatch prepare_training_batch.
    B, C, H, W = 2, 3, 8, 8
    inputs, targets = _make_inputs_and_onehot_targets(B=B, C=C, H=H, W=W)
    seg_logits = torch.zeros(B, C, H, W)
    # Make argmax be consistent with parity-based class_idx.
    # Use onehot to build logits.
    seg_logits = targets["seg"] * 10.0

    def _fake_prepare_training_batch(batch, device_num, num_classes):
        assert num_classes == C
        return inputs, targets

    monkeypatch.setattr(tv.utils, "prepare_training_batch", _fake_prepare_training_batch)

    class _Model(nn.Module):
        def forward(self, x):
            return seg_logits

    model = _Model()
    ensure_tuple_output_fn = lambda out: (out,) if not isinstance(out, tuple) else out

    class _Loader:
        batch_size = B

        def __iter__(self):
            yield object()

    loader = _Loader()
    vis = TrainingVisualizer(num_classes=C)

    model_path = tmp_path / "model_one_task.pytorch"
    vis.plot_predictions(
        model=model,
        validation_loader=loader,
        device_num="cpu",
        model_path=model_path,
        ensure_tuple_output_fn=ensure_tuple_output_fn,
    )

    out_png = tmp_path / "model_one_task_prediction_image.png"
    assert out_png.exists()


def test_plot_predictions_boundary_and_task3_saves_png(monkeypatch, tmp_path):
    B, C, H, W = 2, 3, 8, 8
    inputs, targets = _make_inputs_and_onehot_targets(B=B, C=C, H=H, W=W)

    # Boundary + task3 logits/targets:
    boundary_target = torch.zeros(B, 1, H, W)
    task3_target = torch.zeros(B, 1, H, W)
    boundary_target[:, :, 2:4, 2:4] = 1.0
    task3_target[:, :, 5:7, 5:7] = 1.0
    targets["boundary"] = boundary_target
    targets["task3"] = task3_target

    seg_logits = targets["seg"] * 5.0
    boundary_logits = boundary_target * 10.0 - 5.0  # sigmoid threshold will pick >0.5 in the box
    task3_logits = task3_target * 10.0 - 5.0

    def _fake_prepare_training_batch(batch, device_num, num_classes):
        return inputs, targets

    monkeypatch.setattr(tv.utils, "prepare_training_batch", _fake_prepare_training_batch)

    class _Model(nn.Module):
        def forward(self, x):
            return (seg_logits, boundary_logits, task3_logits)

    model = _Model()
    ensure_tuple_output_fn = lambda out: out if isinstance(out, tuple) else (out,)

    class _Loader:
        batch_size = B

        def __iter__(self):
            yield object()

    loader = _Loader()
    vis = TrainingVisualizer(num_classes=C, use_multitask=True)

    model_path = tmp_path / "model_mt_pred.pytorch"
    vis.plot_predictions(
        model=model,
        validation_loader=loader,
        device_num="cpu",
        model_path=model_path,
        ensure_tuple_output_fn=ensure_tuple_output_fn,
    )

    out_png = tmp_path / "model_mt_pred_prediction_image.png"
    assert out_png.exists()


def test_plot_mean_teacher_predictions_saves_png(monkeypatch, tmp_path):
    B, C, H, W = 2, 3, 8, 8
    inputs, targets = _make_inputs_and_onehot_targets(B=B, C=C, H=H, W=W)
    seg_target = targets["seg"]

    # Student and teacher logits differ slightly.
    student_logits = seg_target * 6.0
    teacher_logits = seg_target * 6.0 + 1.0

    def _fake_prepare_training_batch(batch, device_num, num_classes):
        return inputs, {"seg": seg_target}

    monkeypatch.setattr(tv.utils, "prepare_training_batch", _fake_prepare_training_batch)

    class _Student(nn.Module):
        def forward(self, x):
            return student_logits

    class _Teacher(nn.Module):
        def forward(self, x):
            return teacher_logits

    class _MeanTeacherWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self._student = _Student()
            self._teacher = _Teacher()

        def get_student_model(self):
            return self._student

        def get_teacher_model(self):
            return self._teacher

    mean_teacher = _MeanTeacherWrap()

    class _Loader:
        batch_size = B

        def __iter__(self):
            yield object()

    loader = _Loader()
    vis = TrainingVisualizer(num_classes=C)

    out_path = tmp_path / "mt_vis.png"
    vis.plot_mean_teacher_predictions(
        model=mean_teacher,
        validation_loader=loader,
        device_num="cpu",
        output_path=out_path,
        epoch=1,
        ensure_tuple_output_fn=lambda out: (out,),
    )

    expected = tmp_path / "mt_vis_mean_teacher_epoch_1.png"
    assert expected.exists()


def test_plot_pseudo_labeling_visualization_saves_png_and_uses_teacher(monkeypatch, tmp_path):
    B, C, H, W = 1, 3, 8, 8
    unlabeled_inputs = torch.rand(B, 1, H, W)

    # Build pseudo-label outputs:
    pseudo_labels = torch.zeros(B, H, W, dtype=torch.int64)
    pseudo_labels[:, 2:6, 2:6] = 1
    confidence_map = torch.ones(B, H, W, dtype=torch.float32) * 0.9
    mask = torch.zeros(B, H, W, dtype=torch.uint8)
    mask[:, 2:6, 2:6] = 1
    probs = torch.rand(B, C, H, W, dtype=torch.float32)

    class _PseudoGen:
        use_teacher_for_labels = True

        def __init__(self):
            self.last_use_teacher = None
            self.last_model_for_labels = None

        def generate_pseudo_labels(
            self,
            model_for_labels,
            unlabeled_inputs_arg,
            num_classes,
            use_teacher=False,
        ):
            self.last_model_for_labels = model_for_labels
            self.last_use_teacher = use_teacher
            return {
                "pseudo_labels": pseudo_labels,
                "confidence_map": confidence_map,
                "mask": mask,
                "probs": probs,
            }

    pseudo_gen = _PseudoGen()

    class _Student(nn.Module):
        def forward(self, x):
            return x

    class _BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = _Student()
            self.teacher = _Student()

        def get_student_model(self):
            return self.student

        def get_teacher_model(self):
            return self.teacher

    base_model = _BaseModel()

    class _UnlabeledLoader:
        def __iter__(self):
            yield {"img": unlabeled_inputs}

    loader = _UnlabeledLoader()
    vis = TrainingVisualizer(num_classes=C)

    out_path = tmp_path / "pl_model.pytorch"
    vis.plot_pseudo_labeling_visualization(
        model=base_model,
        unlabeled_loader=loader,
        device_num="cpu",
        output_path=out_path,
        epoch=0,
        pseudo_label_generator=pseudo_gen,
        ensure_tuple_output_fn=lambda out: (out,),
    )

    expected = tmp_path / "pl_model_pseudo_labeling_epoch_0.png"
    assert expected.exists()
    assert pseudo_gen.last_use_teacher is True

