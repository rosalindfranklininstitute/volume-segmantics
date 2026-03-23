import torch
import pytest

from volume_segmantics.utilities.early_stopping import EarlyStopping


def _make_model_and_optimizer():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_early_stopping_initial_save_sets_best_and_checkpoint(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    ckpt_path = tmp_path / "es_checkpoint.pt"

    es = EarlyStopping(
        patience=2,
        verbose=False,
        path=ckpt_path,
        model_dict={"model": "dummy"},
        best_score=None,
    )

    val_loss = 1.25
    es(val_loss, model, optimizer, label_codes={"a": 1})

    assert ckpt_path.exists()
    assert es.best_score == pytest.approx(-val_loss, abs=1e-7)
    assert es.val_loss_min == pytest.approx(val_loss, abs=1e-7)
    assert es.counter == 0
    assert es.early_stop is False

    saved = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in saved
    assert "model_struc_dict" in saved
    assert "optimizer_state_dict" in saved
    assert "loss_val" in saved
    assert saved["loss_val"] == pytest.approx(val_loss, abs=1e-7)
    assert saved["label_codes"] == {"a": 1}


def test_early_stopping_no_improve_triggers_after_patience(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    ckpt_path = tmp_path / "es_checkpoint.pt"

    es = EarlyStopping(patience=2, verbose=False, path=ckpt_path, model_dict={})

    # Initialize best_score at val_loss=1.0
    es(1.0, model, optimizer, label_codes={})
    assert es.counter == 0
    assert es.early_stop is False

    # No improvement: val_loss increases => score decreases
    es(1.1, model, optimizer, label_codes={})
    assert es.counter == 1
    assert es.early_stop is False

    es(1.2, model, optimizer, label_codes={})
    assert es.counter == 2
    assert es.early_stop is True


def test_early_stopping_improve_resets_counter_and_saves(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    ckpt_path = tmp_path / "es_checkpoint.pt"

    es = EarlyStopping(patience=3, verbose=False, path=ckpt_path, model_dict={})

    es(1.0, model, optimizer, label_codes={})
    assert es.counter == 0
    assert es.early_stop is False

    es(1.1, model, optimizer, label_codes={})  # no improve
    assert es.counter == 1

    es(0.9, model, optimizer, label_codes={})  # improvement
    assert es.counter == 0
    assert es.early_stop is False


def test_early_stopping_save_checkpoint_includes_glob_it(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    ckpt_path = tmp_path / "es_checkpoint.pt"

    es = EarlyStopping(patience=2, verbose=False, path=ckpt_path, model_dict={})

    es(1.0, model, optimizer, label_codes={"x": 2}, glob_it=7)
    saved = torch.load(ckpt_path, map_location="cpu")
    assert saved["glob_it"] == 7

