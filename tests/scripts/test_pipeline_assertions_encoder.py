"""``encoder-family``: the checkpoint's weights must corroborate its declared encoder.

"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_assertions as A  # noqa: E402

DINO_KEYS = ["encoder.dino_model.patch_embed.proj.weight", "encoder.dino_model.blocks.0.norm1.weight"]
RESNET_KEYS = ["encoder.conv1.weight", "encoder.layer1.0.conv1.weight"]


def _checkpoint(tmp_path: Path, encoder_name: str, keys, *, with_state=True) -> Path:
    payload = {"model_struc_dict": {"encoder_name": encoder_name, "type": "U_NET"}}
    if with_state:
        payload["model_state_dict"] = {k: torch.zeros(1) for k in keys}
    path = tmp_path / "model.pytorch"
    torch.save(payload, path)
    return path


def test_dino_declared_and_dino_weights_pass(tmp_path):
    A.assert_encoder_family(_checkpoint(tmp_path, "dinov2_vits14", DINO_KEYS), expect_name="dinov2_vits14")


def test_dino_declared_but_resnet_weights_is_the_historical_bug(tmp_path):
    with pytest.raises(AssertionError, match="encoder.dino_model"):
        A.assert_encoder_family(_checkpoint(tmp_path, "dinov2_vits14", RESNET_KEYS))


def test_dino_weights_alongside_resnet_weights_are_rejected_too(tmp_path):
    with pytest.raises(AssertionError, match="encoder.conv1"):
        A.assert_encoder_family(_checkpoint(tmp_path, "dinov2_vits14", DINO_KEYS + RESNET_KEYS))


def test_declared_name_must_match_expect(tmp_path):
    with pytest.raises(AssertionError, match="dinov2_vitb14"):
        A.assert_encoder_family(_checkpoint(tmp_path, "dinov2_vits14", DINO_KEYS), expect_name="dinov2_vitb14")


def test_an_encoder_with_no_signature_cannot_be_corroborated(tmp_path):
    with pytest.raises(AssertionError, match="no weight signature"):
        A.assert_encoder_family(_checkpoint(tmp_path, "resnet34", RESNET_KEYS))


def test_name_only_checks_the_declaration_alone(tmp_path):
    A.assert_encoder_family(
        _checkpoint(tmp_path, "resnet34", RESNET_KEYS), expect_name="resnet34", name_only=True
    )


def test_a_checkpoint_without_a_state_dict_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="no state dict"):
        A.assert_encoder_family(_checkpoint(tmp_path, "dinov2_vits14", DINO_KEYS, with_state=False))


def test_the_cli_subcommand_dispatches(tmp_path):
    path = _checkpoint(tmp_path, "dinov2_vits14", DINO_KEYS)
    A.main(["encoder-family", str(path), "--expect", "dinov2_vits14"])
    with pytest.raises(AssertionError):
        A.main(["encoder-family", str(_checkpoint(tmp_path, "dinov2_vits14", RESNET_KEYS))])
