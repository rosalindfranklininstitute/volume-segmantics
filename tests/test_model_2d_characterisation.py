"""Characterisation tests for ``create_model_on_device``.

"""

import pytest
import torch

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model.model_2d import create_model_on_device

MT = utils.ModelType

# PAN downsamples hard enough that 64 px collapses to 1x1 and the forward
# pass raises, so every case uses 128.
SIZE = 128

SMP_TYPES = [
    MT.U_NET, MT.U_NET_PLUS_PLUS, MT.FPN, MT.DEEPLABV3,
    MT.DEEPLABV3_PLUS, MT.MA_NET, MT.LINKNET, MT.PAN,
]


def struct(model_type, **overrides):
    d = {
        "type": model_type, "encoder_name": "resnet34", "encoder_weights": None,
        "in_channels": 3, "classes": 2,
    }
    d.update(overrides)
    return d


def forward(model, in_channels=3):
    model.eval()
    with torch.no_grad():
        return model(torch.zeros(1, in_channels, SIZE, SIZE))


@pytest.mark.parametrize("model_type", SMP_TYPES)
def test_smp_types_build_and_predict_class_channels(model_type):
    """Every smp architecture returns `classes` channels at input resolution."""
    model = create_model_on_device("cpu", struct(model_type))
    assert list(forward(model).shape) == [1, 2, SIZE, SIZE]


@pytest.mark.parametrize("model_type", SMP_TYPES)
@pytest.mark.parametrize("in_channels", [1, 3])
def test_in_channels_is_honoured(model_type, in_channels):
    model = create_model_on_device("cpu", struct(model_type, in_channels=in_channels))
    out = forward(model, in_channels)
    assert list(out.shape) == [1, 2, SIZE, SIZE]


@pytest.mark.parametrize("model_type", SMP_TYPES)
def test_classes_is_honoured(model_type):
    model = create_model_on_device("cpu", struct(model_type, classes=5))
    assert forward(model).shape[1] == 5


def test_segformer_builds():
    model = create_model_on_device(
        "cpu", struct(MT.SEGFORMER, encoder_name="mit_b0"),
    )
    assert type(model).__name__ == "Segformer"
    assert list(forward(model).shape) == [1, 2, SIZE, SIZE]


def test_vanilla_unet_needs_no_encoder():
    model = create_model_on_device(
        "cpu", {"type": MT.VANILLA_UNET, "in_channels": 3, "classes": 2},
    )
    assert type(model).__name__ == "UNet"
    assert list(forward(model).shape) == [1, 2, SIZE, SIZE]


def test_pipeline_multitask_returns_one_output_per_head():
    model = create_model_on_device("cpu", struct(
        MT.PIPELINE_MULTITASK_UNET,
        heads=[{"name": "semantic", "out_channels": 2},
               {"name": "boundary", "out_channels": 1}],
    ))
    outputs = forward(model)
    assert [list(o.shape) for o in outputs] == [
        [1, 2, SIZE, SIZE], [1, 1, SIZE, SIZE],
    ]


@pytest.mark.parametrize("type_value,expected", [
    ("U_NET", "Unet"),
    ("PIPELINE_MULTITASK_UNET", "PipelineMultitaskUnet"),
])
def test_model_type_accepts_the_string_form(type_value, expected):
    """Pipeline checkpoints store the name; legacy ones store the enum."""
    kwargs = {"heads": [{"name": "semantic", "out_channels": 2}]} \
        if "PIPELINE" in type_value else {}
    model = create_model_on_device("cpu", struct(type_value, **kwargs))
    assert type(model).__name__ == expected


def test_unknown_model_type_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown model type"):
        create_model_on_device("cpu", struct("NOT_A_MODEL"))


def test_legacy_multitask_unet_is_refused():
    """Deprecated in v0.4.0b3; raises rather than building."""
    with pytest.raises(DeprecationWarning, match="deprecated"):
        create_model_on_device("cpu", struct(MT.MULTITASK_UNET))


def test_extra_input_channels_adapt_the_pretrained_first_conv():
    """>3 channels with ImageNet weights widens the stem instead of failing."""
    model = create_model_on_device(
        "cpu", struct(MT.U_NET, in_channels=4, encoder_weights="imagenet"),
    )
    stem = next(p for p in model.parameters() if p.ndim == 4)
    assert stem.shape[1] == 4
    assert list(forward(model, 4).shape) == [1, 2, SIZE, SIZE]


@pytest.mark.parametrize("model_type", [MT.FPN, MT.DEEPLABV3_PLUS, MT.PAN, MT.MA_NET])
def test_unsupported_architectures_refuse_dino(model_type):
    """Refused by name, not by accident.

    Before DINO encoders were registered with smp these raised KeyError
    ("wrong encoder name"). Registration makes them buildable, and they would
    then emit predictions at the wrong resolution or fail deep inside the
    decoder, so the refusal is now explicit.
    """
    with pytest.raises(ValueError, match="does not support DINO encoders"):
        create_model_on_device(
            "cpu", struct(model_type, encoder_name="dinov2_vits14"),
        )


def test_the_lightning_checkpoint_type_accepts_dino_encoders():
    """Every Lightning-trained checkpoint is typed PIPELINE_MULTITASK_UNET, and
    that model builds DINO encoders at training time; prediction must not
    refuse what training produced. Depth 5 is fine: the model clamps to 4."""
    from volume_segmantics.model.model_2d import _check_dino_compatibility

    _check_dino_compatibility(MT.PIPELINE_MULTITASK_UNET, "dinov2_vits14", 5)


@pytest.fixture(scope="module")
def dino_ready():
    """Skip DINO cases when the backbone cannot be fetched.

    DINOEncoder pulls its weights through torch.hub on construction, so a
    machine with no cache and no network cannot run these.
    """
    try:
        create_model_on_device(
            "cpu", struct(MT.U_NET, encoder_name="dinov2_vits14"),
        )
    except Exception as exc:  # noqa: BLE001 - any fetch failure means skip
        pytest.skip(f"DINO backbone unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture
def dinov3_architecture_only(monkeypatch):
    """Meta gates the DINOv3 weights (HTTP 403 without a licence token), so
    the DINOv3 cases build the architecture from the hub repo without them.
    The loader itself no longer papers over that failure: before it was
    fixed these cases passed only because a failed download silently gave
    them a randomly initialised backbone."""
    real_load = torch.hub.load

    def _unweighted_v3(repo, model_name, *, pretrained=True, **kwargs):
        return real_load(repo, model_name, pretrained=pretrained and "dinov3" not in repo, **kwargs)

    monkeypatch.setattr(torch.hub, "load", _unweighted_v3)


@pytest.mark.parametrize("model_type", [MT.U_NET, MT.U_NET_PLUS_PLUS])
def test_dino_encoder_request_is_honoured(model_type, dino_ready, dinov3_architecture_only):
    """A DINO request must build a DINO-backed model.

    Regression test: registration used to fail silently and the fallback
    passed the encoder as smp.Unet(encoder=...), a parameter smp does not
    have. **kwargs swallowed it and encoder_name fell back to "resnet34",
    so training reported DINO and trained a ResNet.

    U-Net++ needs a patch-16 backbone; see the patch-size test below.
    """
    encoder = "dinov2_vits14" if model_type is MT.U_NET else "dinov3_vitl16"
    model = create_model_on_device(
        "cpu", struct(model_type, encoder_name=encoder),
    )
    assert type(model.encoder).__name__ == "DINOEncoder"
    assert list(forward(model).shape) == [1, 2, SIZE, SIZE]


def test_dino_encoder_honours_in_channels(dino_ready):
    """Exercises DINOEncoder.set_in_channels, which smp calls after build."""
    model = create_model_on_device(
        "cpu", struct(MT.U_NET, encoder_name="dinov2_vits14", in_channels=1),
    )
    assert model.encoder.in_channels == 1
    assert model.encoder.out_channels[0] == 1
    assert list(forward(model, 1).shape) == [1, 2, SIZE, SIZE]


@pytest.mark.parametrize("model_type", [MT.U_NET_PLUS_PLUS, MT.LINKNET])
def test_patch_size_must_match_the_depth(model_type, dino_ready, dinov3_architecture_only):
    """U-Net++ reconstructs (H // patch) * 2**depth, so patch must be 2**depth.

    Plain U-Net is anchored by its stem skip and is unaffected; U-Net++'s
    dense skips are not, and would return 144 px for a 128 px input.
    """
    with pytest.raises(ValueError, match="patch size 14"):
        create_model_on_device(
            "cpu", struct(model_type, encoder_name="dinov2_vits14"),
        )
    # The patch-16 backbone is the supported pairing.
    ok = create_model_on_device(
        "cpu", struct(model_type, encoder_name="dinov3_vitl16"),
    )
    assert list(forward(ok).shape) == [1, 2, SIZE, SIZE]


def test_a_dino_pipeline_checkpoint_rebuilds_without_hub_weights(monkeypatch, tmp_path):
    """Prediction fills the ViT from the checkpoint, so it asks the hub for the
    architecture only, and a weights path an older run stored is not passed on
    to the encoder, which refuses checkpoint paths."""
    stored = tmp_path / "dino.pth"
    stored.write_bytes(b"")
    requested = []

    def _architecture(repo, model_name, *, pretrained=True, trust_repo=True):
        requested.append((model_name, pretrained))
        return torch.nn.Module()

    monkeypatch.setattr(torch.hub, "load", _architecture)
    model = create_model_on_device("cpu", struct(
        MT.PIPELINE_MULTITASK_UNET, encoder_name="dinov2_vits14", encoder_weights=str(stored),
        encoder_depth=4, heads=[{"name": "semantic", "out_channels": 2}],
    ))
    assert type(model.encoder).__name__ == "DINOEncoder"
    assert requested == [("dinov2_vits14", False)]
