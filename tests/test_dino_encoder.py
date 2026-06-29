import types

import pytest
import torch
import torch.nn as nn

from volume_segmantics.model.encoders.dino_encoder import DINOEncoder
import volume_segmantics.model.encoders.dino_encoder as dino_mod


class _DummyPatchEmbedV2(nn.Module):
    def __init__(self, *, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)  # (B, D, H', W')
        B, D, Hp, Wp = y.shape
        return y.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)


class _DummyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class DummyDINOv2(nn.Module):
    def __init__(self, *, embed_dim: int, patch_size: int, num_layers: int, pos_patches: int):
        super().__init__()
        self.patch_embed = _DummyPatchEmbedV2(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size
        )
        # pos_embed is (1, N+1, D) in the usual DINO format
        N = pos_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([_DummyBlock(embed_dim) for _ in range(num_layers)])


class DummyDINOv3(nn.Module):
    def __init__(self, *, embed_dim: int, patch_size: int, num_layers: int, token_n_override=None):
        super().__init__()
        # Used by _adapt_input_channels, even for v3 where forward uses get_intermediate_layers
        self.patch_embed = _DummyPatchEmbedV2(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size
        )
        self.num_layers = num_layers
        self.token_n_override = token_n_override

    def get_intermediate_layers(self, x, n, reshape, return_class_token, return_extra_tokens, norm):
        # DINOEncoder calls with n being a list of layer indices (len == depth).
        # We return a tuple/list of tokens (B, N, D), one per requested layer index.
        B, _, H, W = x.shape
        expected_h = H // self.patch_embed.proj.kernel_size[0]
        expected_w = W // self.patch_embed.proj.kernel_size[0]
        expected_n = expected_h * expected_w
        n_tokens = self.token_n_override if self.token_n_override is not None else expected_n
        D = self.patch_embed.proj.out_channels
        tokens = torch.zeros(B, n_tokens, D, device=x.device, dtype=x.dtype)
        # Make each requested layer return slightly different tensors (non-identical is fine)
        if isinstance(n, (list, tuple)):
            out = tuple(tokens + float(i) for i in range(len(n)))
        else:
            out = (tokens,)
        return out


def _dummy_torch_hub_load_factory(record_calls: list):
    def _dummy_load(repo_or_org, model_name, pretrained=False, trust_repo=True, **kwargs):
        record_calls.append(
            {
                "repo_or_org": repo_or_org,
                "model_name": model_name,
                "pretrained": pretrained,
            }
        )
        # Normalize dinov3 hub naming variants to the config keys in DINOEncoder.DINO_MODELS
        normalized = model_name
        if model_name.startswith("dinov3_vitl"):
            normalized = "dinov3_vitl16"
        elif model_name.startswith("dinov3_vit7b"):
            normalized = "dinov3_vit7b16"

        cfg = DINOEncoder.DINO_MODELS.get(normalized)
        if cfg is None:
            raise ValueError(f"Unexpected dummy model_name: {model_name}")

        patch_size = cfg["patch_size"]
        embed_dim = cfg["embed_dim"]
        num_layers = cfg["num_layers"]
        if cfg["version"] == "v2":
            # Provide a fixed pos_embed patch count that usually differs from test input
            return DummyDINOv2(
                embed_dim=embed_dim,
                patch_size=patch_size,
                num_layers=num_layers,
                pos_patches=16,
            )
        # v3
        return DummyDINOv3(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_layers=num_layers,
        )

    return _dummy_load


def test_dino_encoder_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown DINO model"):
        DINOEncoder(model_name="not_a_model")


def test_dino_encoder_respects_torch_hub_available_flag(monkeypatch):
    monkeypatch.setattr(dino_mod, "TORCH_HUB_AVAILABLE", False)
    with pytest.raises(ImportError, match="torch.hub is not available"):
        # Call __init__, which triggers _load_dino_model
        DINOEncoder(model_name="dinov2_vits14", pretrained=False, weights="dinov2")


def test_dino_encoder_v2_forward_output_shapes_and_selected_layers(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.hub, "load", _dummy_torch_hub_load_factory(calls))

    enc = DINOEncoder(
        model_name="dinov2_vits14",
        in_channels=3,
        depth=4,
        weights="dinov2",
        pretrained=False,
    )
    assert enc.patch_size == DINOEncoder.DINO_MODELS["dinov2_vits14"]["patch_size"]
    assert len(enc.selected_layers) == 4
    assert all(0 <= i <= enc.num_layers - 1 for i in enc.selected_layers)

    x = torch.randn(2, 3, 15, 15)  # triggers padding/cropping path
    feats = enc(x)
    assert isinstance(feats, list)
    assert len(feats) == 4

    # For patch_size=14: floor(15/14) == 1
    expected_spatial = 1
    for idx, f in enumerate(feats):
        assert f.shape[0] == 2
        assert f.shape[2] == expected_spatial
        assert f.shape[3] == expected_spatial
        assert f.shape[1] == enc.out_channels[idx]


def test_dino_encoder_input_channel_adaptation_grayscale(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", _dummy_torch_hub_load_factory([]))

    enc = DINOEncoder(
        model_name="dinov2_vits14",
        in_channels=1,
        depth=2,
        weights="dinov2",
        pretrained=False,
    )
    assert enc.dino_model.patch_embed.proj.in_channels == 1

    x = torch.randn(1, 1, 14, 14)
    feats = enc(x)
    assert len(feats) == 2
    assert feats[0].shape[1] == enc.out_channels[0]


def test_dino_encoder_input_channel_adaptation_multi_channel(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", _dummy_torch_hub_load_factory([]))

    enc = DINOEncoder(
        model_name="dinov2_vits14",
        in_channels=5,
        depth=2,
        weights="dinov2",
        pretrained=False,
    )
    assert enc.dino_model.patch_embed.proj.in_channels == 5

    x = torch.randn(1, 5, 14, 14)
    feats = enc(x)
    assert len(feats) == 2
    assert feats[0].shape[1] == enc.out_channels[0]


def test_dino_encoder_v3_forward_and_token_padding(monkeypatch):
    # Make dummy return a wrong token count to exercise padding/truncation in _patch_tokens_to_feature_map.
    def _dummy_v3_load(repo_or_org, model_name, pretrained=False, trust_repo=True, **kwargs):
        cfg_key = "dinov3_vitl16" if model_name.startswith("dinov3_vitl") else "dinov3_vit7b16"
        cfg = DINOEncoder.DINO_MODELS[cfg_key]
        return DummyDINOv3(
            embed_dim=cfg["embed_dim"],
            patch_size=cfg["patch_size"],
            num_layers=cfg["num_layers"],
            token_n_override=3,  # expected for 32x32 w/16 patch is 4 => triggers padding
        )

    monkeypatch.setattr(torch.hub, "load", _dummy_v3_load)

    enc = DINOEncoder(
        model_name="dinov3_vitl16",
        in_channels=3,
        depth=3,
        weights="dinov3",
        pretrained=False,
    )
    x = torch.randn(2, 3, 32, 32)  # expected 2x2 patches => 4 tokens (but dummy returns 3)
    feats = enc(x)
    assert len(feats) == 3
    assert feats[0].shape[2:] == (2, 2) or feats[0].shape[2:] == (2, 2)
    for i, f in enumerate(feats):
        assert f.shape[1] == enc.out_channels[i]

