"""
DINO Vision Transformer encoder for segmentation.

This module provides DINO (v1, v2, v3) encoder wrappers that adapt Vision Transformer
models to work with CNN-based decoders in segmentation_models_pytorch.


"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import DINO models via torch.hub (no extra install needed)
try:
    import torch.hub
    TORCH_HUB_AVAILABLE = True
except ImportError:
    TORCH_HUB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HUB_REPOS = {"v2": "facebookresearch/dinov2", "v3": "facebookresearch/dinov3"}


class DINOEncoder(nn.Module):
    """
    DINO Vision Transformer encoder wrapper for segmentation.
    
    Adapts DINO ViT to provide multi-scale features compatible with
    CNN-based decoders in segmentation_models_pytorch.
    """
    
    # DINO model configurations
    DINO_MODELS = {
        # DINOv2 models (patch size 14x14)
        "dinov2_vits14": {"embed_dim": 384, "num_heads": 6, "num_layers": 12, "params": 21e6, "patch_size": 14, "version": "v2"},
        "dinov2_vitb14": {"embed_dim": 768, "num_heads": 12, "num_layers": 12, "params": 86e6, "patch_size": 14, "version": "v2"},
        "dinov2_vitl14": {"embed_dim": 1024, "num_heads": 16, "num_layers": 24, "params": 300e6, "patch_size": 14, "version": "v2"},
        "dinov2_vitg14": {"embed_dim": 1536, "num_heads": 24, "num_layers": 40, "params": 1100e6, "patch_size": 14, "version": "v2"},
        # DINOv2 with registers (Darcet et al. 2023): the same blocks, and four
        # register tokens between the class token and the patch tokens. Their
        # features come through the backbone's native forward, which places the
        # registers and strips them.
        "dinov2_vits14_reg": {"embed_dim": 384, "num_heads": 6, "num_layers": 12, "params": 21e6, "patch_size": 14, "version": "v2", "num_register_tokens": 4},
        "dinov2_vitb14_reg": {"embed_dim": 768, "num_heads": 12, "num_layers": 12, "params": 86e6, "patch_size": 14, "version": "v2", "num_register_tokens": 4},
        "dinov2_vitl14_reg": {"embed_dim": 1024, "num_heads": 16, "num_layers": 24, "params": 300e6, "patch_size": 14, "version": "v2", "num_register_tokens": 4},
        "dinov2_vitg14_reg": {"embed_dim": 1536, "num_heads": 24, "num_layers": 40, "params": 1100e6, "patch_size": 14, "version": "v2", "num_register_tokens": 4},
        # DINOv3 models (patch size 16x16)
        "dinov3_vitl16": {"embed_dim": 1024, "num_heads": 16, "num_layers": 24, "params": 300e6, "patch_size": 16, "version": "v3"},
        "dinov3_vit7b16": {"embed_dim": 1536, "num_heads": 24, "num_layers": 40, "params": 7000e6, "patch_size": 16, "version": "v3"},
    }
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        in_channels: int = 3,
        depth: int = 5,
        weights: Optional[str] = "dinov2",
        pretrained: bool = True,
        **kwargs
    ):
        """
        Initialize DINO encoder.
        
        Args:
            model_name: DINO model variant
                - DINOv2 variants:
                  - "dinov2_vits14": Small (21M params)
                  - "dinov2_vitb14": Base (86M params) - recommended
                  - "dinov2_vitl14": Large (300M params)
                  - "dinov2_vitg14": Giant (1.1B params)
                - DINOv3 variants:
                  - "dinov3_vitl16": Large (300M params, ViT-L/16)
                  - "dinov3_vit7b16": 7B (7B params, ViT-7B/16) - very large, requires significant GPU memory
            in_channels: Input channels (1 for grayscale, 3 for RGB, N for 2.5D)
            depth: Number of feature levels to extract (typically 4-5)
            weights: Weight source ("dinov2" or "dinov3"). A checkpoint path is
                refused; a fine-tuned model is loaded whole via ``full_weights_path``.
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        
        if model_name not in self.DINO_MODELS:
            raise ValueError(
                f"Unknown DINO model: {model_name}. "
                f"Supported models: {list(self.DINO_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.depth = depth
        self.weights = weights
        self.pretrained = pretrained
        
        # Load DINO model
        self.dino_model = self._load_dino_model(model_name, weights, pretrained)
        
        # Get model configuration
        config = self.DINO_MODELS[model_name]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.patch_size = config.get("patch_size", 14)  # DINOv2 uses 14x14, DINOv3 uses 16x16
        self.dino_version = config.get("version", "v2")  # Track DINO version
        # DINOv3 and the registers variants run their own token path.
        self._native_forward = self.dino_version == "v3" or config.get("num_register_tokens", 0) > 0
        
        # Adapt input channels if needed
        if in_channels != 3:
            self._adapt_input_channels(in_channels)
        
        # Define output channels for each stage
        # SMP convention: out_channels has depth+1 elements, first is "stem"
        stage_channels = self._get_out_channels(depth)
        self._stem_channels = in_channels  # stem = input channels
        self.out_channels = [self._stem_channels] + stage_channels

        # Select layers for feature extraction
        self.selected_layers = self._select_layers(depth)
        # Create projection + upsampling layers to build a multi-scale feature pyramid.
        self._stage_upsample_factors = []
        self.feature_projections = nn.ModuleList()
        for i, out_ch in enumerate(stage_channels):
            # Upsample factor: early stages get 2^(depth-1-i) upsampling
            # e.g. for depth=4: factors are [8, 4, 2, 1]
            factor = 2 ** (depth - 1 - i)
            self._stage_upsample_factors.append(factor)

            # 1x1 channel projection (always needed unless channels match)
            proj_out = out_ch
            if self.embed_dim != out_ch:
                proj = nn.Conv2d(self.embed_dim, out_ch, kernel_size=1)
            else:
                proj = nn.Identity()
                proj_out = self.embed_dim

            if factor > 1:
                # Upsample + 3x3 refinement conv for smoothness
                block = nn.Sequential(
                    proj,
                    nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False),
                    nn.Conv2d(proj_out, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            else:
                # Last stage: just project channels, no upsampling
                block = proj
            self.feature_projections.append(block)
        
        logger.info(
            f"DINOEncoder initialized: {model_name}, "
            f"embed_dim={self.embed_dim}, depth={depth}, "
            f"out_channels={self.out_channels}, "
            f"stage_upsample_factors={self._stage_upsample_factors}"
        )
    
    def _hub_load(self, model_name: str, *, pretrained: bool) -> nn.Module:
        """``torch.hub.load`` of ``model_name`` from its version's repository;
        the hub entry points carry the same names as ``DINO_MODELS``."""
        repo = _HUB_REPOS[self.DINO_MODELS[model_name].get("version", "v2")]
        logger.info("Loading %s from %s (pretrained=%s)", model_name, repo, pretrained)
        return torch.hub.load(repo, model_name, pretrained=pretrained, trust_repo=True)

    def _load_dino_model(self, model_name: str, weights: Optional[str], pretrained: bool) -> nn.Module:
        """The backbone from torch.hub, with Meta's weights when ``pretrained``.
        Whatever fails raises; a randomly initialised backbone is never
        substituted for the pretrained one that was asked for. A checkpoint
        path is refused: it used to load with ``strict=False``, which
        drops a missing tensor silently."""
        if not TORCH_HUB_AVAILABLE:
            raise ImportError(
                "torch.hub is not available. Cannot load DINO models. "
                "Please ensure PyTorch is properly installed."
            )
        if weights and Path(weights).exists():
            raise ValueError(
                f"DINO weights {weights!r} name a file, and a checkpoint no longer initialises a DINO "
                "encoder: it used to load with strict=False, which drops a missing tensor silently. "
                "Set encoder_weights to 'dinov2' or 'dinov3' for the hub-pretrained backbone, or load a "
                "full saved model through full_weights_path."
            )
        return self._hub_load(model_name, pretrained=pretrained)

    @property
    def output_stride(self) -> int:
        """smp encoder protocol: reduction factor of the deepest feature map.

        ``SegmentationModel.check_input_shape`` divides the input size by this
        and refuses non-divisible shapes. Mirrors smp's own base-encoder rule,
        ``min(32, 2 ** depth)``, so a depth-4 DINO encoder behaves like any
        other depth-4 encoder to the decoder above it.
        """
        return min(32, 2 ** self.depth)

    def set_in_channels(
        self, in_channels: int, *, pretrained: bool = True,
    ) -> None:
        """smp encoder protocol: re-key the stem for a different channel count.

        ``smp.encoders.get_encoder`` builds the encoder from the registry's
        static ``params``, which cannot carry a per-call ``in_channels``, then
        calls this. The instance is always freshly built with the default 3
        channels at that point, so this only ever widens or narrows from 3.
        """
        if in_channels == self.in_channels:
            return
        self._adapt_input_channels(in_channels)
        self.in_channels = in_channels
        self._stem_channels = in_channels
        # out_channels[0] is the stem, which smp's decoder uses to size its
        # first skip connection; it must track the real input width.
        self.out_channels = [in_channels] + list(self.out_channels[1:])

    def make_dilated(self, output_stride: int) -> None:
        """smp encoder protocol: unsupported here.

        Only called for ``output_stride != 32``. A ViT's patch stride is fixed,
        so there is no dilation to apply.
        """
        raise ValueError(
            f"DINO encoders have a fixed patch stride of {self.patch_size} and "
            f"cannot be dilated (requested output_stride={output_stride})."
        )

    def _adapt_input_channels(self, in_channels: int):
        """Adapt DINO patch embedding for different input channels."""
        if in_channels == 3:
            return  # No adaptation needed
        
        original_embed = self.dino_model.patch_embed.proj
        
        # Create new patch embedding layer
        new_embed = nn.Conv2d(
            in_channels,
            original_embed.out_channels,
            kernel_size=original_embed.kernel_size,
            stride=original_embed.stride,
            padding=original_embed.padding,
            bias=original_embed.bias is not None
        )
        
        # Initialize weights
        with torch.no_grad():
            if in_channels == 1:
                # For grayscale: average RGB channels
                new_embed.weight.data = original_embed.weight.data.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                # For multi-channel (e.g. 2.5D): use the mean RGB kernel for every input channel.
                # This keeps the weight shape consistent: (out_ch, in_channels, kH, kW)
                rgb_mean = original_embed.weight.data.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)
                new_embed.weight.data = rgb_mean.repeat(1, in_channels, 1, 1)
            else:
                # For 2 channels: use first two RGB channels
                new_embed.weight.data = original_embed.weight.data[:, :in_channels, :, :]
            
            if original_embed.bias is not None:
                new_embed.bias.data = original_embed.bias.data
        
        # Replace the patch embedding layer
        self.dino_model.patch_embed.proj = new_embed
        logger.info(f"Adapted patch embedding from 3 to {in_channels} channels")
    
    def _get_out_channels(self, depth: int) -> List[int]:
        """
        Define output channels for each feature level.
        
        Creates progressively larger channel dimensions:
        - Early layers: smaller channels (fine details)
        - Late layers: larger channels (semantic features)
        """
        base_dim = self.embed_dim
        
        # Create channel progression
        channels = []
        for i in range(depth):
            if i == 0:
                # First level: fine details, smaller channels
                channels.append(base_dim // 4)
            elif i == depth - 1:
                # Last level: semantic features, full dimension
                channels.append(base_dim)
            else:
                # Interpolate between
                ratio = i / (depth - 1)
                ch = int(base_dim // 4 + ratio * (base_dim - base_dim // 4))
                channels.append(ch)
        
        return channels
    
    def _select_layers(self, depth: int) -> List[int]:
        """
        Select transformer layers for feature extraction.
        
        Distributes layer selection across the transformer depth.
        """
        num_layers = self.num_layers
        selected = []
        
        # Distribute selections across layers
        # Include early, middle, and late layers
        for i in range(depth):
            if depth == 1:
                layer_idx = num_layers - 1  # Use last layer
            else:
                # Distribute: early layers for fine details, late for semantics
                ratio = i / (depth - 1)
                # Map to layer indices (skip first layer, use later layers more)
                layer_idx = int(2 + ratio * (num_layers - 3))
                layer_idx = min(layer_idx, num_layers - 1)
            selected.append(layer_idx)
        
        return selected
    
    def _native_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Patch tokens of the selected layers through the backbone's own
        ``get_intermediate_layers``, which runs its native token path (DINOv3's
        RoPE, DINOv2's registers) and returns patch tokens only."""
        kwargs = dict(reshape=False, return_class_token=False, norm=False)
        if self.dino_version == "v3":  # DINOv2's signature has no extra-tokens argument
            kwargs["return_extra_tokens"] = False
        intermediate_outputs = self.dino_model.get_intermediate_layers(x, n=self.selected_layers, **kwargs)
        features = list(intermediate_outputs) if isinstance(intermediate_outputs, tuple) else [intermediate_outputs]
        # Ensure we have exactly depth features: repeat the last one, or read
        # the last layer when none came back.
        while len(features) < self.depth:
            if features:
                features.append(features[-1])
            else:
                last_output = self.dino_model.get_intermediate_layers(x, n=[self.num_layers - 1], **kwargs)
                features.append(last_output[0] if isinstance(last_output, tuple) else last_output)
        return features[:self.depth]

    def _extract_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from multiple DINO transformer layers.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature tensors (B, N, D) where N = num_patches
        """
        B, C, H, W = x.shape
        
        if self._native_forward:
            return self._native_intermediate_features(x)

        # DINOv2: manual extraction (original code)
        # Patch embedding
        x = self.dino_model.patch_embed(x)  # (B, N, D) where N = H' * W'
        
        # Calculate spatial dimensions of patches
        num_patches = x.shape[1]  # N = H_patches * W_patches
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        
        # Add positional embeddings (skip CLS token position)
        # DINO models have CLS token, so pos_embed has shape (1, N+1, D)
        if hasattr(self.dino_model, 'pos_embed'):
            pos_embed = self.dino_model.pos_embed
            
            # Check if we need to interpolate positional embeddings
            if pos_embed.shape[1] == num_patches + 1:
                # Perfect match (has CLS token)
                pos_embed_patches = pos_embed[:, 1:, :]  # Skip CLS token
            elif pos_embed.shape[1] == num_patches:
                # Perfect match (no CLS token)
                pos_embed_patches = pos_embed
            else:
                # Need to interpolate positional embeddings to match new patch count
                import math
                
                # Extract patch embeddings (skip CLS token if present)
                # pos_embed shape is (1, N_orig+1, D) or (1, N_orig, D)
                has_cls = pos_embed.shape[1] > num_patches
                if has_cls:
                    pos_embed_patches_orig = pos_embed[:, 1:, :]  # Skip CLS token: (1, N_orig, D)
                else:
                    pos_embed_patches_orig = pos_embed  # (1, N_orig, D)
                
                original_num_patches = pos_embed_patches_orig.shape[1]
                
                # Infer spatial dimensions from original number of patches
                # Try to find square root first (most DINO models use square inputs)
                sqrt_patches = int(math.sqrt(original_num_patches))
                if sqrt_patches * sqrt_patches == original_num_patches:
                    # Square spatial layout
                    original_h_patches = original_w_patches = sqrt_patches
                else:
                    # Try to find factors (for non-square inputs)
                    # Find closest square root
                    original_h_patches = sqrt_patches
                    original_w_patches = original_num_patches // original_h_patches
                    # If still doesn't match, use square assumption (pad/truncate)
                    if original_h_patches * original_w_patches != original_num_patches:
                        # Use square assumption - this may truncate some patches
                        original_h_patches = original_w_patches = sqrt_patches
                        # Truncate to fit square
                        pos_embed_patches_orig = pos_embed_patches_orig[:, :original_h_patches * original_w_patches, :]
                
                # Reshape to 2D spatial format: (1, H_old, W_old, D)
                pos_embed_2d = pos_embed_patches_orig.view(1, original_h_patches, original_w_patches, -1)
                
                # Interpolate spatial positional embeddings to match new patch dimensions
                # (1, H_old, W_old, D) -> (1, D, H_old, W_old) -> (1, D, H_new, W_new) -> (1, H_new, W_new, D)
                pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)  # (1, D, H_old, W_old)
                pos_embed_interp = F.interpolate(
                    pos_embed_2d,
                    size=(H_patches, W_patches),
                    mode='bilinear',
                    align_corners=False
                )
                pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1)  # (1, H_new, W_new, D)
                pos_embed_patches = pos_embed_interp.reshape(1, H_patches * W_patches, -1)  # (1, N, D)
            
            x = x + pos_embed_patches
        
        # Extract from selected layers
        features = []
        
        for i, block in enumerate(self.dino_model.blocks):
            x = block(x)
            
            # Check if this layer index should be extracted
            if i in self.selected_layers:
                features.append(x.clone())  # (B, N, D) - clone to avoid reference issues
        
        # Ensure we have the right number of features
        # If we didn't get enough (shouldn't happen, but safety check), use the last layer
        while len(features) < self.depth:
            features.append(x.clone())  # Use final layer
        
        # Return exactly depth features
        return features[:self.depth]
    
    def _patch_tokens_to_feature_map(
        self, 
        tokens: torch.Tensor, 
        original_h: int, 
        original_w: int
    ) -> torch.Tensor:
        """
        Convert patch tokens to spatial feature maps.
        
        Args:
            tokens: Patch tokens (B, N, D) where N = num_patches
            original_h: Original image height
            original_w: Original image width
        
        Returns:
            Feature map (B, D, H', W') where H' = original_h // patch_size
        """
        B, N, D = tokens.shape
        
        # Calculate spatial dimensions
        h = original_h // self.patch_size
        w = original_w // self.patch_size
        
        # Handle cases where dimensions don't match exactly
        expected_patches = h * w
        if N != expected_patches:
            # Reshape to match expected dimensions
            # This can happen with padding or different input sizes
            if N > expected_patches:
                # Truncate extra patches
                tokens = tokens[:, :expected_patches, :]
            else:
                # Pad with zeros
                padding = torch.zeros(B, expected_patches - N, D, device=tokens.device, dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding], dim=1)
        
        # Reshape: (B, N, D) -> (B, H', W', D) -> (B, D, H', W')
        tokens = tokens.view(B, h, w, D)
        tokens = tokens.permute(0, 3, 1, 2).contiguous()
        
        return tokens
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from DINO.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature maps at different scales (B, C_i, H_i, W_i)
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size

        # Reflect-pad by one patch_size on each side so that edge patches
        # see real mirrored content instead of zeros. After processing,
        # crop back to the original spatial extent. This eliminates
        # the 1px lighter border and generally improves edge predictions.
        border = patch_size
        x_padded = F.pad(x, (border, border, border, border), mode='reflect')
        H_ext = H + 2 * border
        W_ext = W + 2 * border

        # Pad to be divisible by patch_size (on top of border padding)
        pad_h = (patch_size - (H_ext % patch_size)) % patch_size
        pad_w = (patch_size - (W_ext % patch_size)) % patch_size
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x_padded, (0, pad_w, 0, pad_h), mode='reflect')
        H_proc = H_ext + pad_h
        W_proc = W_ext + pad_w

        # Extract intermediate features from transformer
        token_features = self._extract_intermediate_features(x_padded)

        # Number of patches covering the original (un-bordered) region
        native_h = H // patch_size
        native_w = W // patch_size
        # Offset in patch-grid coordinates for the border region
        border_patches = border // patch_size  # = 1 (since border == patch_size)

        # SMP convention: first element is the "stem" (input-level features).
        feature_maps = [x]  # stem: original input at full resolution

        # Convert patch tokens to spatial feature maps and apply projections
        for idx, tokens in enumerate(token_features):
            feat_map = self._patch_tokens_to_feature_map(tokens, H_proc, W_proc)

            # Crop to the region corresponding to the original input
            # (remove the border padding from both sides in patch-grid space)
            feat_map = feat_map[
                :, :,
                border_patches : border_patches + native_h,
                border_patches : border_patches + native_w,
            ]

            # Apply projection (+ upsampling for early stages)
            feat_map = self.feature_projections[idx](feat_map)
            feature_maps.append(feat_map)

        return feature_maps


# Register DINO encoders with segmentation_models_pytorch
def register_dino_encoders():
    """Register the DINO encoders in ``segmentation_models_pytorch``.

    Makes ``encoder_name="dinov2_vits14"`` resolve anywhere smp builds an
    encoder, so the decoder is sized from the DINO feature pyramid instead of
    a default ResNet's.

    ``pretrained_settings`` is deliberately empty: smp's weight loader expects
    HF-Hub ``repo_id``/``revision`` entries, while DINO weights are fetched by
    :class:`DINOEncoder` itself via ``torch.hub`` during construction. Callers
    therefore pass ``encoder_weights=None`` and still get a pretrained
    backbone; any other value raises from smp naming the valid options.
    """
    # The public registry. Older smp exposed a private ``_encoders``; this is
    # the supported name, and failing loudly here beats silently building a
    # ResNet when a DINO encoder was asked for.
    from segmentation_models_pytorch.encoders import encoders as smp_encoders

    for model_name, config in DINOEncoder.DINO_MODELS.items():
        smp_encoders[model_name] = {
            "encoder": DINOEncoder,
            "pretrained_settings": {},
            "params": {
                "model_name": model_name,
                # Per-name so a v3 backbone is not asked for v2 weights.
                "weights": "dinov3" if config.get("version") == "v3" else "dinov2",
            },
        }

    logger.info(
        "Registered DINO encoders with smp: %s", list(DINOEncoder.DINO_MODELS)
    )


# Auto-register on import. A failure here is worth knowing about: without
# registration, a DINO request silently resolves to smp's default encoder.
try:
    register_dino_encoders()
except Exception:  # noqa: BLE001 - import must not hard-fail on a stale smp
    logger.exception(
        "Could not register DINO encoders with smp; DINO encoder names will "
        "not resolve."
    )
