"""
DINO Vision Transformer encoder for segmentation.

This module provides DINO (v1, v2, v3) encoder wrappers that adapt Vision Transformer
models to work with CNN-based decoders in segmentation_models_pytorch.

Based on:
- DINOv1: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021
- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
- DINOv3: Siméoni et al., "DINOv3", 2025 - arXiv:2508.10104
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
            weights: Weight source ("dinov2", "dinov3", "dinov1", or path to checkpoint)
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
        
        # Adapt input channels if needed
        if in_channels != 3:
            self._adapt_input_channels(in_channels)
        
        # Define output channels for each stage
        self.out_channels = self._get_out_channels(depth)
        
        # Select layers for feature extraction
        self.selected_layers = self._select_layers(depth)
        
        # Create projection layers to match reported out_channels
        # DINO outputs embed_dim for all layers, but we report different channel counts
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(self.embed_dim, out_ch, kernel_size=1) if out_ch != self.embed_dim else nn.Identity()
            for out_ch in self.out_channels
        ])
        
        logger.info(
            f"DINOEncoder initialized: {model_name}, "
            f"embed_dim={self.embed_dim}, depth={depth}, "
            f"out_channels={self.out_channels}"
        )
    
    def _load_dino_model(
        self, 
        model_name: str, 
        weights: str, 
        pretrained: bool
    ) -> nn.Module:
        """Load DINO model with pretrained weights."""
        if not TORCH_HUB_AVAILABLE:
            raise ImportError(
                "torch.hub is not available. Cannot load DINO models. "
                "Please ensure PyTorch is properly installed."
            )
        
        # Determine DINO version from model name or weights parameter
        is_dinov3 = model_name.startswith("dinov3_") or weights == "dinov3"
        is_dinov2 = model_name.startswith("dinov2_") or weights == "dinov2"
        
        if is_dinov3 and pretrained:
            # Use torch.hub to load official DINOv3
            logger.info(f"Loading DINOv3 model: {model_name} from torch.hub")
            # Try multiple possible model names
            hub_model_names = [model_name]  # Try original name first
            if model_name == "dinov3_vitl16":
                hub_model_names.extend(["dinov3_vitl", "dinov3_vitl_16"])
            elif model_name == "dinov3_vit7b16":
                hub_model_names.extend(["dinov3_vit7b", "dinov3_vit7b_16"])
            
            last_error = None
            for hub_model_name in hub_model_names:
                try:
                    model = torch.hub.load(
                        'facebookresearch/dinov3',
                        hub_model_name,
                        pretrained=True,
                        trust_repo=True
                    )
                    logger.info(f"Successfully loaded {model_name} from torch.hub (using {hub_model_name})")
                    return model
                except ImportError as e:
                    error_str = str(e).lower()
                    if 'torchmetrics' in error_str:
                        error_msg = (
                            f"DINOv3 requires 'torchmetrics' package which is not installed. "
                            f"Please install it with: pip install torchmetrics"
                        )
                        logger.error(error_msg)
                        raise ImportError(error_msg) from e
                    elif 'termcolor' in error_str:
                        error_msg = (
                            f"DINOv3 requires 'termcolor' package which is not installed. "
                            f"Please install it with: pip install termcolor"
                        )
                        logger.error(error_msg)
                        raise ImportError(error_msg) from e
                    else:
                        last_error = e
                        continue
                except (RuntimeError, AttributeError) as e:
                    if "Cannot find callable" in str(e) or "has no attribute" in str(e):
                        # Try next name
                        last_error = e
                        continue
                    else:
                        # Other error, re-raise
                        raise
                except Exception as e:
                    last_error = e
                    continue
            
            # If we get here, all model names failed
            if last_error:
                logger.warning(f"Failed to load DINOv3 from torch.hub with all attempted names: {last_error}")
                logger.info("Attempting to create model without pretrained weights")
                # Fall through to create model without weights
        elif is_dinov2 and pretrained:
            # Use torch.hub to load official DINOv2
            logger.info(f"Loading DINOv2 model: {model_name} from torch.hub")
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name,
                    pretrained=True,
                    trust_repo=True
                )
                logger.info(f"Successfully loaded {model_name} from torch.hub")
                return model
            except Exception as e:
                logger.warning(f"Failed to load from torch.hub: {e}")
                logger.info("Attempting to create model without pretrained weights")
                # Fall through to create model without weights
        elif Path(weights).exists() if weights else False:
            # Load from local checkpoint
            logger.info(f"Loading DINO model from checkpoint: {weights}")
            model = self._load_from_checkpoint(model_name, weights)
            return model
        
        # Create model without pretrained weights (fallback)
        if not pretrained or (is_dinov3 and pretrained):
            # For DINOv3, even creating without pretrained weights requires torch.hub
            logger.info(f"Creating {model_name} without pretrained weights")
            # Determine which hub to use based on model name
            if model_name.startswith("dinov3_"):
                hub_repo = 'facebookresearch/dinov3'
                # Try multiple possible model names
                hub_model_names = [model_name]  # Try original name first
                if model_name == "dinov3_vitl16":
                    hub_model_names.extend(["dinov3_vitl", "dinov3_vitl_16"])
                elif model_name == "dinov3_vit7b16":
                    hub_model_names.extend(["dinov3_vit7b", "dinov3_vit7b_16"])
            else:
                hub_repo = 'facebookresearch/dinov2'
                hub_model_names = [model_name]
            
            last_error = None
            for hub_model_name in hub_model_names:
                try:
                    model = torch.hub.load(
                        hub_repo,
                        hub_model_name,
                        pretrained=False,
                        trust_repo=True
                    )
                    logger.info(f"Successfully created {model_name} (using {hub_model_name})")
                    return model
                except ImportError as e:
                    error_str = str(e).lower()
                    if 'torchmetrics' in error_str:
                        error_msg = (
                            f"DINOv3 requires 'torchmetrics' package which is not installed. "
                            f"Please install it with: pip install torchmetrics"
                        )
                        logger.error(error_msg)
                        raise ImportError(error_msg) from e
                    elif 'termcolor' in error_str:
                        error_msg = (
                            f"DINOv3 requires 'termcolor' package which is not installed. "
                            f"Please install it with: pip install termcolor"
                        )
                        logger.error(error_msg)
                        raise ImportError(error_msg) from e
                    else:
                        last_error = e
                        continue
                except (RuntimeError, AttributeError) as e:
                    if "Cannot find callable" in str(e) or "has no attribute" in str(e):
                        # Try next name
                        last_error = e
                        continue
                    else:
                        # Other error, re-raise
                        raise
                except Exception as e:
                    last_error = e
                    continue
            
            # If we get here, all model names failed
            if last_error:
                raise RuntimeError(
                    f"Could not create DINO model {model_name}. "
                    f"Tried model names: {hub_model_names}. "
                    f"Last error: {last_error}. Please ensure DINO is available via torch.hub."
                ) from last_error
        else:
            raise RuntimeError(
                f"Could not load pretrained weights for {model_name}. "
                f"Please check your internet connection or provide a local checkpoint path."
            )
    
    def _load_from_checkpoint(self, model_name: str, checkpoint_path: str) -> nn.Module:
        """Load DINO model from local checkpoint file."""
        # Determine which hub to use based on model name
        if model_name.startswith("dinov3_"):
            hub_repo = 'facebookresearch/dinov3'
            hub_model_name = model_name
            if model_name == "dinov3_vitl16":
                hub_model_name = "dinov3_vitl"
            elif model_name == "dinov3_vit7b16":
                hub_model_name = "dinov3_vit7b"
        else:
            hub_repo = 'facebookresearch/dinov2'
            hub_model_name = model_name
        
        # First load the model architecture
        model = torch.hub.load(
            hub_repo,
            hub_model_name,
            pretrained=False,
            trust_repo=True
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {checkpoint_path}")
        
        return model
    
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
                # For multi-channel (2.5D): replicate and average
                # Repeat RGB weights and average
                weight_repeated = original_embed.weight.data.repeat(1, (in_channels + 2) // 3, 1, 1)
                new_embed.weight.data = weight_repeated[:, :in_channels, :, :].mean(dim=1, keepdim=False)
                # Average across the channel dimension
                if in_channels % 3 != 0:
                    # Handle remainder channels by averaging
                    for i in range(3, in_channels, 3):
                        new_embed.weight.data[:, i:i+3, :, :] = (
                            original_embed.weight.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                        )
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
    
    def _extract_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from multiple DINO transformer layers.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature tensors (B, N, D) where N = num_patches
        """
        B, C, H, W = x.shape
        
        # DINOv3 uses get_intermediate_layers which handles RoPE automatically
        if self.dino_version == "v3":
            # Use get_intermediate_layers for DINOv3 (handles RoPE internally)
            # Convert selected layer indices to 1-based or use directly
            # get_intermediate_layers expects layer indices
            layer_indices = self.selected_layers
            
            # Get features from selected layers
            # get_intermediate_layers returns a tuple, and we want patch tokens only
            intermediate_outputs = self.dino_model.get_intermediate_layers(
                x,
                n=layer_indices,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=False
            )
            
            # Convert tuple to list and ensure we have the right number
            if isinstance(intermediate_outputs, tuple):
                features = list(intermediate_outputs)
            else:
                features = [intermediate_outputs]
            
            # Ensure we have exactly depth features
            while len(features) < self.depth:
                # If we got fewer features, repeat the last one
                if features:
                    features.append(features[-1])
                else:
                    # Fallback: get the last layer
                    last_output = self.dino_model.get_intermediate_layers(
                        x,
                        n=[self.num_layers - 1],
                        reshape=False,
                        return_class_token=False,
                        return_extra_tokens=False,
                        norm=False
                    )
                    if isinstance(last_output, tuple):
                        features.append(last_output[0])
                    else:
                        features.append(last_output)
            
            return features[:self.depth]
        
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
        
        # Pad input to be divisible by patch_size (required by DINO)
        patch_size = self.patch_size
        pad_h = (patch_size - (H % patch_size)) % patch_size
        pad_w = (patch_size - (W % patch_size)) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            # Pad: (pad_left, pad_right, pad_top, pad_bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H_padded = H + pad_h
            W_padded = W + pad_w
        else:
            H_padded = H
            W_padded = W
        
        # Extract intermediate features from transformer
        token_features = self._extract_intermediate_features(x)  # List of (B, N, D)
        
        # Convert patch tokens to spatial feature maps and apply projections
        feature_maps = []
        for idx, tokens in enumerate(token_features):
            # Use padded dimensions for conversion
            feat_map = self._patch_tokens_to_feature_map(tokens, H_padded, W_padded)
            
            # Crop back to original dimensions if padding was applied
            if pad_h > 0 or pad_w > 0:
                # Calculate target feature map dimensions based on original input
                # Use floor division to match what the feature map would be without padding
                target_h = H // patch_size
                target_w = W // patch_size
                
                # Crop feature map to match original input dimensions
                # This ensures the feature maps align with the decoder's expectations
                feat_map = feat_map[:, :, :target_h, :target_w]
            
            # Apply projection to match reported out_channels
            feat_map = self.feature_projections[idx](feat_map)
            feature_maps.append(feat_map)
        
        return feature_maps


# Register DINO encoders with segmentation_models_pytorch
def register_dino_encoders():
    """Register DINO encoders with segmentation_models_pytorch."""
    try:
        # Try multiple ways to access the encoder registry
        _encoders = None
        
        # Method 1: Direct import (may not work in newer versions)
        try:
            from segmentation_models_pytorch.encoders import _encoders
        except ImportError:
            pass
        
        # Method 2: Access via encoders module
        if _encoders is None:
            try:
                import segmentation_models_pytorch.encoders as encoders_module
                _encoders = getattr(encoders_module, '_encoders', None)
            except (ImportError, AttributeError):
                pass
        
        # Method 3: Try accessing via get_encoder function's module
        if _encoders is None:
            try:
                from segmentation_models_pytorch.encoders import get_encoder
                import sys
                encoders_module = sys.modules.get('segmentation_models_pytorch.encoders')
                if encoders_module:
                    _encoders = getattr(encoders_module, '_encoders', None)
            except (ImportError, AttributeError):
                pass
        
        # Method 4: Try accessing via __dict__ or private attributes
        if _encoders is None:
            try:
                import segmentation_models_pytorch.encoders as encoders_module
                # Try to find _encoders in the module's __dict__
                if hasattr(encoders_module, '__dict__'):
                    _encoders = encoders_module.__dict__.get('_encoders', None)
            except (ImportError, AttributeError):
                pass
        
        if _encoders is None:
            raise ImportError("Could not access _encoders registry")
        
        # Register each DINO variant
        for model_name, config in DINOEncoder.DINO_MODELS.items():
            # Calculate out_channels for default depth=5 without creating full model
            # This avoids loading weights during registration
            base_dim = config["embed_dim"]
            depth = 5
            default_out_channels = []
            for i in range(depth):
                if i == 0:
                    default_out_channels.append(base_dim // 4)
                elif i == depth - 1:
                    default_out_channels.append(base_dim)
                else:
                    ratio = i / (depth - 1)
                    ch = int(base_dim // 4 + ratio * (base_dim - base_dim // 4))
                    default_out_channels.append(ch)
            
            # Determine pretrained settings based on DINO version
            config = DINOEncoder.DINO_MODELS[model_name]
            dino_version = config.get("version", "v2")
            
            pretrained_settings = {}
            if dino_version == "v2":
                pretrained_settings["dinov2"] = {
                    "url": f"https://dl.fbaipublicfiles.com/dinov2/{model_name}_pretrain.pth",
                    "input_space": "RGB",
                    "input_size": [3, 518, 518],  # DINOv2 default input size
                    "input_range": [0, 1],
                }
            elif dino_version == "v3":
                pretrained_settings["dinov3"] = {
                    "url": f"https://dl.fbaipublicfiles.com/dinov3/{model_name}_pretrain.pth",
                    "input_space": "RGB",
                    "input_size": [3, 518, 518],  # DINOv3 default input size (may vary)
                    "input_range": [0, 1],
                }
            
            _encoders[model_name] = {
                "encoder": DINOEncoder,
                "pretrained_settings": pretrained_settings,
                "params": {
                    "model_name": model_name,
                    "out_channels": tuple(default_out_channels),
                }
            }
        
        logger.info(f"Registered DINO encoders: {list(DINOEncoder.DINO_MODELS.keys())}")
        
    except ImportError:
        # Registration is optional - DINO can be used directly with MultitaskUnet
        # Silently skip registration failure (expected behavior)
        pass


# Auto-register on import (but don't fail if it doesn't work)
# Registration is optional - DINO works directly with MultitaskUnet
try:
    register_dino_encoders()
except Exception:
    # Silently skip registration failure (expected behavior when _encoders is not accessible)
    pass
