import logging
from pathlib import Path
from typing import Tuple, Optional, Union, Callable, Sequence, Any
from dataclasses import dataclass

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
    initialization as init,
)
from segmentation_models_pytorch.base.utils import is_torch_compiling
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import volume_segmantics.utilities.base_data_utils as utils
from torch.nn import DataParallel
import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.vanilla_unet import UNet as VanillaUNet

# Import DINO encoder (registration happens automatically in dino_encoder.py)
try:
    from volume_segmantics.model.encoders import DINOEncoder
    DINO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"DINO encoder not available: {e}")
    DINO_AVAILABLE = False


@dataclass
class HeadConfig:
    """Configuration for a single segmentation head."""
    classes: int = 1
    activation: Optional[Union[str, Callable]] = None
    decoder_idx: int = 0  # Which decoder this head uses
    kernel_size: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "HeadConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MultitaskSegmentationModel(SegmentationModel):
    """
    Base class for multitask segmentation models with flexible decoder sharing.

    Attributes:
        encoder: The backbone encoder
        decoders: nn.ModuleList of decoder modules
        heads: nn.ModuleList of segmentation head modules
        head_to_decoder: List mapping each head index to its decoder index
    """

    encoder: nn.Module
    decoders: nn.ModuleList
    heads: nn.ModuleList
    head_to_decoder: list
    classification_head: Optional[nn.Module]

    def initialize(self):
        for decoder in self.decoders:
            init.initialize_decoder(decoder)
        for head in self.heads:
            init.initialize_head(head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Validate input shape if required."""
        if hasattr(super(), 'check_input_shape'):
            super().check_input_shape(x)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder, decoders, and heads.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of mask tensors, one per head. If classification_head exists,
            the last element is the classification output.
        """
        if not (torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()):
            self.check_input_shape(x)
        features = self.encoder(x)

        # Cache decoder outputs to avoid redundant computation for shared decoders
        decoder_outputs: dict = {}

        masks = []
        # Store input size for adaptive upsampling (needed for DINO encoders)
        input_h, input_w = x.shape[2], x.shape[3]

        for head_idx, head in enumerate(self.heads):
            dec_idx = self.head_to_decoder[head_idx]
            if dec_idx not in decoder_outputs:
                decoder_outputs[dec_idx] = self.decoders[dec_idx](features)
            mask = head(decoder_outputs[dec_idx])

            # Adaptive upsampling for DINO encoders: upsample to match input size
            # DINO outputs at reduced resolution (input_size // patch_size)
            # Check if mask size doesn't match input size
            if mask.shape[2] != input_h or mask.shape[3] != input_w:
                mask = F.interpolate(
                    mask,
                    size=(input_h, input_w),
                    mode='bilinear',
                    align_corners=False
                )

            masks.append(mask)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return (*masks, labels)

        return tuple(masks)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Inference method. Switches model to eval mode and runs forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of mask tensors (excludes classification output if present)
        """
        if self.training:
            self.eval()

        outputs = self.forward(x)

        # Exclude classification head output if present
        if self.classification_head is not None:
            return outputs[:-1]
        return outputs

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    @property
    def num_decoders(self) -> int:
        return len(self.decoders)


class MultitaskUnet(MultitaskSegmentationModel):
    """
    Flexible Multitask U-Net with configurable decoder sharing.

    This architecture extends the standard U-Net to support multiple output heads,
    where heads can either share decoders or have their own dedicated decoders.

    Args:
        encoder_name: Name of the encoder backbone (e.g., "resnet34", "efficientnet-b0")
        encoder_depth: Number of encoder stages [3-5]. Default is 5.
        encoder_weights: Pretrained weights ("imagenet", None, or other available weights)
        decoder_use_batchnorm: Use BatchNorm in decoder. Options: True, False, "inplace"
        decoder_channels: Channel dimensions for decoder stages
        decoder_attention_type: Attention module in decoder. Options: None, "scse"
        decoder_interpolation_mode: Upsampling mode. Options: "nearest", "bilinear", etc.
        in_channels: Number of input channels (3 for RGB)
        heads_config: List of HeadConfig objects or dicts defining each head:
            - classes: Number of output classes for this head
            - activation: Activation function ("sigmoid", "softmax", None, etc.)
            - decoder_idx: Which decoder this head uses (enables sharing)
            - kernel_size: Kernel size for the segmentation head conv layer
        aux_params: Optional dict for classification head on encoder output
        **kwargs: Additional arguments passed to encoder
    """
    requires_divisible_input_shape = False

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 3,
        heads_config: Optional[Sequence[Union[dict, HeadConfig]]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict,
    ):
        super().__init__()

        # Default to single head if not specified
        if heads_config is None:
            heads_config = [HeadConfig()]

        parsed_heads = []
        for cfg_item in heads_config:
            if isinstance(cfg_item, dict):
                parsed_heads.append(HeadConfig.from_dict(cfg_item))
            else:
                parsed_heads.append(cfg_item)

        if len(parsed_heads) > 3:
            raise ValueError(f"Maximum 3 heads supported, got {len(parsed_heads)}")
        if len(parsed_heads) == 0:
            raise ValueError("At least one head configuration required")

        # Determine decoder indices and validate
        decoder_indices = sorted(set(h.decoder_idx for h in parsed_heads))
        num_decoders = len(decoder_indices)

        if num_decoders > len(parsed_heads):
            raise ValueError(
                f"Number of unique decoders ({num_decoders}) cannot exceed "
                f"number of heads ({len(parsed_heads)})"
            )
        if num_decoders > 3:
            raise ValueError(f"Maximum 3 decoders supported, got {num_decoders}")

        # Remap decoder indices to be contiguous
        idx_remap = {old: new for new, old in enumerate(decoder_indices)}
        self.head_to_decoder = [idx_remap[h.decoder_idx] for h in parsed_heads]

        # Build encoder
        # Check if this is a DINO encoder
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')

        if is_dino:
            # Use DINOEncoder directly (bypasses SMP's get_encoder)
            if not DINO_AVAILABLE:
                raise RuntimeError(
                    f"DINO encoder requested ({encoder_name}) but DINOEncoder is not available. "
                    f"Please ensure volume_segmantics.model.encoders can be imported."
                )

            # DINO encoders typically use depth=4
            dino_depth = encoder_depth if encoder_depth <= 4 else 4
            if encoder_depth != dino_depth:
                logging.info(f"Adjusting DINO encoder depth from {encoder_depth} to {dino_depth}")

            # Determine DINO version and weights source
            if encoder_name.startswith('dinov3_'):
                weights_source = 'dinov3' if encoder_weights in ['dinov3', None, 'None'] else encoder_weights
            else:
                weights_source = 'dinov2' if encoder_weights in ['dinov2', None, 'None'] else encoder_weights

            self.encoder = DINOEncoder(
                model_name=encoder_name,
                in_channels=in_channels,
                depth=dino_depth,
                weights=weights_source,
                pretrained=(encoder_weights not in [None, 'None', False]),
                **kwargs,
            )
            # Update encoder_depth for decoder construction
            encoder_depth = dino_depth
            # Adjust decoder_channels to match encoder_depth if needed
            if len(decoder_channels) != encoder_depth:
                if encoder_depth == 4:
                    # Default for depth=4: (256, 128, 64, 32)
                    decoder_channels = (256, 128, 64, 32)
                    logging.info(f"Adjusting decoder_channels to {decoder_channels} for DINO depth={encoder_depth}")
                else:
                    # Take first encoder_depth elements
                    decoder_channels = decoder_channels[:encoder_depth]
                    logging.info(f"Truncating decoder_channels to {decoder_channels} for encoder_depth={encoder_depth}")
        else:
            # Use standard SMP encoder
            self.encoder = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
                **kwargs,
            )

        # Build decoders
        add_center_block = encoder_name.startswith("vgg")

        # use_norm accepts: bool, str ('batchnorm', 'groupnorm', etc.), or dict
        if decoder_use_batchnorm is True:
            use_norm = 'batchnorm'
        elif decoder_use_batchnorm is False:
            use_norm = False
        elif isinstance(decoder_use_batchnorm, str):
            use_norm = decoder_use_batchnorm
        else:
            use_norm = 'batchnorm'  # default

        decoder_kwargs = dict(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=use_norm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
        )
        # interpolation_mode is not a parameter of UnetDecoder
        self.decoders = nn.ModuleList([
            UnetDecoder(**decoder_kwargs) for _ in range(num_decoders)
        ])

        # Build segmentation heads
        # UnetDecoder with n_blocks outputs decoder_channels[n_blocks-2] channels
        # For n_blocks=4 and decoder_channels=[256, 128, 64, 32], output is 64 (index 2)
        # So we use decoder_channels[encoder_depth-2] or decoder_channels[-2] if lengths match
        if len(decoder_channels) == encoder_depth:
            # When decoder_channels length matches encoder_depth, output is second-to-last
            head_in_channels = decoder_channels[-2]
        else:
            # Fallback: use last element (shouldn't happen with our setup)
            head_in_channels = decoder_channels[-1]

        # Calculate upsampling factor for DINO encoders
        # DINO outputs features at reduced resolution (input_size // patch_size)
        # We need to upsample to match input resolution
        head_upsampling = None
        if is_dino:
            # For DINO, feature maps are at (H // patch_size, W // patch_size)
            # For typical 256×256 input with patch_size=14: 256//14 = 18
            # To get back to ~256, we need ~14x upsampling
            # SegmentationHead upsampling is typically 2^upsampling_factor
            # 2^4 = 16x gets us close (18×16 = 288, close to 256)
            # The head will do adaptive upsampling to match exact input size
            patch_size = getattr(self.encoder, 'patch_size', 14)
            # Calculate approximate upsampling factor: log2(input_size / feature_size)
            # For 256 input and patch_size=14: log2(256/18) ? log2(14.2) ? 3.8, so use 4
            head_upsampling = 4
            logging.info(f"Setting head upsampling to {head_upsampling} for DINO encoder (patch_size={patch_size})")

        self.heads = nn.ModuleList([
            SegmentationHead(
                in_channels=head_in_channels,
                out_channels=cfg.classes,
                activation=cfg.activation,
                kernel_size=cfg.kernel_size,
                upsampling=head_upsampling,
            )
            for cfg in parsed_heads
        ])

        # Optional classification head
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"multitask-u-{encoder_name}"
        self.initialize()


import torch
import torch.nn as nn
import volume_segmantics.utilities.base_data_utils as utils
from torch.nn import DataParallel
import volume_segmantics.utilities.config as cfg

def create_model_on_device(device_num: int, model_struc_dict: dict) -> torch.nn.Module:
    struct_dict_copy = model_struc_dict.copy()
    model_type = struct_dict_copy.pop("type")
    in_channels_requested = struct_dict_copy.get("in_channels", cfg.MODEL_INPUT_CHANNELS)
    encoder_weights = struct_dict_copy.get("encoder_weights", None)

    if model_type == utils.ModelType.U_NET:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')

        if struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:
            model = smp.Unet(**struct_dict_copy, encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
                head_upsampling=2,)
        elif is_dino:
            # DINO encoders: use depth=4 and adjust decoder channels
            # If registration failed, create encoder directly and pass it
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            encoder_weights_param = struct_dict_copy.get('encoder_weights', 'dinov2')
            in_channels_param = struct_dict_copy.get('in_channels', 3)
            # Remove encoder_depth from struct_dict_copy to avoid duplicate keyword argument
            struct_dict_copy.pop('encoder_depth', None)

            # Determine DINO version from encoder name
            if encoder_name.startswith('dinov3_'):
                weights_source = 'dinov3' if encoder_weights_param in ['dinov3', None, 'None'] else encoder_weights_param
            else:
                weights_source = 'dinov2' if encoder_weights_param in ['dinov2', None, 'None'] else encoder_weights_param

            # Try to create model with encoder_name first (if registration worked)
            try:
                model = smp.Unet(**struct_dict_copy, encoder_depth=dino_depth,
                    decoder_channels=(256, 128, 64, 32) if dino_depth == 4 else (256, 128, 64, 32, 16),
                    head_upsampling=2,)
            except (ValueError, KeyError, AttributeError) as e:
                # Registration failed, create encoder directly
                if DINO_AVAILABLE:
                    logging.info(f"DINO encoder not registered with smp, creating directly: {e}")
                    dino_encoder = DINOEncoder(
                        model_name=encoder_name,
                        in_channels=in_channels_param,
                        depth=dino_depth,
                        weights=weights_source,
                        pretrained=(encoder_weights_param not in [None, 'None', False])
                    )
                    # Create model with encoder parameter instead of encoder_name
                    struct_dict_copy_no_encoder = struct_dict_copy.copy()
                    struct_dict_copy_no_encoder.pop('encoder_name', None)
                    struct_dict_copy_no_encoder.pop('encoder_weights', None)
                    model = smp.Unet(
                        encoder=dino_encoder,
                        encoder_depth=dino_depth,
                        decoder_channels=(256, 128, 64, 32) if dino_depth == 4 else (256, 128, 64, 32, 16),
                        head_upsampling=2,
                        **struct_dict_copy_no_encoder
                    )
                else:
                    raise RuntimeError(f"DINO encoder not available and registration failed: {e}")
        else:
            model = smp.Unet(**struct_dict_copy)
        logging.info(f"Sending the U-Net model to device {device_num}")
    elif model_type == utils.ModelType.U_NET_PLUS_PLUS:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')

        if struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:
            model = smp.UnetPlusPlus(**struct_dict_copy, encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
                head_upsampling=2,)
        elif is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            encoder_weights_param = struct_dict_copy.get('encoder_weights', 'dinov2')
            in_channels_param = struct_dict_copy.get('in_channels', 3)
            # Remove encoder_depth from struct_dict_copy to avoid duplicate keyword argument
            struct_dict_copy.pop('encoder_depth', None)

            # Determine DINO version from encoder name
            if encoder_name.startswith('dinov3_'):
                weights_source = 'dinov3' if encoder_weights_param in ['dinov3', None, 'None'] else encoder_weights_param
            else:
                weights_source = 'dinov2' if encoder_weights_param in ['dinov2', None, 'None'] else encoder_weights_param

            # Try to create model with encoder_name first (if registration worked)
            try:
                model = smp.UnetPlusPlus(**struct_dict_copy, encoder_depth=dino_depth,
                    decoder_channels=(256, 128, 64, 32) if dino_depth == 4 else (256, 128, 64, 32, 16),
                    head_upsampling=2,)
            except (ValueError, KeyError, AttributeError) as e:
                # Registration failed, create encoder directly
                if DINO_AVAILABLE:
                    logging.info(f"DINO encoder not registered with smp, creating directly: {e}")
                    dino_encoder = DINOEncoder(
                        model_name=encoder_name,
                        in_channels=in_channels_param,
                        depth=dino_depth,
                        weights=weights_source,
                        pretrained=(encoder_weights_param not in [None, 'None', False])
                    )
                    # Create model with encoder parameter instead of encoder_name
                    struct_dict_copy_no_encoder = struct_dict_copy.copy()
                    struct_dict_copy_no_encoder.pop('encoder_name', None)
                    struct_dict_copy_no_encoder.pop('encoder_weights', None)
                    model = smp.UnetPlusPlus(
                        encoder=dino_encoder,
                        encoder_depth=dino_depth,
                        decoder_channels=(256, 128, 64, 32) if dino_depth == 4 else (256, 128, 64, 32, 16),
                        head_upsampling=2,
                        **struct_dict_copy_no_encoder
                    )
                else:
                    raise RuntimeError(f"DINO encoder not available and registration failed: {e}")
        else:
            model = smp.UnetPlusPlus(**struct_dict_copy)
        logging.info(f"Sending the U-Net++ model to device {device_num}")
    elif model_type == utils.ModelType.FPN:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.FPN(**struct_dict_copy)
        logging.info(f"Sending the FPN model to device {device_num}")
    elif model_type == utils.ModelType.DEEPLABV3:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.DeepLabV3(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3 model to device {device_num}")
    elif model_type == utils.ModelType.DEEPLABV3_PLUS:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.DeepLabV3Plus(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3+ model to device {device_num}")
    elif model_type == utils.ModelType.MA_NET:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.MAnet(**struct_dict_copy)
        logging.info(f"Sending the MA-Net model to device {device_num}")
    elif model_type == utils.ModelType.LINKNET:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.Linknet(**struct_dict_copy)
    elif model_type == utils.ModelType.PAN:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.PAN(**struct_dict_copy)
        logging.info(f"Sending the PAN model to device {device_num}")
    elif model_type == utils.ModelType.SEGFORMER:
        encoder_name = struct_dict_copy.get('encoder_name', '')
        is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
        if is_dino:
            dino_depth = struct_dict_copy.get('encoder_depth', 4)
            struct_dict_copy['encoder_depth'] = dino_depth
        model = smp.Segformer(**struct_dict_copy)
        logging.info(f"Sending the Segformer model to device {device_num}")
    elif model_type == utils.ModelType.VANILLA_UNET:
        in_channels = struct_dict_copy.get("in_channels", 1)
        model = VanillaUNet(in_channels=in_channels, out_classes=struct_dict_copy["classes"], up_sample_mode='conv_transpose')
        logging.info(f"Sending the Vanilla Unet model to device {device_num}")
    elif model_type == utils.ModelType.MULTITASK_UNET:

        heads_config = struct_dict_copy.pop("heads_config", None)
        decoder_sharing = struct_dict_copy.pop("decoder_sharing", "shared")  # "shared" or "separate"
        task_out_channels = struct_dict_copy.pop("task_out_channels", None)  # Remove if present (not used by MultitaskUnet)

        classes = struct_dict_copy.pop("classes", 1)

        if heads_config is None:
            num_tasks = struct_dict_copy.pop("num_tasks", 1)

            if decoder_sharing == "shared":
                # All heads share decoder 0
                heads_config = [
                    {"classes": classes, "decoder_idx": 0, "activation": None}
                ]
                if num_tasks >= 2:
                    # FIX: Use None instead of "sigmoid" - loss function applies sigmoid internally
                    heads_config.append({"classes": 1, "decoder_idx": 0, "activation": None})
                if num_tasks >= 3:
                    heads_config.append({"classes": 1, "decoder_idx": 0, "activation": None})
            else:
                # Each head has its own decoder
                heads_config = [
                    {"classes": classes, "decoder_idx": 0, "activation": None}
                ]
                if num_tasks >= 2:
                    heads_config.append({"classes": 1, "decoder_idx": 1, "activation": None})
                if num_tasks >= 3:
                    heads_config.append({"classes": 1, "decoder_idx": 2, "activation": None})

        model = MultitaskUnet(heads_config=heads_config, **struct_dict_copy)
        logging.info(f"Sending the Multitask U-Net model to device {device_num}")

    # If using pretrained weights with >3 input channels, adapt first conv by averaging RGB.
    # Skip this for DINO encoders as they handle channel adaptation internally
    encoder_name = struct_dict_copy.get('encoder_name', '')
    is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')

    try:
        if encoder_weights and encoder_weights != "None" and in_channels_requested > 3 and not is_dino:
            _adapt_first_conv_from_imagenet_avg(model, model_type, struct_dict_copy, in_channels_requested)
    except Exception as e:
        logging.warning(f"Could not adapt first conv weights for in_channels={in_channels_requested}: {e}")

    if torch.cuda.device_count() > 1 and cfg.USE_ALL_GPUS:
        logging.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = DataParallel(model)
        model.to("cuda")
    else:
        model.to(device_num)
    return model


def _adapt_first_conv_for_channels(model: torch.nn.Module, old_channels: int, new_channels: int) -> None:
    """
    Adapt first conv layer when input channels change.
    Replaces the first Conv2d layer with a new one that has the correct in_channels.

    Args:
        model: The model to adapt
        old_channels: Original number of input channels
        new_channels: New number of input channels
    """
    if not hasattr(model, 'encoder'):
        # Check if model is wrapped in DataParallel
        if hasattr(model, 'module') and hasattr(model.module, 'encoder'):
            encoder = model.module.encoder
            is_wrapped = True
        else:
            return
    else:
        encoder = model.encoder
        is_wrapped = False

    # Find the first Conv2d layer and its parent module
    first_conv = None
    parent_module = None
    conv_name = None

    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            # Get parent module name
            parts = name.split('.')
            if len(parts) > 1:
                parent_name = '.'.join(parts[:-1])
                conv_name = parts[-1]
                parent_module = dict(encoder.named_modules())[parent_name]
            else:
                conv_name = name
                parent_module = encoder
            break

    if first_conv is None:
        return

    current_weight = first_conv.weight.data.clone()
    current_bias = first_conv.bias.data.clone() if first_conv.bias is not None else None
    current_shape = current_weight.shape  # [out_channels, in_channels, H, W]

    if current_shape[1] == new_channels:
        return  # A OK

    # Create new weights
    with torch.no_grad():
        if new_channels > old_channels:
            # Increase channels: repeat existing channels
            if old_channels == 1:
                # Single channel: repeat it (divide by new_channels to maintain similar activation scale)
                new_weight = current_weight.repeat(1, new_channels, 1, 1) / new_channels
            else:
                # Multiple channels: average existing channels and repeat
                mean_weight = current_weight.mean(dim=1, keepdim=True)  # Average across input channels
                new_weight = mean_weight.repeat(1, new_channels, 1, 1)
        else:
            # Decrease channels: take first N channels
            new_weight = current_weight[:, :new_channels, :, :]

    # Create new Conv2d layer with correct in_channels
    new_conv = nn.Conv2d(
        in_channels=new_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode
    )

    # Set weights and bias
    new_conv.weight.data = new_weight.to(current_weight.device).type(current_weight.dtype)
    if current_bias is not None:
        new_conv.bias.data = current_bias.to(current_bias.device).type(current_bias.dtype)

    setattr(parent_module, conv_name, new_conv)

    logging.info(
        f"Adapted first conv layer from {old_channels} to {new_channels} input channels. "
        f"Weight shape: {current_shape} -> {new_weight.shape}"
    )


def _adapt_first_conv_from_imagenet_avg(model: torch.nn.Module, model_type, struct_dict_copy: dict, in_channels_requested: int) -> None:
    """When using pretrained encoders and in_channels > 3, adapt the first conv layer weights by
    averaging the 3-channel Imagenet weights and repeating to match the requested channel count.
    This keeps the rest of the encoder on pretrained weights while providing a better init.

    Note: DINO encoders handle channel adaptation internally, so this function is skipped for them.
    """

    # Skip DINO encoders as they handle channel adaptation internally
    encoder_name = struct_dict_copy.get('encoder_name', '')
    is_dino = encoder_name.startswith('dinov2_') or encoder_name.startswith('dinov3_') or encoder_name.startswith('dinov1_')
    if is_dino:
        return

    if not hasattr(model, 'encoder'):
        return
    encoder = model.encoder
    first_conv = None
    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        return

    # Create a temporary 3-channel model with same architecture to pull pretrained RGB weights
    struct_rgb = struct_dict_copy.copy()
    struct_rgb['in_channels'] = 3
    model_type_local = model_type
    if model_type_local == utils.ModelType.U_NET:
        tmp = smp.Unet(**struct_rgb)
    elif model_type_local == utils.ModelType.U_NET_PLUS_PLUS:
        tmp = smp.UnetPlusPlus(**struct_rgb)
    elif model_type_local == utils.ModelType.FPN:
        tmp = smp.FPN(**struct_rgb)
    elif model_type_local == utils.ModelType.DEEPLABV3:
        tmp = smp.DeepLabV3(**struct_rgb)
    elif model_type_local == utils.ModelType.DEEPLABV3_PLUS:
        tmp = smp.DeepLabV3Plus(**struct_rgb)
    elif model_type_local == utils.ModelType.MA_NET:
        tmp = smp.MAnet(**struct_rgb)
    elif model_type_local == utils.ModelType.LINKNET:
        tmp = smp.Linknet(**struct_rgb)
    elif model_type_local == utils.ModelType.PAN:
        tmp = smp.PAN(**struct_rgb)
    elif model_type_local == utils.ModelType.SEGFORMER:
        tmp = smp.Segformer(**struct_rgb)
    elif model_type_local == utils.ModelType.MULTITASK_UNET:
        # For multitask, create a temporary model with RGB input
        heads_config = struct_dict_copy.get("heads_config", None)
        if heads_config is None:
            num_tasks = struct_dict_copy.get("num_tasks", 1)
            classes = struct_dict_copy.get("classes", 1)
            decoder_sharing = struct_dict_copy.get("decoder_sharing", "shared")
            if decoder_sharing == "shared":
                heads_config = [{"classes": classes, "decoder_idx": 0}]
                if num_tasks >= 2:
                    heads_config.append({"classes": 1, "decoder_idx": 0, "activation": None})
                if num_tasks >= 3:
                    heads_config.append({"classes": 1, "decoder_idx": 0, "activation": None})
            else:
                heads_config = [{"classes": classes, "decoder_idx": 0}]
                if num_tasks >= 2:
                    heads_config.append({"classes": 1, "decoder_idx": 1, "activation": None})
                if num_tasks >= 3:
                    heads_config.append({"classes": 1, "decoder_idx": 2, "activation": None})
        struct_rgb_copy = struct_rgb.copy()
        struct_rgb_copy["heads_config"] = heads_config
        tmp = MultitaskUnet(**struct_rgb_copy)
    else:
        return
    tmp_encoder = tmp.encoder
    tmp_first_conv = None
    for m in tmp_encoder.modules():
        if isinstance(m, nn.Conv2d):
            tmp_first_conv = m
            break
    if tmp_first_conv is None:
        return


    rgb_w = tmp_first_conv.weight.data
    if rgb_w.shape[1] != 3:
        return
    mean_w = rgb_w.mean(dim=1, keepdim=True)
    new_w = mean_w.repeat(1, in_channels_requested, 1, 1)

    with torch.no_grad():
        if first_conv.weight.data.shape == new_w.shape:
            first_conv.weight.data.copy_(new_w.to(first_conv.weight.device).type(first_conv.weight.dtype))
        else:
            # If shapes differ do nothing
            return


def create_model_from_file(
    weights_fn: Path, gpu: bool = True, device_num: int = 0, settings=None,
) -> Tuple[torch.nn.Module, int, dict]:
    """Creates and returns a model and the number of segmentation labels
    that are predicted by the model.

    Args:
        weights_fn: Path to model weights file
        gpu: Whether to use GPU
        device_num: GPU device number
        settings: Optional settings object to override input channels for prediction
    """
    if gpu:
        map_location = f"cuda:{device_num}"
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading model dictionary from file.")
    model_dict = torch.load(weights_fn, map_location=map_location, weights_only=False)

    saved_in_channels = model_dict["model_struc_dict"].get("in_channels", 1)

    requested_in_channels = saved_in_channels
    if settings is not None:
        requested_in_channels = cfg.get_model_input_channels(settings)

    # Create model with saved channels first so we can load the saved weights
    model = create_model_on_device(device_num, model_dict["model_struc_dict"])
    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict["model_state_dict"], strict=False)

    # Adapt first conv layer if input channels changed
    if requested_in_channels != saved_in_channels:
        logging.info(
            f"Adapting model input channels from {saved_in_channels} to {requested_in_channels} "
            f"based on prediction settings (2.5D mode: {getattr(settings, 'use_2_5d_prediction', False)})"
        )
        _adapt_first_conv_for_channels(model, saved_in_channels, requested_in_channels)

        # Update in_channels in model structure dict for future reference
        model_dict["model_struc_dict"]["in_channels"] = requested_in_channels

    return model, model_dict["model_struc_dict"]["classes"], model_dict["label_codes"]


def create_model_from_file_full_weights(
    weights_fn: Path, model_struc_dict : dict, device_num: int = 0, gpu: bool = True,
) -> Tuple[torch.nn.Module, int, dict]:
    """Creates and returns a model and the number of segmentation labels
    that are predicted by the model."""
    if gpu:
        map_location = f"cuda:{device_num}"
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading model dictionary from file.")


    model = create_model_on_device(device_num, model_struc_dict)
    model_dict = torch.load(weights_fn, map_location=map_location, weights_only=False)

    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict, strict=False)
    model.to(device=map_location)
    num_classes = 2 #not used
    return model, num_classes, None
