"""
Pipeline-mode multi-head U-Net.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder

from volume_segmantics.model.model_2d import MultitaskSegmentationModel

# Import the DINO escape hatch + availability flag from model_2d.
try:
    from volume_segmantics.model.model_2d import DINO_AVAILABLE
    from volume_segmantics.model.encoders import DINOEncoder
except ImportError:
    DINO_AVAILABLE = False


logger = logging.getLogger(__name__)


def _is_dino(encoder_name: str) -> bool:
    return (
        encoder_name.startswith("dinov2_")
        or encoder_name.startswith("dinov3_")
        or encoder_name.startswith("dinov1_")
    )


class PipelineMultitaskUnet(MultitaskSegmentationModel):
    """Multi-head U-Net for pipeline.yaml-driven training.

    Parameters
    ----------
    head_modules
        Pre-built head :class:`nn.Module` list from
        :func:`build_head_modules`. Forward order = list order.
    encoder_name
        SMP encoder name (or DINO model name like ``"dinov2_vitb14"``).
    encoder_depth
        Number of encoder stages. Default 5; DINO encoders force 4.
    encoder_weights
        ``"imagenet"``, ``"dinov2"``, ``"dinov3"``, or ``None``.
    decoder_use_batchnorm
        BatchNorm in the decoder. ``True``/``False``/``"batchnorm"``/...
    decoder_channels
        Per-stage decoder channel count. Defaults are
        ``(256, 128, 64, 32, 16)`` for depth=5 and
        ``(256, 128, 64, 32)`` for depth=4 (DINO).
    decoder_attention_type
        SMP decoder attention. ``None`` or ``"scse"``.
    in_channels
        Network input channel count. b3 default = 1 (grayscale).
    """

    requires_divisible_input_shape = False

    def __init__(
        self,
        head_modules: Sequence[nn.Module],
        *,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: Union[bool, str] = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
        **encoder_kwargs,
    ) -> None:
        super().__init__()

        if not head_modules:
            raise ValueError(
                "PipelineMultitaskUnet requires at least one head module"
            )
        if encoder_depth is None:
            encoder_depth = 5
        encoder_depth = int(encoder_depth)

        #  Encoder 
        is_dino = _is_dino(encoder_name)
        if is_dino:
            if not DINO_AVAILABLE:
                raise RuntimeError(
                    f"DINO encoder requested ({encoder_name}) but "
                    f"DINOEncoder is not available; check that "
                    f"volume_segmantics.model.encoders imports cleanly"
                )
            dino_depth = encoder_depth if encoder_depth <= 4 else 4
            if encoder_depth != dino_depth:
                logger.info(
                    "Adjusting DINO encoder depth from %d to %d",
                    encoder_depth, dino_depth,
                )
                encoder_depth = dino_depth
            if encoder_name.startswith("dinov3_"):
                weights_source = (
                    "dinov3" if encoder_weights in (None, "None")
                    else encoder_weights
                )
            else:
                weights_source = (
                    "dinov2" if encoder_weights in (None, "None")
                    else encoder_weights
                )
            self.encoder = DINOEncoder(
                model_name=encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=weights_source,
                pretrained=(encoder_weights not in (None, "None", False)),
                **encoder_kwargs,
            )
            # DINO depth=4 wants 4 decoder stages.
            if len(decoder_channels) != encoder_depth:
                if encoder_depth == 4:
                    decoder_channels = (256, 128, 64, 32)
                else:
                    decoder_channels = decoder_channels[:encoder_depth]
                logger.info(
                    "Adjusted decoder_channels to %s for DINO depth=%d",
                    decoder_channels, encoder_depth,
                )
            head_upsampling: Optional[int] = 2
        else:
            self.encoder = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
                **encoder_kwargs,
            )
            head_upsampling = None

        #  Decoder 
        # Single shared decoder. (b3 drops decoder_sharing: separate.)
        if decoder_use_batchnorm is True:
            use_norm: Union[bool, str] = "batchnorm"
        elif decoder_use_batchnorm is False:
            use_norm = False
        elif isinstance(decoder_use_batchnorm, str):
            use_norm = decoder_use_batchnorm
        else:
            use_norm = "batchnorm"

        decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=use_norm,
            add_center_block=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )
        self.decoders = nn.ModuleList([decoder])

        #  Heads 
        # Validate head modules' input-channel expectation matches the
        # decoder's output. Heads built via build_head_modules use
        # in_channels = decoder_channels[-1].
        decoder_out_channels = int(decoder_channels[-1])
        for i, head in enumerate(head_modules):
            # Most b3 heads expose .conv.in_channels directly. If a
            # future head wraps multiple convs, this check is best-
            # effort; we don't make it a hard error.
            if hasattr(head, "conv") and hasattr(head.conv, "in_channels"):
                if head.conv.in_channels != decoder_out_channels:
                    raise ValueError(
                        f"head_modules[{i}] ({type(head).__name__}) was "
                        f"built with in_channels={head.conv.in_channels} "
                        f"but the decoder's last stage outputs "
                        f"{decoder_out_channels} channels"
                    )

        self.heads = nn.ModuleList(list(head_modules))
        # Single shared decoder: every head reads decoder index 0.
        self.head_to_decoder: List[int] = [0] * len(head_modules)

        # No optional classification head in b3.
        self.classification_head = None

        # Stash the canonical head names so callers can verify the
        # forward order without poking into `.heads[i].name`.
        self.head_names: tuple = tuple(
            getattr(h, "name", f"head_{i}") for i, h in enumerate(head_modules)
        )

        # Stash for prediction-time upsampling decisions.
        self._head_upsampling = head_upsampling
        self.name = f"pipeline-multitask-u-{encoder_name}"
        self.initialize()

    #  Forward 
    # Inherits MultitaskSegmentationModel.forward, which:
    #   features = self.encoder(x)
    #   for head_idx, head in enumerate(self.heads):
    #       dec = self.decoders[self.head_to_decoder[head_idx]](features)
    #       mask = head(dec)
    #       if mask.shape mismatches input: bilinear upsample
    #   return tuple(masks)
    #
    # Because all heads share decoder 0, the parent caches the decoder
    # output on the first head and reuses it for the others.

    @property
    def num_heads(self) -> int:
        return len(self.heads)


__all__ = ["PipelineMultitaskUnet"]
