"""Pipeline-mode multi-head model on any segmentation_models_pytorch architecture.

:class:`~volume_segmantics.model.pipeline_multitask_unet.PipelineMultitaskUnet`
builds the U-Net decoder itself. This module covers the other ``model.type``
values (FPN, DeepLabV3, DeepLabV3+, MA-Net, Linknet, PAN, Segformer, U-Net++
and the vanilla U-Net): it builds the plain architecture with
:func:`~volume_segmantics.model.model_2d.build_architecture` and puts the
pipeline heads where that architecture's segmentation head was, reading the
decoder's output channel count from it. The head outputs are upsampled to the
input size by :meth:`MultitaskSegmentationModel.forward`, so an architecture
whose decoder works at a reduced resolution (FPN, DeepLabV3, PAN, Segformer)
keeps its own output scale.

A checkpoint of such a model records the base architecture as its ``type``
(``"FPN"``, ...) plus a ``heads`` descriptor list; ``create_model_on_device``
rebuilds it from those.
"""
from __future__ import annotations

import logging
from typing import Any, List, Mapping, Tuple

import torch
import torch.nn as nn

import volume_segmantics.utilities.base_data_utils as utils
from volume_segmantics.model.heads import build_head_modules
from volume_segmantics.model.model_2d import (
    MultitaskSegmentationModel,
    build_architecture,
    coerce_model_type,
)
from volume_segmantics.model.pipeline_multitask_unet import check_decoder_sharing
from volume_segmantics.model.vanilla_unet import UNet as VanillaUNet

logger = logging.getLogger(__name__)

#: ``model.type`` values that :class:`PipelineMultitaskUnet` builds instead:
#: the pipeline U-Net itself, and the v0.4 ``Multitask_Unet`` name for it.
UNET_FAMILY = (
    utils.ModelType.U_NET,
    utils.ModelType.PIPELINE_MULTITASK_UNET,
    utils.ModelType.MULTITASK_UNET,
)


class _VanillaEncoder(nn.Module):
    """The vanilla U-Net's contracting path as a feature-list encoder."""

    def __init__(self, net: VanillaUNet) -> None:
        super().__init__()
        self.down_conv1 = net.down_conv1
        self.down_conv2 = net.down_conv2
        self.down_conv3 = net.down_conv3
        self.down_conv4 = net.down_conv4
        self.double_conv = net.double_conv

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)
        x = self.double_conv(x)
        return [skip1, skip2, skip3, skip4, x]


class _VanillaDecoder(nn.Module):
    """The vanilla U-Net's expanding path, up to the 64-channel map its last
    convolution reads."""

    def __init__(self, net: VanillaUNet) -> None:
        super().__init__()
        self.up_conv4 = net.up_conv4
        self.up_conv3 = net.up_conv3
        self.up_conv2 = net.up_conv2
        self.up_conv1 = net.up_conv1

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        skip1, skip2, skip3, skip4, x = features
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        return self.up_conv1(x, skip1)


def _split_base(base: nn.Module) -> Tuple[nn.Module, nn.Module, int]:
    """``(encoder, decoder, decoder output channels)`` of a plain architecture."""
    if isinstance(base, VanillaUNet):
        return _VanillaEncoder(base), _VanillaDecoder(base), int(base.conv_last.in_channels)
    head = getattr(base, "segmentation_head", None)
    conv = head[0] if head is not None and len(head) > 0 else None
    if not hasattr(base, "encoder") or not hasattr(base, "decoder") or not isinstance(conv, nn.Conv2d):
        raise TypeError(
            f"{type(base).__name__} does not expose encoder, decoder and a "
            f"segmentation head; it cannot carry pipeline heads."
        )
    return base.encoder, base.decoder, int(conv.in_channels)


class PipelineMultitaskModel(MultitaskSegmentationModel):
    """Pipeline heads on the decoder of any plain architecture.

    Parameters
    ----------
    heads_cfg
        The parsed ``heads:`` block (``name -> HeadConfig``); only enabled
        heads are built, in dict order, which is the forward output order.
    num_classes
        Semantic class count the heads are sized from.
    architecture
        The ``model.type``: a :class:`ModelType` or its name. The U-Net
        family is refused here; :class:`PipelineMultitaskUnet` builds it.
    encoder_name, encoder_depth, encoder_weights, in_channels
        The architecture's own constructor arguments.
    decoder_sharing
        ``"shared"`` (one decoder read by every head) or ``"separate"`` (one
        decoder per head).
    """

    requires_divisible_input_shape = False

    def __init__(
        self,
        heads_cfg: Mapping[str, Any],
        num_classes: int,
        *,
        architecture,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights=None,
        in_channels: int = 1,
        decoder_sharing: str = "shared",
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        model_type = coerce_model_type(architecture)
        if model_type in UNET_FAMILY:
            raise ValueError(
                f"{model_type.name} is built by PipelineMultitaskUnet, not PipelineMultitaskModel"
            )
        head_names = [name for name, head in heads_cfg.items() if getattr(head, "enabled", True)]
        if not head_names:
            raise ValueError("PipelineMultitaskModel requires at least one enabled head")
        decoder_sharing = check_decoder_sharing(decoder_sharing)

        struct = dict(
            encoder_name=encoder_name,
            encoder_depth=int(encoder_depth or 5),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=1,  # the base's own head is discarded
            **encoder_kwargs,
        )
        base = build_architecture(model_type, struct)
        self.encoder, first_decoder, decoder_out_channels = _split_base(base)
        decoders = [first_decoder]
        if decoder_sharing == "separate":
            # The spare bases exist only for their decoders: no second
            # download of pretrained encoder weights.
            spare = dict(struct, encoder_weights=None)
            for _ in range(len(head_names) - 1):
                decoders.append(_split_base(build_architecture(model_type, spare))[1])
        self.decoders = nn.ModuleList(decoders)
        self.head_to_decoder: List[int] = (
            list(range(len(head_names))) if decoder_sharing == "separate"
            else [0] * len(head_names)
        )

        head_modules = build_head_modules(
            heads_cfg, in_channels=decoder_out_channels, num_classes=int(num_classes), dim=2,
        )
        self.heads = nn.ModuleList(list(head_modules))
        self.head_names: tuple = tuple(
            getattr(h, "name", f"head_{i}") for i, h in enumerate(head_modules)
        )
        self.classification_head = None
        self.architecture = model_type
        self.decoder_sharing = decoder_sharing
        self.decoder_out_channels = decoder_out_channels
        self.name = f"pipeline-multitask-{model_type.name.lower()}-{encoder_name}"
        self.initialize()

    @property
    def num_heads(self) -> int:
        return len(self.heads)


__all__ = ["PipelineMultitaskModel", "UNET_FAMILY"]
