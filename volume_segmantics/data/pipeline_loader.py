
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml


logger = logging.getLogger(__name__)


# Schema-known enum sets 


KNOWN_HEAD_NAMES: frozenset = frozenset({
    "semantic", "boundary", "distance", "sdm",
})

#: Loss names known to the registry shipped in B3.B.
KNOWN_LOSS_NAMES: frozenset = frozenset({
    # Semantic / multi-class
    "dice_ce", "dice", "cross_entropy", "combined_ce_dice",
    "class_weighted_dice", "tversky", "generalized_dice",
    # Binary / boundary
    "bce", "boundary_bce", "boundary_dice", "boundary_bce_dice",
    "boundary_loss", "boundary_dou",
    # Distance / SDM
    "distance_l1", "distance_mse", "sdm_l1", "sdm_mse",
})

#: SDM head variants.
KNOWN_SDM_VARIANTS: frozenset = frozenset({"binary", "per_class"})

#: Augmentation backends.
KNOWN_AUGMENTATION_BACKENDS: frozenset = frozenset({
    "auto", "albumentations", "monai",
})

#: Loss-schedule kinds (matches v0.5's three).
KNOWN_LOSS_SCHEDULES: frozenset = frozenset({
    "constant", "linear_warmup", "linear_decay",
})

#: Inference modes registered in B3.F.
KNOWN_INFERENCE_MODES: frozenset = frozenset({
    "single_axis", "multi_axis", "sliding_window", "tta_uncertainty",
})

#: Per-axis 2D producers registered in B3.G.
KNOWN_PER_AXIS_PRODUCERS: frozenset = frozenset({
    "distance_watershed", "semantic_cc",
})

#: Instance-assembly backends shipped in pipeline version.
KNOWN_ASSEMBLY_BACKENDS: frozenset = frozenset({"usegment3d"})

#: Per-axis names accepted by uSegment3D + the per-axis producers.
KNOWN_AXES: frozenset = frozenset({"xy", "xz", "yz"})

#: Schema version this parser understands.
SCHEMA_VERSION: str = "pipeline_v1"


#  Errors 


class PipelineConfigError(ValueError):
    """Raised when a pipeline.yaml fails parse-time shape validation."""


#  Dataclasses 


@dataclass
class HeadConfig:
    """One head's config.

    Attributes
    ----------
    enabled
        Whether the head is built and contributes to the loss.
    out_channels
        Override the head's auto-detected channel count. ``None`` means
        "let the head decide" (semantic -> ``num_classes``; sdm ->
        ``num_classes - 1`` for ``per_class`` variant else ``1``;
        boundary / distance -> ``1``).
    loss
        Loss name from :data:`KNOWN_LOSS_NAMES`. ``None`` means use the
        head's per-head default (semantic -> ``dice_ce``; boundary ->
        ``bce_dice``; distance -> ``distance_l1``; sdm -> ``sdm_l1``).
    loss_weight
        Static weight applied to the head's loss in the calculator's
        weighted sum. Combined multiplicatively with any
        :class:`LossScheduleEntryConfig` ramp.
    extra
        Per-head knobs (``variant``, ``d_clip``, ``width``, etc.). The
        loader carries unknown keys through unchanged so head modules
        can read them via ``HeadConfig.extra``.
    """

    enabled: bool = False
    out_channels: Optional[int] = None
    loss: Optional[str] = None
    loss_weight: float = 1.0
    deep_supervision: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PipelineConfigError(
                f"HeadConfig.enabled must be bool; got "
                f"{type(self.enabled).__name__}"
            )
        if self.out_channels is not None and self.out_channels <= 0:
            raise PipelineConfigError(
                f"HeadConfig.out_channels must be positive when set; "
                f"got {self.out_channels}"
            )
        if self.loss is not None and self.loss not in KNOWN_LOSS_NAMES:
            raise PipelineConfigError(
                f"HeadConfig.loss {self.loss!r} not in known losses "
                f"{sorted(KNOWN_LOSS_NAMES)}"
            )
        if self.loss_weight < 0.0:
            raise PipelineConfigError(
                f"HeadConfig.loss_weight must be non-negative; "
                f"got {self.loss_weight}"
            )


@dataclass
class TargetsConfig:
    """Target-generator parameters."""

    boundary: Dict[str, Any] = field(
        default_factory=lambda: {"width": 3}
    )
    distance: Dict[str, Any] = field(
        default_factory=lambda: {"distance_transform": "edt"}
    )
    sdm: Dict[str, Any] = field(
        default_factory=lambda: {"distance_transform": "edt"}
    )

    def __post_init__(self) -> None:
        if "width" in self.boundary and self.boundary["width"] < 1:
            raise PipelineConfigError(
                f"targets.boundary.width must be >= 1; "
                f"got {self.boundary['width']}"
            )
        for head_name in ("distance", "sdm"):
            block = getattr(self, head_name)
            if block.get("distance_transform", "edt") != "edt":
                raise PipelineConfigError(
                    f"targets.{head_name}.distance_transform: only "
                    f"'edt' is supported "
                    f"{block['distance_transform']!r}"
                )


@dataclass
class TransformSpec:
    """Single augmentation transform spec (Albumentations or MONAI)."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PipelineConfigError(
                "TransformSpec.name must be a non-empty string"
            )


@dataclass
class AugmentationsConfig:
    """Augmentation backend + train-transform list."""

    backend: str = "auto"
    train_transforms: List[TransformSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.backend not in KNOWN_AUGMENTATION_BACKENDS:
            raise PipelineConfigError(
                f"augmentations.backend {self.backend!r} not in "
                f"{sorted(KNOWN_AUGMENTATION_BACKENDS)}"
            )


@dataclass
class EMAScheduleConfig:
    """Linear-ramp EMA decay schedule."""

    alpha_warmup: float = 0.99
    alpha_end: float = 0.999
    warmup_steps: int = 500

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha_warmup <= 1.0:
            raise PipelineConfigError(
                f"ema_teacher.schedule.alpha_warmup must be in [0, 1]; "
                f"got {self.alpha_warmup}"
            )
        if not 0.0 <= self.alpha_end <= 1.0:
            raise PipelineConfigError(
                f"ema_teacher.schedule.alpha_end must be in [0, 1]; "
                f"got {self.alpha_end}"
            )
        if self.warmup_steps < 0:
            raise PipelineConfigError(
                f"ema_teacher.schedule.warmup_steps must be "
                f"non-negative; got {self.warmup_steps}"
            )


@dataclass
class EMATeacherConfig:
    """Mean-teacher SSL config."""

    enabled: bool = False
    schedule: EMAScheduleConfig = field(default_factory=EMAScheduleConfig)
    consistency_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for head_name, weight in self.consistency_weights.items():
            if head_name not in KNOWN_HEAD_NAMES:
                raise PipelineConfigError(
                    f"ema_teacher.consistency_weights: unknown head "
                    f"{head_name!r}; known: {sorted(KNOWN_HEAD_NAMES)}"
                )
            if weight < 0.0:
                raise PipelineConfigError(
                    f"ema_teacher.consistency_weights[{head_name!r}] "
                    f"must be non-negative; got {weight}"
                )


@dataclass
class LossScheduleEntryConfig:
    """One head's loss-weight schedule."""

    schedule: str = "constant"
    start_weight: float = 1.0
    end_weight: float = 1.0
    warmup_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.schedule not in KNOWN_LOSS_SCHEDULES:
            raise PipelineConfigError(
                f"loss_schedule.schedule {self.schedule!r} not in "
                f"{sorted(KNOWN_LOSS_SCHEDULES)}"
            )
        if self.start_weight < 0.0 or self.end_weight < 0.0:
            raise PipelineConfigError(
                f"loss_schedule weights must be non-negative; "
                f"got start={self.start_weight} end={self.end_weight}"
            )
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise PipelineConfigError(
                f"loss_schedule.warmup_fraction must be in [0, 1]; "
                f"got {self.warmup_fraction}"
            )


@dataclass
class PerAxisInstancesConfig:
    """Per-axis 2D instance producer config."""

    enabled: bool = False
    producer: Optional[str] = None  # None = auto-select
    axes: Tuple[str, ...] = ("xy", "xz", "yz")
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.producer is not None and self.producer not in KNOWN_PER_AXIS_PRODUCERS:
            raise PipelineConfigError(
                f"per_axis_instances.producer {self.producer!r} not in "
                f"{sorted(KNOWN_PER_AXIS_PRODUCERS)} (or null for auto)"
            )
        if not self.axes:
            raise PipelineConfigError(
                "per_axis_instances.axes must be a non-empty subset of "
                f"{sorted(KNOWN_AXES)}"
            )
        bad = [a for a in self.axes if a not in KNOWN_AXES]
        if bad:
            raise PipelineConfigError(
                f"per_axis_instances.axes contains unknown axis names "
                f"{bad}; known: {sorted(KNOWN_AXES)}"
            )
        # Coerce list -> tuple for consistency.
        self.axes = tuple(self.axes)


@dataclass
class PredictionConfig:
    """Inference-mode + uncertainty + per-axis-producer wiring."""

    inference_mode: str = "multi_axis"
    uncertainty_provider: Optional[str] = None
    per_axis_instances: PerAxisInstancesConfig = field(
        default_factory=PerAxisInstancesConfig
    )

    def __post_init__(self) -> None:
        if self.inference_mode not in KNOWN_INFERENCE_MODES:
            raise PipelineConfigError(
                f"prediction.inference_mode {self.inference_mode!r} not "
                f"in {sorted(KNOWN_INFERENCE_MODES)}"
            )
        # ``tta_uncertainty`` mode auto-sets the provider when it's null.
        if (
            self.inference_mode == "tta_uncertainty"
            and self.uncertainty_provider is None
        ):
            self.uncertainty_provider = "tta_uncertainty"


@dataclass
class InstanceAssemblyConfig:
    """3D instance-assembly backend config"""

    backend: Optional[str] = None  # None = no instance assembly
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.backend is not None
            and self.backend not in KNOWN_ASSEMBLY_BACKENDS
        ):
            raise PipelineConfigError(
                f"instance_assembly.backend {self.backend!r} not in "
                f"{sorted(KNOWN_ASSEMBLY_BACKENDS)}"
            )


@dataclass
class PipelineConfig:
    """Top-level :file:`pipeline.yaml` aggregator.

    """

    schema_version: str = SCHEMA_VERSION
    heads: Dict[str, HeadConfig] = field(default_factory=dict)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    augmentations: AugmentationsConfig = field(
        default_factory=AugmentationsConfig
    )
    ema_teacher: EMATeacherConfig = field(default_factory=EMATeacherConfig)
    loss_schedule: Dict[str, LossScheduleEntryConfig] = field(
        default_factory=dict
    )
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    instance_assembly: InstanceAssemblyConfig = field(
        default_factory=InstanceAssemblyConfig
    )

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PipelineConfigError(
                f"pipeline.yaml schema_version {self.schema_version!r} "
                f"does not match the {SCHEMA_VERSION!r} this parser "
                f"understands"
            )

        # Head dict validation: every key must be a known head name.
        for head_name in self.heads:
            if head_name not in KNOWN_HEAD_NAMES:
                raise PipelineConfigError(
                    f"heads.{head_name}: unknown head; known: "
                    f"{sorted(KNOWN_HEAD_NAMES)}"
                )

        # At least one head must be enabled.
        if not any(h.enabled for h in self.heads.values()):
            raise PipelineConfigError(
                "at least one head must be enabled in pipeline.yaml; "
                "got: " + ", ".join(
                    f"{n}=disabled" for n in sorted(self.heads)
                )
            )

        # Loss-schedule keys must reference known heads.
        for head_name in self.loss_schedule:
            if head_name not in KNOWN_HEAD_NAMES:
                raise PipelineConfigError(
                    f"loss_schedule.{head_name}: unknown head; known: "
                    f"{sorted(KNOWN_HEAD_NAMES)}"
                )

        # SDM variant lives in HeadConfig.extra; validate when present.
        sdm_cfg = self.heads.get("sdm")
        if sdm_cfg is not None and "variant" in sdm_cfg.extra:
            variant = sdm_cfg.extra["variant"]
            if variant not in KNOWN_SDM_VARIANTS:
                raise PipelineConfigError(
                    f"heads.sdm.variant {variant!r} not in "
                    f"{sorted(KNOWN_SDM_VARIANTS)}"
                )

        # SSL -> at least the semantic head must be enabled.
        if self.ema_teacher.enabled:
            sem = self.heads.get("semantic")
            if sem is None or not sem.enabled:
                raise PipelineConfigError(
                    "ema_teacher.enabled requires heads.semantic.enabled "
                    "to also be true"
                )

        # Per-axis producer head dependencies .
        producer = self.prediction.per_axis_instances.producer
        if producer == "distance_watershed":
            d = self.heads.get("distance")
            if d is None or not d.enabled:
                raise PipelineConfigError(
                    "per_axis_instances.producer='distance_watershed' "
                    "requires heads.distance.enabled=true"
                )
        elif producer == "semantic_cc":
            s = self.heads.get("semantic")
            if s is None or not s.enabled:
                raise PipelineConfigError(
                    "per_axis_instances.producer='semantic_cc' requires "
                    "heads.semantic.enabled=true"
                )
        # producer == None -> auto-selected at use time; no validation.

        # uSegment3D requires per-axis instances to be enabled.
        if self.instance_assembly.backend == "usegment3d":
            if not self.prediction.per_axis_instances.enabled:
                logger.info(
                    "instance_assembly.backend='usegment3d' implicitly "
                    "enables prediction.per_axis_instances"
                )
                self.prediction.per_axis_instances.enabled = True


#  YAML -> PipelineConfig parser 


def _consume_known_keys(
    src: Mapping[str, Any], known: Sequence[str], block_label: str,
) -> Dict[str, Any]:
    """Pop ``known`` keys from a copy of ``src``; return them as a dict.

    Unknown leftover keys at top level are not silently dropped — we
    log a warning so typos surface during development. (At sub-block
    level the policy is "carry through unknown keys as ``extra``" so
    head-specific knobs survive.)
    """
    consumed = {k: src[k] for k in known if k in src}
    return consumed


def _parse_head_config(name: str, src: Mapping[str, Any]) -> HeadConfig:
    """Parse a single heads.<name> block.

    Unknown keys are carried into HeadConfig.extra 
    """
    if not isinstance(src, Mapping):
        raise PipelineConfigError(
            f"heads.{name}: expected a mapping, got {type(src).__name__}"
        )
    KNOWN = {
        "enabled", "out_channels", "loss", "loss_weight", "deep_supervision",
    }
    kwargs: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in src.items():
        if k in KNOWN:
            kwargs[k] = v
        else:
            extra[k] = v
    if extra:
        kwargs["extra"] = extra
    return HeadConfig(**kwargs)


def _parse_transform_list(src: Sequence[Any]) -> List[TransformSpec]:
    out: List[TransformSpec] = []
    for i, item in enumerate(src):
        if not isinstance(item, Mapping):
            raise PipelineConfigError(
                f"augmentations.train_transforms[{i}]: expected mapping"
            )
        if "name" not in item:
            raise PipelineConfigError(
                f"augmentations.train_transforms[{i}]: missing 'name'"
            )
        params = {k: v for k, v in item.items() if k != "name"}
        out.append(TransformSpec(name=item["name"], params=params))
    return out


def parse_pipeline_dict(data: Mapping[str, Any]) -> PipelineConfig:
    """Parse a pre-loaded YAML dict into a :class:`PipelineConfig`.
    """
    if not isinstance(data, Mapping):
        raise PipelineConfigError(
            f"pipeline.yaml top-level must be a mapping; got "
            f"{type(data).__name__}"
        )

    schema_version = data.get("schema_version", SCHEMA_VERSION)

    heads_src = data.get("heads", {})
    heads: Dict[str, HeadConfig] = {
        name: _parse_head_config(name, src) for name, src in heads_src.items()
    }

    targets_src = data.get("targets", {}) or {}
    targets = TargetsConfig(
        boundary=dict(targets_src.get("boundary", {"width": 3})),
        distance=dict(targets_src.get("distance", {"distance_transform": "edt"})),
        sdm=dict(targets_src.get("sdm", {"distance_transform": "edt"})),
    )

    aug_src = data.get("augmentations", {}) or {}
    augmentations = AugmentationsConfig(
        backend=aug_src.get("backend", "auto"),
        train_transforms=_parse_transform_list(
            aug_src.get("train_transforms", []) or []
        ),
    )

    ema_src = data.get("ema_teacher", {}) or {}
    sched_src = ema_src.get("schedule", {}) or {}
    ema_teacher = EMATeacherConfig(
        enabled=ema_src.get("enabled", False),
        schedule=EMAScheduleConfig(**sched_src),
        consistency_weights=dict(ema_src.get("consistency_weights", {})),
    )

    loss_schedule_src = data.get("loss_schedule", {}) or {}
    loss_schedule: Dict[str, LossScheduleEntryConfig] = {}
    for head_name, entry in loss_schedule_src.items():
        loss_schedule[head_name] = LossScheduleEntryConfig(**entry)

    pred_src = data.get("prediction", {}) or {}
    paxis_src = pred_src.get("per_axis_instances", {}) or {}
    per_axis = PerAxisInstancesConfig(
        enabled=paxis_src.get("enabled", False),
        producer=paxis_src.get("producer"),
        axes=tuple(paxis_src.get("axes", ("xy", "xz", "yz"))),
        params=dict(paxis_src.get("params", {})),
    )
    prediction = PredictionConfig(
        inference_mode=pred_src.get("inference_mode", "multi_axis"),
        uncertainty_provider=pred_src.get("uncertainty_provider"),
        per_axis_instances=per_axis,
    )

    asm_src = data.get("instance_assembly", {}) or {}
    instance_assembly = InstanceAssemblyConfig(
        backend=asm_src.get("backend"),
        params=dict(asm_src.get("params", {})),
    )

    return PipelineConfig(
        schema_version=schema_version,
        heads=heads,
        targets=targets,
        augmentations=augmentations,
        ema_teacher=ema_teacher,
        loss_schedule=loss_schedule,
        prediction=prediction,
        instance_assembly=instance_assembly,
    )


def load_pipeline_yaml(path: Union[str, Path]) -> PipelineConfig:
    """Load and parse`pipeline.yaml.

    Raises :class:`PipelineConfigError` on schema violations and
    :class:`FileNotFoundError` if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pipeline.yaml not found at {path}")
    logger.info("Loading pipeline.yaml from %s", path)
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    return parse_pipeline_dict(data)


#  Legacy SimpleNamespace -> PipelineConfig adapter 


def _legacy_loss_name(loss_criterion: str) -> str:
    """Map the legacy loss_criterion string to a registry name.
    """
    LEGACY_TO_REGISTRY = {
        "CombinedCEDiceLoss": "combined_ce_dice",
        "BCEDiceLoss": "combined_ce_dice",   # deprecated alias
        "BCELoss": "bce",
        "DiceLoss": "dice",
        "CrossEntropyLoss": "cross_entropy",
        "GeneralizedDiceLoss": "generalized_dice",
        "TverskyLoss": "tversky",
        "BoundaryDoULoss": "boundary_dou",
        "BoundaryLoss": "boundary_loss",
        "ClassWeightedDiceLoss": "class_weighted_dice",
    }
    return LEGACY_TO_REGISTRY.get(loss_criterion, "combined_ce_dice")


def _legacy_inference_mode(settings: SimpleNamespace) -> str:
    """Map legacy ``quality`` + ``use_sliding_window`` to pipeline version modes."""
    if getattr(settings, "use_sliding_window", False):
        return "sliding_window"
    quality = getattr(settings, "quality", "medium")
    if quality == "low":
        return "single_axis"
    return "multi_axis"


def legacy_settings_to_pipeline_config(
    settings: SimpleNamespace,
) -> PipelineConfig:
    
    # Semantic head from legacy loss_criterion.
    loss_criterion = getattr(settings, "loss_criterion", "CombinedCEDiceLoss")
    semantic = HeadConfig(
        enabled=True,
        loss=_legacy_loss_name(loss_criterion),
        loss_weight=1.0,
    )

    # SSL knobs from legacy ``use_semi_supervised`` / ``use_pseudo_labeling``.
    use_ssl = (
        getattr(settings, "use_semi_supervised", False)
        or getattr(settings, "use_pseudo_labeling", False)
    )
    ema_decay = getattr(settings, "ema_decay", 0.99)
    ema_teacher = EMATeacherConfig(
        enabled=use_ssl,
        schedule=EMAScheduleConfig(
            alpha_warmup=ema_decay,
            alpha_end=ema_decay,
            warmup_steps=0,  # legacy single-decay behaviour
        ),
        consistency_weights={"semantic": 1.0} if use_ssl else {},
    )

    return PipelineConfig(
        heads={"semantic": semantic},
        ema_teacher=ema_teacher,
        prediction=PredictionConfig(
            inference_mode=_legacy_inference_mode(settings),
        ),
    )


#  Registry-aware deferred validation 


def validate_pipeline_config(config: PipelineConfig) -> None:
    """Cross-check config against the populated registries.

    Called by trainer / predictor entry points after import-time
    registrations have run. Raises PipelineConfigError when:

    * Any head name in config.heads is not in the head registry.
    * Any loss name on an enabled head is not in the loss registry.
    * The instance-assembly backend is not registered.
    * Any per-axis producer name is not registered.
    * The inference-mode is not registered.

    """
    # Local imports to break circular dep — pipeline_loader is imported
    # eagerly from data/__init__.py, but the registries are populated
    # by other modules at their own import time.
    from volume_segmantics.data.pipeline_registry import (
        list_heads, list_losses, list_target_generators,
    )

    known_heads = list_heads()
    if known_heads:  # registry has been populated
        for head_name, head_cfg in config.heads.items():
            if head_cfg.enabled and head_name not in known_heads:
                raise PipelineConfigError(
                    f"head {head_name!r} is enabled but not registered; "
                    f"registered: {known_heads}"
                )

    known_losses = list_losses()
    if known_losses:
        for head_name, head_cfg in config.heads.items():
            if head_cfg.enabled and head_cfg.loss is not None:
                if head_cfg.loss not in known_losses:
                    raise PipelineConfigError(
                        f"head {head_name!r} loss {head_cfg.loss!r} "
                        f"not registered; registered: {known_losses}"
                    )

    # Target generators are required only for non-semantic enabled heads.
    known_targets = list_target_generators()
    if known_targets:
        for head_name in ("boundary", "distance", "sdm"):
            head_cfg = config.heads.get(head_name)
            if head_cfg is not None and head_cfg.enabled:
                if head_name not in known_targets:
                    raise PipelineConfigError(
                        f"head {head_name!r} enabled but no target "
                        f"generator registered for it; registered: "
                        f"{known_targets}"
                    )




__all__ = [
    "AugmentationsConfig",
    "EMAScheduleConfig",
    "EMATeacherConfig",
    "HeadConfig",
    "InstanceAssemblyConfig",
    "KNOWN_ASSEMBLY_BACKENDS",
    "KNOWN_AUGMENTATION_BACKENDS",
    "KNOWN_AXES",
    "KNOWN_HEAD_NAMES",
    "KNOWN_INFERENCE_MODES",
    "KNOWN_LOSS_NAMES",
    "KNOWN_LOSS_SCHEDULES",
    "KNOWN_PER_AXIS_PRODUCERS",
    "KNOWN_SDM_VARIANTS",
    "LossScheduleEntryConfig",
    "PerAxisInstancesConfig",
    "PipelineConfig",
    "PipelineConfigError",
    "PredictionConfig",
    "SCHEMA_VERSION",
    "TargetsConfig",
    "TransformSpec",
    "legacy_settings_to_pipeline_config",
    "load_pipeline_yaml",
    "parse_pipeline_dict",
    "validate_pipeline_config",
]
