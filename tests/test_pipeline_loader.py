"""Tests for ``volume_segmantics.data.pipeline_loader``.


"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from volume_segmantics.data.pipeline_loader import (
    AugmentationsConfig,
    EMAScheduleConfig,
    EMATeacherConfig,
    HeadConfig,
    InstanceAssemblyConfig,
    KNOWN_HEAD_NAMES,
    KNOWN_INFERENCE_MODES,
    KNOWN_PER_AXIS_PRODUCERS,
    KNOWN_LOSS_NAMES,
    LossScheduleEntryConfig,
    PerAxisInstancesConfig,
    PipelineConfig,
    PipelineConfigError,
    PredictionConfig,
    SCHEMA_VERSION,
    TargetsConfig,
    TransformSpec,
    legacy_settings_to_pipeline_config,
    load_pipeline_yaml,
    parse_pipeline_dict,
)
from volume_segmantics.settings import PIPELINE_DEFAULT_PATH


# Module-level constants 


def test_schema_version_constant():
    assert SCHEMA_VERSION == "pipeline_v1"


def test_known_head_names_set():
    assert KNOWN_HEAD_NAMES == frozenset(
        {"semantic", "boundary", "distance", "sdm"}
    )


def test_known_inference_modes_set():
    assert KNOWN_INFERENCE_MODES == frozenset({
        "single_axis", "multi_axis", "sliding_window", "tta_uncertainty",
    })


def test_known_per_axis_producers_set():
    assert KNOWN_PER_AXIS_PRODUCERS == frozenset(
        {"distance_watershed", "semantic_cc"}
    )


def test_known_losses_include_per_head_defaults():
    # Defaults referenced in pipeline.default.yaml must all be known.
    for name in (
        "dice_ce", "boundary_bce_dice", "distance_l1", "sdm_l1",
    ):
        assert name in KNOWN_LOSS_NAMES, name


# HeadConfig 


def test_head_config_defaults():
    h = HeadConfig()
    assert h.enabled is False
    assert h.out_channels is None
    assert h.loss is None
    assert h.loss_weight == 1.0
    assert h.deep_supervision is False
    assert h.extra == {}


def test_head_config_unknown_loss_raises():
    with pytest.raises(PipelineConfigError, match="not in known losses"):
        HeadConfig(enabled=True, loss="hallucinated_loss")


def test_head_config_negative_loss_weight_raises():
    with pytest.raises(PipelineConfigError, match="non-negative"):
        HeadConfig(enabled=True, loss_weight=-0.5)


def test_head_config_zero_out_channels_raises():
    with pytest.raises(PipelineConfigError, match="positive when set"):
        HeadConfig(enabled=True, out_channels=0)


def test_head_config_extra_carries_through():
    h = HeadConfig(extra={"d_clip": 8.0, "variant": "binary"})
    assert h.extra["d_clip"] == 8.0
    assert h.extra["variant"] == "binary"


# EMAScheduleConfig 


def test_ema_schedule_defaults_match_v05():
    s = EMAScheduleConfig()
    assert s.alpha_warmup == 0.99
    assert s.alpha_end == 0.999
    assert s.warmup_steps == 500


def test_ema_schedule_rejects_out_of_range_alpha():
    with pytest.raises(PipelineConfigError, match="alpha_warmup"):
        EMAScheduleConfig(alpha_warmup=1.5)
    with pytest.raises(PipelineConfigError, match="alpha_end"):
        EMAScheduleConfig(alpha_end=-0.1)


def test_ema_schedule_rejects_negative_warmup_steps():
    with pytest.raises(PipelineConfigError, match="warmup_steps"):
        EMAScheduleConfig(warmup_steps=-1)


# PerAxisInstancesConfig 


def test_per_axis_default():
    p = PerAxisInstancesConfig()
    assert p.enabled is False
    assert p.producer is None  # auto-select
    assert p.axes == ("xy", "xz", "yz")


def test_per_axis_unknown_producer_raises():
    with pytest.raises(PipelineConfigError, match="not in"):
        PerAxisInstancesConfig(producer="omnipose_flow")  # not shipped in b3


def test_per_axis_empty_axes_raises():
    with pytest.raises(PipelineConfigError, match="non-empty"):
        PerAxisInstancesConfig(axes=())


def test_per_axis_unknown_axis_raises():
    with pytest.raises(PipelineConfigError, match="unknown axis"):
        PerAxisInstancesConfig(axes=("xy", "diagonal"))


def test_per_axis_axes_coerced_to_tuple():
    p = PerAxisInstancesConfig(axes=["xy", "xz"])
    assert p.axes == ("xy", "xz")
    assert isinstance(p.axes, tuple)


# PredictionConfig 


def test_prediction_default_inference_mode():
    p = PredictionConfig()
    assert p.inference_mode == "multi_axis"


def test_prediction_unknown_inference_mode_raises():
    with pytest.raises(PipelineConfigError, match="not in"):
        PredictionConfig(inference_mode="z_only")  # legacy quality, not a mode


def test_tta_uncertainty_auto_sets_provider():
    p = PredictionConfig(inference_mode="tta_uncertainty")
    assert p.uncertainty_provider == "tta_uncertainty"


def test_explicit_uncertainty_provider_preserved():
    p = PredictionConfig(
        inference_mode="multi_axis",
        uncertainty_provider="tta_uncertainty",
    )
    assert p.uncertainty_provider == "tta_uncertainty"


# InstanceAssemblyConfig 


def test_instance_assembly_default_no_backend():
    a = InstanceAssemblyConfig()
    assert a.backend is None


def test_instance_assembly_unknown_backend_raises():
    with pytest.raises(PipelineConfigError, match="not in"):
        InstanceAssemblyConfig(backend="mutex_watershed")  # deferred in b3


def test_instance_assembly_usegment3d_accepted():
    a = InstanceAssemblyConfig(backend="usegment3d")
    assert a.backend == "usegment3d"


# PipelineConfig top-level 


def test_top_level_defaults_require_at_least_one_enabled_head():
    with pytest.raises(PipelineConfigError, match="at least one head"):
        PipelineConfig()  # no heads dict at all


def test_top_level_all_heads_disabled_raises():
    with pytest.raises(PipelineConfigError, match="at least one head"):
        PipelineConfig(heads={"semantic": HeadConfig(enabled=False)})


def test_minimal_semantic_only_config_valid():
    cfg = PipelineConfig(
        heads={"semantic": HeadConfig(enabled=True, loss="dice_ce")},
    )
    assert cfg.heads["semantic"].enabled
    assert cfg.schema_version == "pipeline_v1"


def test_unknown_head_in_dict_raises():
    with pytest.raises(PipelineConfigError, match="unknown head"):
        PipelineConfig(
            heads={"flow": HeadConfig(enabled=True, loss="distance_l1")},
        )


def test_unknown_head_in_loss_schedule_raises():
    with pytest.raises(PipelineConfigError, match="unknown head"):
        PipelineConfig(
            heads={"semantic": HeadConfig(enabled=True)},
            loss_schedule={"affinity": LossScheduleEntryConfig()},
        )


def test_sdm_unknown_variant_raises():
    with pytest.raises(PipelineConfigError, match="variant"):
        PipelineConfig(
            heads={
                "semantic": HeadConfig(enabled=True),
                "sdm": HeadConfig(
                    enabled=True, loss="sdm_l1",
                    extra={"variant": "garbage"},
                ),
            },
        )


def test_ema_teacher_without_semantic_raises():
    with pytest.raises(PipelineConfigError, match="ema_teacher"):
        PipelineConfig(
            heads={"boundary": HeadConfig(enabled=True, loss="bce")},
            ema_teacher=EMATeacherConfig(enabled=True),
        )


def test_distance_watershed_without_distance_head_raises():
    with pytest.raises(PipelineConfigError, match="distance_watershed"):
        PipelineConfig(
            heads={"semantic": HeadConfig(enabled=True)},
            prediction=PredictionConfig(
                per_axis_instances=PerAxisInstancesConfig(
                    enabled=True, producer="distance_watershed",
                ),
            ),
        )


def test_semantic_cc_without_semantic_head_raises():
    with pytest.raises(PipelineConfigError, match="semantic_cc"):
        PipelineConfig(
            heads={"boundary": HeadConfig(enabled=True, loss="bce")},
            prediction=PredictionConfig(
                per_axis_instances=PerAxisInstancesConfig(
                    enabled=True, producer="semantic_cc",
                ),
            ),
        )


def test_usegment3d_implicitly_enables_per_axis_instances():
    cfg = PipelineConfig(
        heads={
            "semantic": HeadConfig(enabled=True),
            "distance": HeadConfig(enabled=True, loss="distance_l1"),
        },
        instance_assembly=InstanceAssemblyConfig(backend="usegment3d"),
    )
    assert cfg.prediction.per_axis_instances.enabled is True


def test_unknown_schema_version_raises():
    with pytest.raises(PipelineConfigError, match="schema_version"):
        PipelineConfig(
            schema_version="pipeline_v999",
            heads={"semantic": HeadConfig(enabled=True)},
        )


# parse_pipeline_dict 


def test_parse_minimal_dict():
    cfg = parse_pipeline_dict({
        "schema_version": "pipeline_v1",
        "heads": {"semantic": {"enabled": True, "loss": "dice_ce"}},
    })
    assert cfg.heads["semantic"].enabled
    assert cfg.heads["semantic"].loss == "dice_ce"


def test_parse_extra_head_keys_carry_to_extra():
    cfg = parse_pipeline_dict({
        "heads": {
            "sdm": {
                "enabled": True, "loss": "sdm_l1",
                "variant": "binary", "d_clip": 8.0,
            },
            "semantic": {"enabled": True},
        },
    })
    sdm = cfg.heads["sdm"]
    assert sdm.extra == {"variant": "binary", "d_clip": 8.0}


def test_parse_full_config_yaml_string():
    yaml_text = """
    schema_version: pipeline_v1
    heads:
      semantic:
        enabled: true
        loss: dice_ce
        loss_weight: 1.0
      boundary:
        enabled: true
        loss: boundary_bce_dice
        loss_weight: 0.5
        width: 3
      distance:
        enabled: true
        loss: distance_l1
        d_clip: 10.0
      sdm:
        enabled: true
        loss: sdm_l1
        variant: per_class
        d_clip: 10.0
    augmentations:
      backend: albumentations
      train_transforms:
        - { name: HorizontalFlip, p: 0.5 }
        - { name: Rotate, limit: 30, p: 0.5 }
    ema_teacher:
      enabled: true
      schedule:
        alpha_warmup: 0.95
        alpha_end: 0.999
        warmup_steps: 200
      consistency_weights: { semantic: 0.5, boundary: 0.5 }
    loss_schedule:
      boundary: { schedule: linear_warmup, start_weight: 0.0,
                  end_weight: 1.0, warmup_fraction: 0.1 }
    prediction:
      inference_mode: tta_uncertainty
      per_axis_instances:
        enabled: true
        producer: distance_watershed
        axes: [xy, xz, yz]
        params: { peak_min_distance: 5 }
    instance_assembly:
      backend: usegment3d
      params: { axes: [xy, xz, yz] }
    """
    cfg = parse_pipeline_dict(yaml.safe_load(yaml_text))

    assert all(cfg.heads[n].enabled for n in
               ("semantic", "boundary", "distance", "sdm"))
    assert cfg.heads["sdm"].extra["variant"] == "per_class"
    assert cfg.heads["distance"].extra["d_clip"] == 10.0
    assert cfg.augmentations.backend == "albumentations"
    assert len(cfg.augmentations.train_transforms) == 2
    assert cfg.augmentations.train_transforms[0].name == "HorizontalFlip"
    assert cfg.augmentations.train_transforms[0].params == {"p": 0.5}
    assert cfg.ema_teacher.enabled
    assert cfg.ema_teacher.schedule.warmup_steps == 200
    assert cfg.loss_schedule["boundary"].schedule == "linear_warmup"
    assert cfg.prediction.inference_mode == "tta_uncertainty"
    assert cfg.prediction.uncertainty_provider == "tta_uncertainty"
    assert cfg.prediction.per_axis_instances.producer == "distance_watershed"
    assert cfg.instance_assembly.backend == "usegment3d"


def test_parse_top_level_not_mapping_raises():
    with pytest.raises(PipelineConfigError, match="must be a mapping"):
        parse_pipeline_dict([1, 2, 3])  # type: ignore[arg-type]


def test_transform_spec_missing_name_raises():
    with pytest.raises(PipelineConfigError, match="missing 'name'"):
        parse_pipeline_dict({
            "heads": {"semantic": {"enabled": True}},
            "augmentations": {
                "train_transforms": [{"p": 0.5}],  # name omitted
            },
        })


# load_pipeline_yaml 


def test_load_packaged_default():
    cfg = load_pipeline_yaml(PIPELINE_DEFAULT_PATH)
    assert cfg.schema_version == "pipeline_v1"
    # The packaged default is semantic-only.
    assert cfg.heads["semantic"].enabled
    assert cfg.heads["boundary"].enabled is False
    assert cfg.heads["distance"].enabled is False
    assert cfg.heads["sdm"].enabled is False
    # No instance assembly by default.
    assert cfg.instance_assembly.backend is None
    # SSL off.
    assert cfg.ema_teacher.enabled is False


def test_load_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_pipeline_yaml(Path("/no/such/pipeline.yaml"))


def test_load_invalid_yaml_propagates_parse_error(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: pipeline_v1\n"
        "heads:\n"
        "  flow:\n"           # unknown head
        "    enabled: true\n"
        "    loss: dice_ce\n"
    )
    with pytest.raises(PipelineConfigError, match="unknown head"):
        load_pipeline_yaml(bad)


# Legacy adapter 


def test_legacy_adapter_minimal_settings():
    settings = SimpleNamespace(
        loss_criterion="CombinedCEDiceLoss",
        quality="medium",
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert list(cfg.heads.keys()) == ["semantic"]
    assert cfg.heads["semantic"].enabled
    assert cfg.heads["semantic"].loss == "combined_ce_dice"
    assert cfg.prediction.inference_mode == "multi_axis"
    assert cfg.ema_teacher.enabled is False
    assert cfg.instance_assembly.backend is None


def test_legacy_adapter_low_quality_maps_to_single_axis():
    settings = SimpleNamespace(
        loss_criterion="DiceLoss",
        quality="low",
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.prediction.inference_mode == "single_axis"
    assert cfg.heads["semantic"].loss == "dice"


def test_legacy_adapter_z_only_maps_to_multi_axis():
    settings = SimpleNamespace(
        loss_criterion="CombinedCEDiceLoss",
        quality="z_only",
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.prediction.inference_mode == "multi_axis"


def test_legacy_adapter_sliding_window():
    settings = SimpleNamespace(
        loss_criterion="CombinedCEDiceLoss",
        quality="medium",
        use_sliding_window=True,
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.prediction.inference_mode == "sliding_window"


def test_legacy_adapter_propagates_ssl_flags():
    settings = SimpleNamespace(
        loss_criterion="CombinedCEDiceLoss",
        quality="medium",
        use_semi_supervised=True,
        ema_decay=0.97,
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.ema_teacher.enabled is True
    assert cfg.ema_teacher.schedule.alpha_warmup == 0.97
    assert cfg.ema_teacher.schedule.alpha_end == 0.97
    assert cfg.ema_teacher.schedule.warmup_steps == 0
    assert cfg.ema_teacher.consistency_weights == {"semantic": 1.0}


def test_legacy_adapter_pseudo_labeling_alone_enables_ema_teacher():
    """``use_pseudo_labeling`` alone is enough — pseudo-labeling needs the
    teacher path even when Mean Teacher is off."""
    settings = SimpleNamespace(
        loss_criterion="CombinedCEDiceLoss",
        quality="medium",
        use_pseudo_labeling=True,
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.ema_teacher.enabled is True


def test_legacy_adapter_unknown_loss_falls_back_to_combined_ce_dice():
    settings = SimpleNamespace(
        loss_criterion="MysteryLoss",  # not in the legacy -> registry map
        quality="medium",
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.heads["semantic"].loss == "combined_ce_dice"


def test_legacy_adapter_bce_dice_loss_aliased_to_combined_ce_dice():
    """v0.4's ``BCEDiceLoss`` is the deprecated name for
    ``CombinedCEDiceLoss``; the adapter aliases them."""
    settings = SimpleNamespace(
        loss_criterion="BCEDiceLoss",
        quality="medium",
    )
    cfg = legacy_settings_to_pipeline_config(settings)
    assert cfg.heads["semantic"].loss == "combined_ce_dice"


# Public re-exports from data/__init__.py 


def test_public_data_module_reexports():
    from volume_segmantics.data import (
        PipelineConfig as PC,
        PipelineConfigError as PCE,
        legacy_settings_to_pipeline_config as adapt,
        load_pipeline_yaml as loader,
        parse_pipeline_dict as parser,
    )
    assert PC is PipelineConfig
    assert PCE is PipelineConfigError
    assert adapt is legacy_settings_to_pipeline_config
    assert loader is load_pipeline_yaml
    assert parser is parse_pipeline_dict
