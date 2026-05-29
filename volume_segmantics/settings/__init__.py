"""
Tests and the trainer use this as the fallback when neither a project-local ``volseg-settings/pipeline.yaml``
nor a legacy ``2d_model_train_settings.yaml`` is present.
"""

from __future__ import annotations

from pathlib import Path

PIPELINE_DEFAULT_PATH: Path = Path(__file__).parent / "pipeline.default.yaml"

__all__ = ["PIPELINE_DEFAULT_PATH"]
