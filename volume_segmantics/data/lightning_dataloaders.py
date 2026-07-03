"""LightningDataModule 

Thin wrapper over the existing pipeline-mode dataloader factory in
:mod:`volume_segmantics.data.dataloaders`. Exists so the Lightning
trainer entry point gets a single-method handle for train + val
loaders.

Always routes through the **pipeline-mode** dataloader:
when no project ``pipeline.yaml`` is present, the trainer entry point
synthesises a semantic-only :class:`PipelineConfig` via
:func:`legacy_settings_to_pipeline_config` before constructing this
data module. This means the Lightning path always sees pipeline-mode
dict batches — the LightningModule never has to handle the legacy
tuple/MONAI shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from torch.utils.data import DataLoader

from volume_segmantics.data.dataloaders import get_pipeline_training_dataloaders
from volume_segmantics.data.pipeline_loader import PipelineConfig

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover
    pl = None
    _pl_import_error = exc
else:
    _pl_import_error = None


logger = logging.getLogger(__name__)


class VolSeg2dDataModule(pl.LightningDataModule if pl is not None else object):
    """Builds train and val pipeline-mode dataloaders on demand.

    Parameters
    ----------
    image_dir, label_dir
        Directories of pre-sliced image / mask PNG / TIFF files.
    settings
        Legacy ``SimpleNamespace`` settings.
    pipeline_config
        Parsed :class:`PipelineConfig`. The trainer entry point
        synthesises this from legacy settings via
        :func:`legacy_settings_to_pipeline_config` when no
        ``pipeline.yaml`` is present.
    num_classes
        Number of semantic classes including background.
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        settings: SimpleNamespace,
        pipeline_config: PipelineConfig,
        num_classes: int,
    ) -> None:
        if pl is None:  # pragma: no cover
            raise ImportError(
                "pytorch-lightning is not installed."
            ) from _pl_import_error
        super().__init__()
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.settings = settings
        self.pipeline_config = pipeline_config
        self.num_classes = int(num_classes)

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_loader is not None and self._val_loader is not None:
            return
        # The factory does train/val split and DataLoader construction.
        train_loader, val_loader = get_pipeline_training_dataloaders(
            self.image_dir,
            self.label_dir,
            self.settings,
            self.pipeline_config,
            num_classes=self.num_classes,
        )
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is None:
            self.setup()
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        if self._val_loader is None:
            self.setup()
        return self._val_loader


__all__ = ["VolSeg2dDataModule"]
