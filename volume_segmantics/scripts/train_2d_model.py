import logging
import os
import sys
from datetime import date
from pathlib import Path

import volume_segmantics.utilities.config as cfg
from volume_segmantics.data import (TrainingDataSlicer, get_settings_data)
from volume_segmantics.model import VolSeg2dTrainer
from volume_segmantics.utilities import get_2d_training_parser

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

<<<<<<< HEAD
=======

>>>>>>> c68f176 (optimization using optuna)
def main():
    logging.basicConfig(
        level=logging.INFO, format=cfg.LOGGING_FMT, datefmt=cfg.LOGGING_DATE_FMT
    )
    # Parse args
    parser = get_2d_training_parser()
    args = parser.parse_args()
    data_vols  = getattr(args, cfg.TRAIN_DATA_ARG)
    label_vols = getattr(args, cfg.LABEL_DATA_ARG)
    root_path  = Path(getattr(args, cfg.DATA_DIR_ARG)).resolve()

    # Optimization branch
    # If --optimization is passed, hand off to Optuna and return.
    optimization_config = getattr(args, 'optimization', None)
    if optimization_config is not None:
        try:
            from volume_segmantics.optimization.optuna_optimizer import (
                OptunaOptimizer,
            )
        except ImportError:
            logging.error(
                "Optuna not installed. Run: pip install optuna"
            )
            sys.exit(1)

        if not Path(optimization_config).exists():
            logging.error(
                f"Optimization config not found: {optimization_config}"
            )
            sys.exit(1)

        settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.TRAIN_SETTINGS_FN)

        optimizer = OptunaOptimizer(
            data_paths=data_vols,
            label_paths=label_vols,
            base_config=settings_path,
            optuna_config=optimization_config,
            root_path=root_path,
        )
        study = optimizer.optimize()

        if optimizer.optuna_config.get('visualize', False):
            optimizer.visualize(study)

        return
   
    mode = getattr(args, "mode")
    max_label_no = getattr(args, "max_label_no")
    task2_dir = getattr(args, "task2", None)
    task3_dir = getattr(args, "task3", None)
    unlabeled_data_dir = getattr(args, "unlabeled_data_dir", None)
<<<<<<< HEAD
    print("Mode: ",mode )

    # Check if slicing unlabeled data (mode=slicer and no labels provided)
    is_unlabeled_slicing = (mode == 'slicer' and label_vols is None)

    # Create the settings object
    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.TRAIN_SETTINGS_FN)
    settings = get_settings_data(settings_path)

    # Override unlabeled_data_dir from command line if provided
    if unlabeled_data_dir is not None:
        settings.unlabeled_data_dir = str(Path(unlabeled_data_dir).resolve())
        logging.info(f"Using unlabeled_data_dir from command line: {settings.unlabeled_data_dir}")

    task2_im_out_dir = root_path / "task2"  # dir for task2 imgs
    task3_im_out_dir = root_path / "task3"  # dir for task3 imgs
=======
    print("Mode: ", mode)

    is_unlabeled_slicing = (mode == 'slicer' and label_vols is None)

    settings_path = Path(root_path, cfg.SETTINGS_DIR, cfg.TRAIN_SETTINGS_FN)
    settings = get_settings_data(settings_path)

    if unlabeled_data_dir is not None:
        settings.unlabeled_data_dir = str(Path(unlabeled_data_dir).resolve())
        logging.info(
            f"Using unlabeled_data_dir from command line: "
            f"{settings.unlabeled_data_dir}"
        )

    task2_im_out_dir = root_path / "task2"
    task3_im_out_dir = root_path / "task3"
>>>>>>> c68f176 (optimization using optuna)

    task2_vols = None
    task3_vols = None
    if task2_dir is not None:
        task2_vols = [task2_dir] if isinstance(task2_dir, str) else task2_dir
        settings.task2_dir = str(task2_im_out_dir.resolve())
        if not getattr(settings, "use_multitask", False):
            settings.use_multitask = True
            logging.info(
                "Auto-enabling multitask mode because --task2 was provided"
            )
    if task3_dir is not None:
        task3_vols = [task3_dir] if isinstance(task3_dir, str) else task3_dir
        settings.task3_dir = str(task3_im_out_dir.resolve())
        if not getattr(settings, "use_multitask", False):
            settings.use_multitask = True
<<<<<<< HEAD
            logging.info("Auto-enabling multitask mode because --task3 was provided")
    data_im_out_dir = root_path / settings.data_im_dirname  # dir for data imgs
    seg_im_out_dir = root_path / settings.seg_im_out_dirname  # dir for seg imgs

    if(mode=='slicer'):
=======
            logging.info(
                "Auto-enabling multitask mode because --task3 was provided"
            )

    data_im_out_dir = root_path / settings.data_im_dirname
    seg_im_out_dir  = root_path / settings.seg_im_out_dirname

    if mode == 'slicer':
>>>>>>> c68f176 (optimization using optuna)
        if is_unlabeled_slicing:
            if unlabeled_data_dir:
                unlabeled_output_dir = Path(unlabeled_data_dir)
                if not unlabeled_output_dir.is_absolute():
                    unlabeled_output_dir = root_path / unlabeled_output_dir
            else:
                unlabeled_output_dir = root_path / "unlabeled_data"
            run_unlabeled_slicer(data_vols, unlabeled_output_dir, settings)
        else:
            if label_vols is None:
                logging.error(
                    "Labels are required when slicing labeled data. "
                    "Use --labels to provide label volumes."
                )
                sys.exit(1)
            _, max_label_no = run_slicer(
                data_vols, label_vols, data_im_out_dir, seg_im_out_dir,
                settings, task2_vols, task3_vols,
                task2_im_out_dir, task3_im_out_dir,
            )
    elif mode == 'trainer':
        calculated_max_label_no = _calculate_max_label_no_from_slices(
            seg_im_out_dir
        )
        if max_label_no is not None and max_label_no != 2:
            logging.info(
                f"Using provided max_label_no: {max_label_no} "
                f"(overriding calculated value: {calculated_max_label_no})"
            )
            run_trainer(
                data_im_out_dir, seg_im_out_dir,
                max_label_no, settings, root_path,
            )
        else:
            logging.info(
                f"Using calculated max_label_no: {calculated_max_label_no}"
            )
            run_trainer(
                data_im_out_dir, seg_im_out_dir,
                calculated_max_label_no, settings, root_path,
            )
    else:
        if label_vols is None:
            logging.error(
                "Labels are required for training. "
                "Use --labels to provide label volumes."
            )
            sys.exit(1)
        slicer, max_label_no = run_slicer(
            data_vols, label_vols, data_im_out_dir, seg_im_out_dir,
            settings, task2_vols, task3_vols,
            task2_im_out_dir, task3_im_out_dir,
        )
        run_trainer(
            data_im_out_dir, seg_im_out_dir,
            max_label_no, settings, root_path,
        )
        slicer.clean_up_slices()

def run_unlabeled_slicer(data_vols, unlabeled_output_dir: Path, settings):
<<<<<<< HEAD
    """
    Slice unlabeled volumes into 2D images (no labels required).

    Args:
        data_vols: List of paths to unlabeled data volumes
        unlabeled_output_dir: Directory to save sliced unlabeled images
        settings: Settings object
    """
    logging.info(f"Slicing {len(data_vols)} unlabeled volume(s) to {unlabeled_output_dir}")
    os.makedirs(unlabeled_output_dir, exist_ok=True)

    for count, data_vol_path in enumerate(data_vols):
        logging.info(f"Slicing unlabeled volume {count + 1}/{len(data_vols)}: {data_vol_path}")
        # Create slicer without labels
        slicer = TrainingDataSlicer(data_vol_path, label_vol=None, settings=settings)
        data_prefix = f"unlabeled_data{count}"
        slicer.output_data_slices(unlabeled_output_dir, data_prefix)

    logging.info(f"Unlabeled data slicing complete. Slices saved to: {unlabeled_output_dir}")
    logging.info(f"You can now use this directory with --unlabeled_data_dir when training")
=======
    logging.info(
        f"Slicing {len(data_vols)} unlabeled volume(s) to "
        f"{unlabeled_output_dir}"
    )
    os.makedirs(unlabeled_output_dir, exist_ok=True)

    for count, data_vol_path in enumerate(data_vols):
        logging.info(
            f"Slicing unlabeled volume {count + 1}/{len(data_vols)}: "
            f"{data_vol_path}"
        )
        slicer = TrainingDataSlicer(
            data_vol_path, label_vol=None, settings=settings
        )
        slicer.output_data_slices(
            unlabeled_output_dir, f"unlabeled_data{count}"
        )

    logging.info(
        f"Unlabeled data slicing complete. Slices saved to: "
        f"{unlabeled_output_dir}"
    )
    logging.info(
        "You can now use this directory with --unlabeled_data_dir when training"
    )
>>>>>>> c68f176 (optimization using optuna)


def run_slicer(
    data_vols, label_vols, data_im_out_dir, seg_im_out_dir, settings,
    task2_vols=None, task3_vols=None,
    task2_im_out_dir=None, task3_im_out_dir=None,
):
    if label_vols is None:
        logging.error("Labels are required for labeled data slicing.")
        sys.exit(1)
    if len(data_vols) != len(label_vols):
        logging.error(
            "Number of data volumes and number of label volumes must be equal!"
        )
        sys.exit(1)
<<<<<<< HEAD

=======
>>>>>>> c68f176 (optimization using optuna)
    if task2_vols is not None and len(task2_vols) != len(data_vols):
        logging.error(
            "Number of task2 volumes must equal number of data volumes!"
        )
        sys.exit(1)
    if task3_vols is not None and len(task3_vols) != len(data_vols):
        logging.error(
            "Number of task3 volumes must equal number of data volumes!"
        )
        sys.exit(1)

    max_label_no = 0
    label_codes  = None

    for count, (data_vol_path, label_vol_path) in enumerate(
        zip(data_vols, label_vols)
    ):
        slicer = TrainingDataSlicer(data_vol_path, label_vol_path, settings)
        data_prefix, label_prefix = f"data{count}", f"seg{count}"
        slicer.output_data_slices(data_im_out_dir, data_prefix)
        slicer.output_label_slices(seg_im_out_dir, label_prefix)

        if task2_vols is not None and task2_im_out_dir is not None:
            task2_vol_path = (
                task2_vols[count]
                if isinstance(task2_vols, list)
                else task2_vols
            )
            logging.info(
                f"Slicing task2 (boundary) data volume: {task2_vol_path}"
            )
            task2_slicer = TrainingDataSlicer(
                data_vol_path, task2_vol_path, settings, label_type="task2"
            )
            task2_slicer.output_label_slices(
                task2_im_out_dir, f"task2_{count}"
            )
            if not hasattr(slicer, 'task2_im_out_dir'):
                slicer.task2_im_out_dir = task2_im_out_dir

        if task3_vols is not None and task3_im_out_dir is not None:
            task3_vol_path = (
                task3_vols[count]
                if isinstance(task3_vols, list)
                else task3_vols
            )
            logging.info(f"Slicing task3 data volume: {task3_vol_path}")
            task3_slicer = TrainingDataSlicer(
                data_vol_path, task3_vol_path, settings, label_type="task3"
            )
            task3_slicer.output_label_slices(
                task3_im_out_dir, f"task3_{count}"
            )
            if not hasattr(slicer, 'task3_im_out_dir'):
                slicer.task3_im_out_dir = task3_im_out_dir

        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes  = slicer.codes

    assert label_codes is not None
    print("max_label_no: ", max_label_no)
    return slicer, max_label_no


def _calculate_max_label_no_from_slices(seg_im_out_dir: Path) -> int:
<<<<<<< HEAD
    """Calculate maximum label number from existing label slices.

    Args:
        seg_im_out_dir: Directory containing segmentation label slices

    Returns:
        Maximum label number (number of classes)
    """
=======
>>>>>>> c68f176 (optimization using optuna)
    import imageio
    import numpy as np

    label_files = sorted(list(seg_im_out_dir.glob("*.png")))

    if not label_files:
        raise ValueError(f"No label files found in {seg_im_out_dir}")

<<<<<<< HEAD
    max_label = -1

    sample_size = min(100, len(label_files))
    sample_files = label_files[:sample_size]

    logging.info(f"Scanning {sample_size} label files to determine number of classes...")
=======
    max_label   = -1
    sample_size = min(100, len(label_files))
    sample_files = label_files[:sample_size]

    logging.info(
        f"Scanning {sample_size} label files to determine number of classes..."
    )
>>>>>>> c68f176 (optimization using optuna)

    for label_file in sample_files:
        try:
            label_img = imageio.imread(str(label_file))
            if label_img.size > 0:
                current_max = int(np.max(label_img))
                max_label = max(max_label, current_max)
        except Exception as e:
            logging.warning(f"Could not read {label_file}: {e}")
            continue

<<<<<<< HEAD
    # (since labels are 0-indexed)
    num_classes = max_label + 1

    if num_classes <= 0:
        raise ValueError(f"Could not determine number of classes from label files in {seg_im_out_dir}")

    logging.info(f"Detected {num_classes} classes (labels 0-{max_label}) from label slices")
=======
    num_classes = max_label + 1

    if num_classes <= 0:
        raise ValueError(
            f"Could not determine number of classes from label files "
            f"in {seg_im_out_dir}"
        )

    logging.info(
        f"Detected {num_classes} classes (labels 0-{max_label}) "
        f"from label slices"
    )
>>>>>>> c68f176 (optimization using optuna)
    return num_classes


def run_trainer(
    data_im_out_dir, seg_im_out_dir, max_label_no, settings, root_path
):
    logging.info(f"Setting up trainer with max_label_no: {max_label_no}")
    trainer = VolSeg2dTrainer(
        data_im_out_dir, seg_im_out_dir, max_label_no, settings
    )

    num_cyc_frozen   = settings.num_cyc_frozen
    num_cyc_unfrozen = settings.num_cyc_unfrozen

    use_multitask = getattr(settings, "use_multitask", False)
    if use_multitask:
        import volume_segmantics.utilities.base_data_utils as utils
        original_model_type = utils.get_model_type(settings)
        if original_model_type != utils.ModelType.MULTITASK_UNET:
            logging.warning(
                f"use_multitask is enabled but model type is set to "
                f"'{original_model_type.name}'. Model type will be "
                f"automatically changed to 'MULTITASK_UNET'."
            )
        model_type_name = "MULTITASK_UNET"
    else:
        import volume_segmantics.utilities.base_data_utils as utils
        if isinstance(settings.model["type"], str):
            model_type_enum = utils.get_model_type(settings)
            model_type_name = model_type_enum.name
        else:
            model_type_name = settings.model["type"].name

<<<<<<< HEAD
    model_fn = f"{date.today()}_{model_type_name}_{settings.model_output_fn}.pytorch"
=======
    model_fn  = (
        f"{date.today()}_{model_type_name}_{settings.model_output_fn}.pytorch"
    )
>>>>>>> c68f176 (optimization using optuna)
    model_out = Path(root_path, model_fn)

    if num_cyc_frozen > 0:
        logging.info(
            f"Starting frozen encoder training: {num_cyc_frozen} epochs"
        )
        trainer.train_model(
            model_out, num_cyc_frozen, settings.patience,
            create=True, frozen=True,
        )
        logging.info(
            f"Completed frozen encoder training: {num_cyc_frozen} epochs"
        )
    if num_cyc_unfrozen > 0 and num_cyc_frozen > 0:
        logging.info(
            f"Starting unfrozen encoder training: {num_cyc_unfrozen} epochs"
        )
        trainer.train_model(
            model_out, num_cyc_unfrozen, settings.patience,
            create=False, frozen=False,
        )
        logging.info(
            f"Completed unfrozen encoder training: {num_cyc_unfrozen} epochs"
        )
    elif num_cyc_unfrozen > 0 and num_cyc_frozen == 0:
        logging.info(
            f"Starting unfrozen encoder training (no frozen phase): "
            f"{num_cyc_unfrozen} epochs"
        )
        trainer.train_model(
            model_out, num_cyc_unfrozen, settings.patience,
            create=True, frozen=False,
        )
        logging.info(
            f"Completed unfrozen encoder training: {num_cyc_unfrozen} epochs"
        )

    trainer.output_loss_fig(model_out)
    trainer.output_prediction_figure(model_out)


if __name__ == "__main__":
    main()