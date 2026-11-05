# Volume Segmantics

A toolkit for semantic segmentation of volumetric data using PyTorch deep learning models.

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04691/status.svg)](https://doi.org/10.21105/joss.04691)

Volume Segmantics provides a command-line interface and a Python API for training 2D PyTorch segmentation models (U-Net, U-Net++, FPN, DeepLabV3+, Segformer and others) on 3D image volumes, and for using those models to segment larger volumes. Training slices the volume along the x, y and z axes; prediction segments each slice and merges the predictions made in orthogonal planes and rotations, which reduces the artefacts of segmenting 3D data with a 2D network.

The method is optimised for small training datasets, for example a single labelled volume of between $128^3$ and $512^3$ voxels. Every model uses a pre-trained encoder, and image augmentation expands the effective size of the training set.

The library builds on [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the model architectures and encoders, [Albumentations](https://albumentations.ai/) and [MONAI](https://github.com/project-monai/monai) for augmentation, [PyTorch Lightning](https://lightning.ai/) for training, and the loss functions and metrics of Adrian Wolny's [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).

## Requirements

A machine with a CUDA-capable NVIDIA GPU and a recent PyTorch. The package declares Python 3.10 to 3.12 and PyTorch 2.5.1 or later. On Windows you will also need the Visual Studio Build Tools and the CUDA Toolkit; see [the CUDA docs](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). Training and prediction fall back to the CPU when no CUDA device is available, which is only practical for small tests.

## Installation

### Conda or virtualenv

The release version of Volume Segmantics (0.4.0) is available on [PyPI](https://pypi.org/project/volume-segmantics/). Create a new Conda or Miniconda environment (Python 3.12 is recommended), activate it, then:

```
pip install volume-segmantics
```

Two optional extras add features that pull in heavier dependencies:

```
pip install "volume-segmantics[optuna]"       # Optuna hyperparameter search (model-train-2d --optuna)
pip install "volume-segmantics[usegment3d]"   # u-Segment3D instance assembly (model-predict-pipeline)
```

To install the latest development version instead, clone the repository, change into the `volume-segmantics` directory and run:

```
pip install poetry
poetry install
```

To check that the installed PyTorch is a CUDA build, start a Python interpreter in the environment and run:

```
import torch; torch.__version__
```

The version should carry a `cu` suffix, for example `2.5.1+cu124`. If it ends in `cpu`, install a PyTorch build that matches your CUDA version from https://pytorch.org/get-started/locally/.

### Docker or Apptainer container (quay.io)

A container image with the latest published release is available on the Rosalind Franklin Institute's quay.io instance. Pull and run it with Apptainer:

```
apptainer run --nv docker://quay.io/rosalindfranklininstitute/volume-segmantics
```

or with Docker:

```
docker run \
    --gpus all \
    --ipc=host \
    -v /path/to/data:/data \
    quay.io/rosalindfranklininstitute/volume-segmantics
```

`--ipc=host` gives the container access to host shared memory; Docker's default allocation of 64 MB can be too small for the data loaders. More details are in the [Volume EM Container documentation](https://rosalindfranklininstitute.github.io/volume-em-container-documentation/software/volume-segmantics/).

## Configuration and command-line use

After installation three commands are available while the environment is active: `model-train-2d`, `model-predict-2d` and `model-predict-pipeline`.

They read their settings from YAML files in a directory named `volseg-settings` inside the directory you run them from, or inside the directory given by `--data_dir`. Copy the files from [volseg-settings](https://github.com/rosalindfranklininstitute/volume-segmantics/blob/main/volseg-settings) in this repository as a starting point:

| File | Purpose |
| --- | --- |
| `2d_model_train_settings.yaml` | Everything about training: data layout, normalisation, model and encoder, learning rate, loss, evaluation metric, 2.5D slicing. |
| `2d_model_predict_settings.yaml` | Everything about prediction: quality, outputs, normalisation, 2.5D prediction, sliding-window inference. |
| `pipeline.yaml` (optional) | The multi-head pipeline: which heads to train, their losses and schedules, the augmentation backend, the EMA teacher, the inference mode and instance assembly. When it is absent a pipeline configuration is derived from the training settings with a semantic head only. A commented two-head example is `volseg-settings/pipeline_multitask_example.yaml`; see [Multi-task training](#multi-task-training-with-pipelineyaml). |
| `optuna_config.yaml` (optional) | The search space for `model-train-2d --optuna`. |

Check the normalisation settings carefully for your data; different datasets need different choices, and the prediction settings must match the ones used for training.

### Training a 2D model on a 3D volume and its labels

Input volumes can be HDF5, multi-page TIFF or MRC files; label volumes can be HDF5 or TIFF.

```shell
model-train-2d --data path/to/image/data.tiff --labels path/to/corresponding/segmentation/labels.tiff
```

Several volumes can be given after `--data` and the matching labels after `--labels`. Training follows `volseg-settings/2d_model_train_settings.yaml` and writes into the working directory, or into `--data_dir`:

- the model checkpoint, a `.pytorch` file named with the date, model type and `model_output_fn` from the settings, holding the epoch with the best validation foreground Dice;
- the final epoch's weights beside it with a `_final.pth` suffix;
- a loss plot, a per-epoch statistics CSV, and a figure comparing ground-truth and predicted segmentations for some validation images;
- the sliced 2D training images in `data/` and `seg/`, which are deleted after training unless the environment variable `VOLSEG_KEEP_SLICES=1` is set.

Each checkpoint records the image size and the 2.5D stack it was trained with, so prediction can check its settings against them.

Options of `model-train-2d`:

| Option | Effect |
| --- | --- |
| `--data_dir PATH` | Directory holding `volseg-settings` and receiving the outputs (default: the working directory). |
| `--mode slicer` / `--mode trainer` | Run only the slicing step, or only the training step on slices already on disk. Slicing without `--labels` slices unlabeled volumes for semi-supervised training. |
| `--max_label_no N` | The number of label classes, needed when the trainer runs separately from the slicer. |
| `--unlabeled_data_dir PATH` | A directory of unlabeled 2D slices for Mean Teacher or pseudo-labelling; required when either is enabled in the settings. |
| `--devices auto` / `--devices 2` | Train on every visible GPU, or on a given number, with Lightning's distributed data parallel strategy. The default is the single GPU named by `cuda_device` in the settings. |
| `--strategy ddp` | Override the Lightning strategy. |
| `--optuna [CONFIG]` | Run an Optuna hyperparameter search instead of a single training run; needs the `optuna` extra. See [docs/optuna.md](docs/optuna.md). |
| `--task2 DIR` / `--task3 DIR` | Enable the `boundary` or `distance` head beside the semantic one, see [Multi-task training](#multi-task-training-with-pipelineyaml). The default trainer derives those heads' targets from the semantic labels, so the directory given is not read; the legacy trainer reads precomputed labels from it. |
| `--precision 16-mixed` | Train under automatic mixed precision, overriding `precision` in the settings; `bf16-mixed` for bfloat16. |
| `--legacy-trainer` | Use the deprecated raw-PyTorch trainer, scheduled for removal in version 0.5. |

### Predicting a 3D segmentation with a 2D model

Input volumes can be HDF5, multi-page TIFF, MRC or Zarr files.

```shell
model-predict-2d path/to/model_file.pytorch path/to/data_for_prediction.tiff
```

The volume is segmented following `volseg-settings/2d_model_predict_settings.yaml`. The `quality` setting chooses how many passes are merged by maximum probability:

| `quality` | Passes |
| --- | --- |
| `low` | one axis, chosen by `prediction_axis` |
| `medium` | the three axes |
| `high` | the three axes in four rotations each, twelve passes |
| `z_only` | the z axis in four rotations |

The segmented volume is written to the working directory, or to `--data_dir`, as TIFF, HDF5 or MRC according to `OUTPUT_FORMAT` in `volume_segmantics/utilities/config.py`; any other value is refused before prediction starts. With `output_probs: True` a volume of the winning class's probability is written alongside, and with `output_entropy: True` (medium, high or z_only quality) volumes of per-voxel entropy and vote counts as well. `native_scale: True` predicts at the volume's own slice size instead of resizing each slice to the training image size, and `downsample: True` predicts at half resolution and returns every output to the input volume's shape.

A model trained with 2.5D slicing must be used with `use_2_5d_prediction: True` and the same `num_slices`; the checkpoint records its stack and a mismatch is refused with a message naming both values.

A directory of 2D images can be segmented with the module `volume_segmantics.scripts.predict_2d_model_imagedir`, which takes the model, the image directory and `--output`.

### Pipeline prediction with Zarr output and instance assembly

```shell
model-predict-pipeline path/to/model_file.pytorch path/to/data_for_prediction.tiff --output-zarr path/to/prediction.zarr
```

This wraps the Python API, `volume_segmantics.api.predict`, and reads `2d_model_predict_settings.yaml` together with the optional `pipeline.yaml`. Options:

| Option | Effect |
| --- | --- |
| `--data_dir PATH` | Directory holding `volseg-settings`. |
| `--output-zarr PATH` | Where to write the `prediction_v1` Zarr store; `--overwrite-zarr` replaces an existing one. |
| `--inference-mode MODE` | `single_axis`, `multi_axis`, `sliding_window` or `tta_uncertainty`, overriding `prediction.inference_mode` in `pipeline.yaml`. |
| `--instance-assembly-backend NAME` | `slice_overlap`, `watershed_3d` or `usegment3d`, overriding `instance_assembly.backend`. `usegment3d` needs the `usegment3d` extra. |
| `--voxel-size-zyx Z Y X` | Physical voxel size, passed to instance assembly. |

The Zarr store holds `teacher_argmax` and `teacher_probs` for the semantic head, a `boundary_map`, `distance_map` or `sdm_map` for each extra head that was trained, `tta_variance_map` and `tta_entropy_map` under `tta_uncertainty` mode, and `per_axis_instances/<axis>` and `instance_labels` when instance assembly runs.

## Training features

### Trainers

The default trainer is built on PyTorch Lightning and supports mixed precision, multi-GPU training with distributed data parallel, and the multi-head pipeline. 

`precision` in the training settings selects full precision (`32`, the default), automatic mixed precision in float16 (`16-mixed`, CUDA only) or in bfloat16 (`bf16-mixed`); `--precision` on the command line overrides it. The learning-rate sweep runs in full precision.

Training runs in two phases set by `num_cyc_frozen` and `num_cyc_unfrozen`: first with the pre-trained encoder frozen, then with every layer trainable.Under the default trainer, early stopping watches the validation foreground Dice with the patience given by `patience`, and the same metric selects the saved checkpoint.


### Models and encoders

`model.type` selects the architecture: `U_Net`, `U_Net_Plus_plus`, `FPN`, `DeepLabV3`, `DeepLabV3_Plus`, `MA_Net`, `Linknet`, `PAN`, `Segformer` or `Vanilla_Unet` (no pre-trained encoder). The default trainer builds that architecture and puts the pipeline heads on its decoder, names the checkpoint after it (for example `<date>_FPN_<model_output_fn>.pytorch`) and records it in the checkpoint, from which `model-predict-2d` rebuilds it. `model.encoder_name` selects any encoder supported by segmentation-models-pytorch, including timm encoders with the `tu-` prefix, and the DINO vision transformers `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`, their `_reg` variants with registers, `dinov3_vitl16` and `dinov3_vit7b16`. DINO encoders work with `U_Net`, `Segformer`, and with `U_Net_Plus_plus` and `Linknet` for the patch-16 models; an `encoder_depth` of 4 is recommended for them. `model.encoder_weights` is `imagenet` for pre-trained weights or `null` for random initialisation.

Two further settings initialise a model from your own files: `full_weights_path` loads a saved model's weights before training, and `encoder_weights_path` loads encoder weights alone, adapting the first convolution to the number of input channels. 

### Losses and metrics

`loss_criterion` selects `CombinedCEDiceLoss` (the default, with `ce_weight`, `dice_weight`, `dice_weight_mode` and `exclude_background_from_dice`), `DiceLoss`, `GeneralizedDiceLoss`, `CrossEntropyLoss`, `BCELoss`, `TverskyLoss`, `BoundaryDoULoss`, `BoundaryLoss` or `ClassWeightedDiceLoss`. 

The validation Dice is always reported: per class, as the mean the settings ask for (`dice_averaging` is `macro` or `weighted`, and `exclude_background_from_dice` leaves class 0 out of the mean), and as the foreground Dice that selects the saved checkpoint. `eval_metric: MeanIoU` adds the per-class and mean intersection over union to the epoch summary, the training stats CSV and the loss plot. `DiceCoefficient` reports Dice alone. 

### 2.5D slicing

With `use_2_5d_slicing: True` each training image is a stack of `num_slices` adjacent slices, an odd number of 3 or more, so the model sees context from neighbouring slices. The encoder's first convolution is adapted to that many input channels. Prediction must then set `use_2_5d_prediction: True` with the same `num_slices`.

### Multi-task training with pipeline.yaml

Multi-task training adds prediction heads beside the semantic one: `boundary`, `distance` and `sdm` (signed distance map), each with its own loss, weight and warm-up schedule. Every head reads the decoder of the architecture `model.type` names, and their targets are derived from the semantic labels on the fly, so no extra label volumes are needed. There are two ways to enable heads.

The quick way is the command line: `--task2` enables the boundary head and `--task3` the distance head, with their default losses. The argument is a path for the legacy trainer, which reads precomputed labels from it; the default trainer does not read it, so the semantic labels can be given again:

```shell
model-train-2d --data training_data/vessels_256cube_DATA.h5 --labels training_data/vessels_256cube_LABELS.h5 --task2 training_data/vessels_256cube_LABELS.h5
```

The full way is a `pipeline.yaml` in `volseg-settings`, which is read whenever it exists and takes precedence over `--task2` and `--task3`. Copy `volseg-settings/pipeline_multitask_example.yaml` to `volseg-settings/pipeline.yaml` and edit it; this is its heads block:

```yaml
schema_version: pipeline_v1
heads:
  semantic:
    enabled: true
    loss: dice_ce
    loss_weight: 1.0
  boundary:
    enabled: true
    loss: boundary_bce_dice       # bce | boundary_bce | boundary_dice | boundary_bce_dice
    loss_weight: 0.5
  distance:
    enabled: true
    d_clip: 10.0                  # distances are clipped to this many voxels
    loss: distance_l1             # distance_l1 | distance_mse
    loss_weight: 0.5
  sdm:
    enabled: false
```

Then train as usual:

```shell
model-train-2d --data training_data/vessels_256cube_DATA.h5 --labels training_data/vessels_256cube_LABELS.h5
```

The heads share one decoder by default; `decoder_sharing: separate` in the training settings gives each head its own decoder, which adds a decoder's parameters per extra head. The checkpoint is named `<date>_PIPELINE_MULTITASK_<type>_<model_output_fn>.pytorch` and `model-predict-2d` writes the semantic segmentation from it; `model-predict-pipeline` writes every head's map to a Zarr store and can assemble instances from them. 

The same `pipeline.yaml` also selects the augmentation backend (`albumentations` or `monai`), the inference mode and the instance-assembly backend; the packaged default, `volume_segmantics/settings/pipeline.default.yaml`, documents every key. (The legacy trainer's `use_multitask`, `num_tasks` and `Multitask_Unet` settings are not needed: `Multitask_Unet` as `model.type` builds the pipeline U-Net.)

### Semi-supervised learning

Slice unlabeled volumes with `model-train-2d --mode slicer --data unlabeled.h5 --unlabeled_data_dir slices/` (no `--labels`), then train with `--unlabeled_data_dir slices/` and `use_semi_supervised: True` for Mean Teacher, `use_pseudo_labeling: True` for confidence-masked pseudo-labels, or both. The consistency weight ramps up over `rampup_start` to `rampup_end` optimiser steps.

### Reproducibility and environment

`random_seed` in the training settings makes the train/validation split, shuffling and data-loader workers reproducible. `VOLSEG_NUM_WORKERS` sets the number of data-loader workers (default 8, or 4 on Windows), and `VOLSEG_KEEP_SLICES=1` keeps the sliced images after training.

## Tutorials and documentation

- [training_data/README.md](training_data/README.md): a short walk-through segmenting blood vessels in synchrotron X-ray micro-CT data of human placental tissue, using the sample data shipped in `training_data`.
- [docs/tutorials.md](docs/tutorials.md): a longer tutorial from choosing a region of interest to computing a Dice score.
- [docs/optuna.md](docs/optuna.md): hyperparameter search with Optuna.

## Testing

The unit tests run with `pytest`; GPU tests are marked `gpu` and can be deselected with `-m "not gpu"`. The script-driven regression harness in `tests/scripts` trains and predicts on the sample data under named smoke and QA configurations; see [tests/scripts/README.md](tests/scripts/README.md).

## Contributing

We welcome contributions from the community. Please take a look at our [contribution guidelines](https://github.com/rosalindfranklininstitute/volume-segmantics/blob/main/CONTRIBUTING.md) for more information.

## Citation

If you use this package for your research, please cite:

[King O.N.F, Bellos, D. and Basham, M. (2022). Volume Segmantics: A Python Package for Semantic Segmentation of Volumetric Data Using Pre-trained PyTorch Deep Learning Models. Journal of Open Source Software, 7(78), 4691. doi: 10.21105/joss.04691](https://doi.org/10.21105/joss.04691)

```bibtex
@article{King2022,
    doi = {10.21105/joss.04691},
    url = {https://doi.org/10.21105/joss.04691},
    year = {2022},
    publisher = {The Open Journal},
    volume = {7},
    number = {78},
    pages = {4691},
    author = {Oliver N. F. King and Dimitrios Bellos and Mark Basham},
    title = {Volume Segmantics: A Python Package for Semantic Segmentation of Volumetric Data Using Pre-trained PyTorch Deep Learning Models},
    journal = {Journal of Open Source Software} }
```

## References

**Albumentations**

Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A.A. (2020). Albumentations: Fast and Flexible Image Augmentations. Information 11. [https://doi.org/10.3390/info11020125](https://doi.org/10.3390/info11020125).

**MONAI**

Cardoso, M. Jorge, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot, Yiheng Wang, Benjamin Murrey et al. (2022) "Monai: An open-source framework for deep learning in healthcare." [arXiv preprint arXiv:2211.02701](https://arxiv.org/abs/2211.02701).

**Segmentation Models PyTorch**

Yakubovskiy, P. (2020). Segmentation Models Pytorch. [GitHub](https://github.com/qubvel/segmentation_models.pytorch).

**PyTorch-3dUnet**

Wolny, A., Cerrone, L., Vijayan, A., Tofanelli, R., Barro, A.V., Louveaux, M., Wenzl, C., Strauss, S., Wilson-Sánchez, D., Lymbouridou, R., et al. (2020). Accurate and versatile 3D segmentation of plant tissues at cellular resolution. ELife 9, e57613. [https://doi.org/10.7554/eLife.57613](https://doi.org/10.7554/eLife.57613).
