# Changelog

All notable changes to Volume Segmantics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

* `model-predict-pipeline` console script exposing the pipeline prediction CLI (`volume_segmantics.scripts.predict_pipeline`) with Zarr output and instance assembly.
* DINOv2 encoders with registers (`dinov2_vits14_reg`, `dinov2_vitb14_reg`, `dinov2_vitl14_reg`, `dinov2_vitg14_reg`).
* Multi-GPU slicing guard: under DDP only rank 0 slices the training volumes; re-launched worker ranks wait for its sentinel in the run directory, derive the class count from the slices on disk, and never re-slice or delete them.
* MONAI as the pipeline augmentation backend: `augmentations.backend: monai` in `pipeline.yaml` (or `auto` with `augmentation_library: monai` in the train settings) translates the configured transforms to MONAI dictionary transforms applied consistently across the image and every head target, accepts MONAI-only transforms by name, and seeds MONAI's shared random state per process. Falls back to Albumentations when MONAI is absent. Applies to the labeled and the unlabeled pipeline datasets.

### Changed

* A file path in `model.encoder_weights` no longer initialises a DINO encoder (it loaded with `strict=False`, dropping missing tensors silently). Use `dinov2`/`dinov3` for the hub backbone, or load a full saved model via `full_weights_path`.

### Fixed

* DINO encoders are registered with `segmentation_models_pytorch` through its public `encoders` registry, and `DINOEncoder` implements the smp encoder protocol (`output_stride`, `set_in_channels`, `make_dilated`). 
* A failed DINO weight download raises instead of silently continuing with a randomly initialised DINOv3 backbone.
* Prediction rebuilds a DINO model from the checkpoint alone and no longer fetches hub weights it then overwrites.
* `create_model_on_device` verifies after every construction that the built encoder's class matches the requested `encoder_name`, so a substituted backbone raises instead of training or predicting silently.
* `create_model_from_file` and `create_model_from_file_full_weights` check a checkpoint's declared `encoder_name` against its state-dict keys before building anything, refusing the mislabelled "DINO" checkpoints.
* The default (Lightning) trainer now feeds `--unlabeled_data_dir` into training correcly: every labeled batch is paired with a batch of unlabeled slices prepared exactly like the labeled ones, Mean Teacher's consistency loss is taken between the student's augmented view and the teacher's plain view of those slices, and pseudo-labelling adds a confidence-masked cross-entropy on them with its documented weight, ramp and threshold schedule. 
* The legacy settings' `consistency_weight` (default 0.1), `rampup_start` and `rampup_end` are honoured by the default trainer; `pipeline.yaml` gains an optional `ema_teacher.consistency_rampup` block.
* Various fixes: frequency-weighted Dice (`inverse_freq`, `inverse_sqrt_freq`) gives a class absent from the batch no weight instead of nearly all of it; prediction resizes slices to the `image_size` the checkpoint now records (`native_scale` in the predict settings opts out); a named `border_mode` in `pipeline.yaml` augmentations resolves to its OpenCV constant, so the packaged default pipeline trains; `output_entropy` works at medium and high quality; multi-task prediction no longer fails its alignment diagnostic on non-square slices; the Lightning trainer trains on the settings' `cuda_device` (multi-GPU is opt-in via `--devices`); batch size is taken from the memory the driver reports free; the epoch summary line, epoch history and stats CSV carry each epoch's own metrics and tolerate short or missing series; an unknown `OUTPUT_FORMAT` is refused before prediction instead of writing nothing.
* Test harness: an `encoder-family` assertion checks that a checkpoint's weights corroborate its declared encoder; the DINO smoke configs now run it, and the harness collects assertion failures instead of aborting a config at the first one.




## [0.4.0]

### Added


* Multitask support through a YAML-configurable pipeline.
* Support for MONAI dataloaders and augmentations.
* Multi-slice support.
* Pytorch Lightning trainer with support for mixed precision and multi-GPU training.
* New encoders including DINO encoders.
* New and revised loss functions.
* Extensive additional logging and training plots. 
* MRC I/O file support.
* Zarr output support for structured outputs like multi-task and confidence maps.
* Confidence maps from Test-Time Augmentation.
* The use of Optuna for automated hyperparameter tuning. 
* This release has a testing harness for script-driven regression testing. 
* Added sample data for multiclass and multitask training.
* Added Github pages documentation.

### Fixed

* Improved determinism, atomic IO and I/O error handling.



## [0.3.3] 

### Added
* Added Surface Distance loss.
* Initial support for prediction over Zarr.
* Allow prediction over a directory of images.
* Additional train settings for providing weights.
* Allow train script to run slicer and train mode separately.

### Fixed
* Downgrades Poetry version to resolve build errors.
* Fixes test to use dataset length rather than batch size.
* Fix build issues.
* Fix for surface_dice import issue.
* Ensure certain arguments are required on command line.

### Changed



## [0.3.2] 

### Changed
* Moved development to RFI.

## [0.3.1] 


### Changed
* Alters maximum PyTorch version dependency.
* Relaxes dependency constraint for albumentations library.

## [0.3.0] 

### Added

* Adds PAN model architecture and four new encoder types.
* Adds tests for specifying training and prediction axis.


### Changed
* Allows specifying specific axis for prediction.



## [0.2.8]

* Updates tests action to use latest cache operation.
* Updates Python setup version in tests workflow.
* Increases minimum dimensions for random volume in tests.

### Fixed
* Fixes regex to in datasets.py use a raw string.




## [0.2.7]

* Adds tests for VolSeg2DPredictor.
* More consolidation of pytorch-3dunet code.
* Adds tests for VolSegPredictionManager.


### Changed

* Formatting changes for VolSeg2DPredictor tests.


## [0.2.6]

### Added

* Adds training data.
* Adds imagecodecs as a dependency to enable opening compressed tiff files.
* Adds action to zip training data and add to release as asset.
* Rationalise testing for pytorch-3dunet functions.


### Changed
* Replaces GeneralisedAveragePrecision metric with DiceCoefficient.


## [0.2.5]

* Reinstates running tests on any branch push.


## [0.2.4]

* Initial public release. 

