# Changelog

All notable changes to Volume Segmantics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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

