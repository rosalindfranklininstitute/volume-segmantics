---
title: 'Volume Segmantics: an annotation-efficient volumetric bioimage segmentation toolkit (new release)'
tags:
  - Python
  - Deep learning
  - Image segmentation
  - Volume electron microscopy
  - Cryo-electron tomography
  - Bioimage analysis
authors:
  - name: Avery Pennington
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Dolapo Adebo
    affiliation: 1
  - name: Sam Kersley
    affiliation: 1
  - name: Neville Yee
    affiliation: 1
  - name: Piper Fowler-Wright
    affiliation: 1
  - name: Mark Basham
    corresponding: true
    affiliation: 1
  - name: Michele Darrow
    corresponding: true
    affiliation: 1
affiliations:
  - name: Rosalind Franklin Institute, Harwell Campus, Rutherford Appleton Laboratory, Didcot OX11 0QX, United Kingdom
    index: 1
date: 4 July 2026
bibliography: paper.bib
---

# Summary

This is a new version of Volume Segmantics with changes that allow more robust
segmentation of complex volumetric bioimages and that make it ready for use in
large-scale pipelines for analysis of modalities such as volume electron
microscopy (vEM) and cryo-electron tomography (cryo-ET). This release builds on
the functionality of the original Volume Segmantics [@King2022], which enabled
volumetric segmentation using a multi-axis approach to be usable by biologists
and non-machine-learning specialists. This new version enables multi-slice 2.5D
segmentation and multi-task training for improved segmentation quality, and the
lightweight command-line interface is easy to use in automated and HPC-based
workflows. Additionally, enhanced support is provided for an expanded range of
encoder and decoder architectures including transformer- and DINO-based encoders,
as well as additional loss functions and training options which enable highly
configurable pipelines that make efficient use of limited annotations.

# Statement of need

Volumetric bio-image segmentation is a key step in the analysis of biological
samples imaged with modalities such as 2D and 3D room-temperature or cryogenic
transmission electron microscopy (TEM) or scanning electron microscopy (SEM),
including approaches such as volume electron microscopy (vEM) and serial section
microscopy [@McCafferty2024]. Since its release in 2021, Volume Segmantics has
fulfilled a need for an open-source tool designed for non-machine-learning
specialists to train deep learning models for volumetric image segmentation.

Volume Segmantics now supports the use of multi-slice training and prediction,
where a variable number of slices forward and back in the stack are used to
train the prediction of the current slice. This additional 3D context often
results in smoother segmentations of 3D structures with less undesirable
slice-to-slice variation. Secondly, there is support for multi-task training,
where models are extended to multi-head and multi-decoder variants, for example
to allow training of models on a dataset combining multi-class segmentation,
boundary prediction and distance map prediction. 

Recent developments in deep learning, including transformer models and foundation
models, open opportunities for reducing the amount of annotation required to
train a high-performing segmentation model. This new version supports the use of
transformer and DINO-based encoders [@Caron2021; @Oquab2023] and integrates that
support with its multi-axis, multi-slice and multi-task capabilities.
Increasingly, training larger models on larger datasets, as well as the process
of hyperparameter optimisation, necessitates the use of HPC. Overall, Volume
Segmantics is a lightweight toolkit built on PyTorch that works well in HPC
settings and supports a command-line interface that allows for simple integration
into scripted workflows (e.g. for large-scale training).

# State of the field

There are a variety of implementations of volumetric segmentation pipelines.
Examples include nnU-Net [@Isensee2021], a widely used package that specialises
in 3D CNN segmentation; MONAI [@Cardoso2022], a library for medical image
segmentation; and micro-SAM [@Archit2025], a tool with GUI support for using
Segment Anything [@Kirillov2023] for volumetric images; Empanada, a tool with
integrated support for instance segmentation; and Easymode [@SoLast2026], a tool
with pretrained models for common EM segmentation tasks. Several commercial
packages exist, such as Dragonfly [@Dragonfly2025], WebKnossos [@Boergens2017]
and Avizo [@Avizo2025], which provide commercial, often proprietary, deep
learning frameworks for volumetric image segmentation. Overall, open-source
packages often require some coding to use fully, and there are few packages 
for deep-learning segmentation of volumetric images designed to allow highly
configurable and annotation-efficient segmentation pipelines outside of commercial 
options.

# New features

In addition to its core functionality of multi-axis prediction using 2D deep
learning models, Volume Segmantics has been extended with the following features:

- Multi-slice 2.5D prediction (using an integer number of additional slices).
- Multi-task learning including boundary and distance map / signed distance
  function (SDF) prediction.
- Confidence map generation using a Test-Time Augmentation method.
- Support for new encoder architectures including Vision Transformer and DINO v1
  and v2 architectures [@Caron2021; @Oquab2023].
- Support for loading pretrained encoders and decoders.
- Additional loss functions (class-weighted Dice and boundary loss).
- Support for additional augmentations through MONAI [@Cardoso2022].
- Multi-GPU training and mixed-precision training with PyTorch Lightning.
- Automatic hyperparameter optimisation with Optuna.
- Sliding-window prediction.
- MRC file support and support for Zarr output.
- Improved training metrics, easing the debugging of model training.

![Schematic representation of (A) multi-axis, (B) multi-slice and (C, D)
multi-task training supported by Volume Segmantics. The multi-axis training (A)
allows improved segmentation performance for isotropic data and supports the
confidence map approach described below. The multi-slice training (B) adds
additional 3D context that often smooths slice-to-slice noise. Multi-task
training (C) involves the prediction of additional targets through, e.g.,
semantic segmentation and boundary map, and semantic segmentation, boundary map
and distance map (D). The additional targets and multi-task training can improve
segmentation on particular tasks (e.g. samples with thin boundaries) and the
multi-target outputs can flow into post-processing to produce instance
segmentations.\label{fig:training}](media/image1.png)

# Confidence maps

To facilitate error analysis and other downstream geometric evaluation of
segmented objects, the new version of Volume Segmantics can provide two extra
volumetric outputs: full probability and entropy, that depict the model
confidence in a segmentation task. The entropy is normalised based on the number
of labels, and the resulting value estimates the uncertainty of an ensemble of
test-time averaged predictions (flips and rotations). A higher value for a voxel
naturally implies that the trained model is in general less determined about
which label to choose.

![(A) A slice of a 3D cryogenic electron tomogram showing three mature HIV viral
particles as an example of the task of segmenting the viral membranes. (B) The
entropy map from the prediction of the same slice shown in (A), showing
segmentation predictions with high uncertainty (yellow) and low uncertainty
(purple). The scale bar is 100 nm and the colour bar ranges from 0 to
1.\label{fig:confidence}](media/image2.png)

# Software design

Volume Segmantics supports configuration via two YAML files where all the
training and prediction parameters are set. The software is run by calling two
commands, `model-train-2d` and `model-predict-2d`, that accept arguments pointing
to training data, specifying training modes and pointing to output locations.
Training is monitored using logging in the terminal. When training is finished,
several summary plots are saved including an example validation prediction and
training loss plots.

The training settings file is divided into several sections, covering the data
loading, normalisation, training parameters, loss functions, model architecture
and pretrained encoder choice, multi-axis, multi-slice, multi-task, and
semi-supervised training options. The prediction file has been extended where
necessary to complement the settings in the training file and has sliding-window
prediction settings.

In addition, the Optuna configuration settings file supports automated
hyperparameter optimisation, with sections covering study settings (name, number
of trials, optimisation direction, seed, timeout), pruning, and a search space in
which parameters such as model architecture, encoder choice, learning rate
schedule, loss function, and training cycle counts are declared with their
sampling type and bounds. This can be run by calling `--optuna
optuna_config.yaml` after the training argument.

Volume Segmantics has a Python-based class interface through an API, with a
training manager class and a prediction manager class allowing the software to be
used as a library in a Jupyter notebook or external application, as in its use
within the SuRVoS2 GUI [@Pennington2022].

# Implementation and extensibility

Volume Segmantics is implemented using PyTorch and the MONAI deep learning
library and allows augmentations to be performed by either the Albumentations or
the MONAI libraries. The software has been driven by Slurm scripts on HPC, where
the two commands and respective command-line parameters allow complex
segmentation jobs to be scaled up (e.g. training multiple models in a single
Slurm script).

# Research impact

Volume Segmantics has been used in a variety of applications from cryogenic
electron tomography (cryo-ET), to cryo-focused ion beam SEM, to synchrotron X-ray
micro-CT [@Glynn2025; @Dumoux2023; @Tun2021]. In addition to bio-image
segmentation, the Volume Segmantics package can be used in segmentation of other
types of volumetric images and has been used recently for concrete aggregate
segmentation [@Werner2025] through its interface with SuRVoS2 [@Pennington2022].

![Example of Volume Segmantics use cases and research impact. (A) Cryogenic
electron tomography data showing vesicles (orange) from [@Glynn2025], segmented
using a ConvNeXtV2 encoder and U-Net architecture. (B) Cryogenic focused ion beam
scanning electron microscopy data from [@Dumoux2023] showing segmentation of a
challenging imaging modality during development of new imaging approaches. (C)
Room-temperature imaging of wax-embedded fixed human placental tissue. Top row,
left, shows an image of the reconstructed tomogram from I13 at Diamond Light
Source with the labelled fetal vascular network in white pixels, the villous
tissue in light grey and the intervillous space in dark grey. Top row, middle and
right, villous tissue and fetal vascular segmentations. Bottom row, 3D
visualisation of the villous tissue (left, pink) and fetal vascular (right, green)
segmentations.\label{fig:impact}](media/image3.png)

# Future developments

There is ongoing work to extend Volume Segmantics to allow additional support for
generating segmentation uncertainty maps and support for using additional
foundation models. The addition of MONAI for augmentations enables alternative
medical and bio-image modality-specific augmentations, and we hope to see more
applications of Volume Segmantics to new imaging modalities.

# AI usage disclosure

AI code generation (Claude, ChatGPT, Co-pilot) was used at several points of the 
development process, though all code was reviewed by and all core software design 
decisions were made by the authors. All segmentation method validation and scientific 
interpretation was performed and reviewed by the authors. Claude was used for 
formatting the paper and bibliography. All generative AI output was critically 
reviewed and manually revised by the authors. 

# Acknowledgements

The authors acknowledge support from BBSRC grant number BB/Y005953/1 (MCD) and
the Wellcome Leap *In Utero* program (MCD). Synchrotron CT imaging of human
placenta was performed on Beamline I13-2 of the Diamond Light Source synchrotron
in Oxfordshire, UK. The testing of the software and some of the segmentations were
performed using the Baskerville Tier 2 HPC service
(<https://www.baskerville.ac.uk/>). Baskerville was funded by the EPSRC and UKRI
through the World Class Labs scheme (EP/T022221/1) and the Digital Research
Infrastructure project (EP/W032244/1) and is operated by Advanced Research
Computing at the University of Birmingham.

The authors would like to thank Maud Dumoux, Audrey Le Bas, and Ana Katsini for
support in testing the software, as well as Rohan Lewis, Davis Laundon, Bram
Sengers and Aaron Grewal for enlightening discussions about how to improve
segmentation of placental villi and villous trees.

# References
