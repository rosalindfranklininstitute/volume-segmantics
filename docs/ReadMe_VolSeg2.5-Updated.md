# Volume Segmantics - Version 2.5

A toolkit for semantic segmentation of volumetric data using PyTorch deep learning models.

[![DOI](https://joss.theoj.org/papers/10.21105/joss.04691/status.svg)](https://doi.org/10.21105/joss.04691) ![example workflow](https://github.com/DiamondLightSource/volume-segmantics/actions/workflows/tests.yml/badge.svg) ![example workflow](https://github.com/DiamondLightSource/volume-segmantics/actions/workflows/release.yml/badge.svg)

The Volume-Segmentics PyTorch Deep-learning Segmentation Toolkit provides a simple command-line interface and API that allows researchers to quickly train a variety of 2D PyTorch segmentation models (e.g.  U-Net, U-Net++, FPN, DeepLabV3+) on their 3D datasets. These models use pre-trained encoders, enabling fast training on small datasets. Subsequently, the library enables using these trained models to segment larger 3D datasets, automatically merging predictions made in orthogonal planes and rotations to reduce artifacts that may result from predicting 3D segmentation using a 2D network.

Given a 3D image volume and corresponding dense labels (the segmentation), a 2D model is trained on image slices taken along the x, y, and z axes. The method is optimised for smaller training datasets, e.g. a single dataset in between 128^3 and 512^3 pixels, though larger training data can also be used effectively using an established in-house workflow to create predictions onto much larger data volumes; image augmentations and adaptable training settings are used to expand the size of the training dataset.

This work utilises the abilities afforded by the excellent [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) library in combination with augmentations made available via [Albumentations](https://albumentations.ai/) and [MONAI](https://project-monai.github.io/). The metrics and loss functions used make use of the hard work done by Adrian Wolny in his [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) repository. 

## Requirements

A machine capable of running CUDA enabled PyTorch version 2.0 or greater is required. This generally means a reasonably modern NVIDIA GPU. The exact requirements differ according to the operating system. For example, on Windows you will need Visual Studio Build Tools as well as CUDA Toolkit installed. See [the CUDA docs](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for more details. 

## Installation

The preferred use of this package involves the use of a conda and/or virtual environments with Python and Pip. For more information, the documentation for [conda](https://docs.conda.io/en/latest/) and [pip](https://pip.pypa.io/en/stable/) can be found here respectively. 

Once an environment has been created, you can install the required package using the following format. When creating an environment for this package, it is recommended that Python version 3.9 or higher be used.

```shell
pip install volume-segmantics
```

If a CUDA-enabled build of PyTorch is not being installed by pip, you can try adaptation; this particularity seems to be an issue on Windows.

```shell
pip install volume-segmantics --extra-index-url https://download.pytorch.org/whl
```

## Basic Usage

The package uses an image and a corresponding label file, representing specific components of interest from the original image. This model will aim to identify the visual components of the image based on the labels provided, whereby the model can then estimate similar features onto another comparable image, creating a prediction. The format of this prediction will return as a new label file showing the estimated features of the new image based on the training image and label data. 
 
After installation, two new commands will be available from your terminal whilst your environment with volume-segmantics is activated; one that *trains* a new model and one that uses a previously trained model to *predict* onto another image; *?model-train-2d?* and *?model-predict-2d?* respectively. 

To run these commands within a terminal, once you have activated your environment, you must navigate to the directory where you installed volume-segmantics before you execute them. These commands also require access to specific settings, stored in YAML files as part of the volume-segmantics installation; directory named ?volseg-settings?. If these are not present when you install the package, or you wish to see with original default settings after changes have been made, they can be copied from [here](https://github.com/rosalindfranklininstitute/volume-segmantics/blob/main/volseg-settings).

The file *?2d_model_train_settings.yaml?* can be edited in order to change training parameters such as *number of epochs, loss functions, evaluation metrics* and also *model and encoder architectures*. The file *?2d_model_predict_settings.yaml?* can be edited to change parameters such as the *prediction quality* and *output probability*. Further editing of these files can improve the outcome of your segmentation predictions depending on your intended usage; instructions for basic training settings fine-tuning can be found further on. 

### - For training a 2D model on a 3D image volume and corresponding label
Run the following command to complete the training process. The image and label input files can be in .HDF5 or multi-page .TIFF format. The *--data* and *--labels* arguments define the training image and label files used within the model training, describe a single image and label match as an ?image-label? pair. The relative size of each X-Y-Z measurement does not have to be equal (not a perfect square), however the image-label pairs' comparable in 3 dimensions must be the same size spatially and fit into each other without gaps to be successful.

```shell
model-train-2d --data path_to_image_data.h5 --labels path_to_corresponding_segmentation_labels.h5
```
```shell
model-train-2d --data path_to_image_data.tif --labels path_to_corresponding_segmentation_labels.tif
```
 
Paths to multiple image-label pairs can be used after the --data and --labels arguments respectively; these pairs must be provided to the command in order so that they match together when training starts 

> - An example command can be found below showing 2 image-label pairs; where image_1 and label_1 are one image-label pair and image_2 and label_2 are a second image-label pair (.h5 files).

```shell
model-train-2d --data image_1_path.h5 image-2_path.h5 --labels label_1_path.h5 label_2_path.h5 
```

A model will be trained according to the settings defined in *'2d_model_train_settings.yaml'* at the time of command execution. The setting of that training will not be saved as part of the training process, therefore it is recommended that a copy of the exact setting and placed into the model directory generated by the training process for your records. 

The output model will be saved to your volume-segmantics working directory within a new model directory containing 4 files; 

- 1] `*.pytorch*` model file; the segmentation model created by the train command that can be used to predict on other images,
- 2] `*model_loss-Plot.png*`; a graph showing the training and validation loss over the course of the training epochs (a *graphical* representation of the model's success),
- 3] `*model_prediction_image.png*`; showing specific slices of the image and labels used within the model, and the resultant prediction test outcomes for that slice (a *visual* representation model's success),
- and 4] `*stats.csv*`; a record of the training loss, validation loss and evaluation score per epoch (comparable to a dice-score (1.0 equals 100% accuracy); accuracy of the test prediction generated versus the original label). 

### - For predicting a 3D volume segmentation label using a 2D model

Run the following command to complete the prediction process. The image input file can be in .HDF5 or multi-page .TIFF format, though the model file will be in .pytorch format.

```shell
model-predict-2d path_to_model_file.pytorch path_to_image_data.h5
```
```shell
model-predict-2d path_to_model_file.pytorch path_to_image_data.tif
```

The two components of the prediction command include a path to a .pytorch training model, output by the training command previously described, and an image file. The image file must be of the same format as those used in the training image-label pairs; however the size of the new image can be either the same or larger in comparison. 

The input data will be segmented using the input model following the settings specified in *'2d_model_predict_settings.yaml'*. Depending on whether the `output_format` setting in *config.py* is set to *`.tif`* or *`.hdf`*, a *.tiff* or *.hdf5* file containing the segmented volume will be saved to your Vol-Seg working directory.

## Basic Functionality

### Split-Command Execution Shortcuts

When executing the training command, 2 processes occur: *'slicing'* the input data, and then *'training'* a model using that sliced data. If you plan to either execute multiple training runs on the same data slices with different training parameters, or execute training on predetermined slices, you can split the process and 'slice' the data once (saving it to the Vol-Seg directory in the process), and later run multiple 'train' commands on that same sliced data (reducing the overall segmentation process time) multiple times. You can do this by defining the process you wish to occur using the following arguments;

```shell
--mode=slicer
```
```shell
--mode=trainer
```

> - e.g. 'model-train-2d --data path_to_image_data.tif --labels path_to_corresponding_segmentation_labels.tif --mode=slicer'

Running the *'slicer'* argument will create 2 folders within your working volume-segmantics directory: *data* and *seg*. This will contain the sliced data from the image and label inputs, of which the *'trainer'* will then use to execute the model training. The output sliced data and seg files can be copied and renamed for the purposes of streamlining a workflow; however, when the "--trainer" argument is executed, it will only train on the data within the data and seg directories names within the Vol-Seg installation. 

*Be aware that if there are no such directories with these names within your Vol-Seg directory, the training argument will produce an error when executed.* The data and seg folders produced by the 'slicer' argument are overwritten if a full training (slicing and training through the basic command) is executed; the data and seg folders will also be deleted as part of the combined process. 

> [!NOTE]
> *Further functionality is summarised in the [Advanced Usage and Functionality Documentation.](TBC)*

### Training and Prediction Settings Fine-Tuning 

#### 2d_model_train_settings.yaml

The settings within this file have comments as to their use next to their specific components; however for further clarification for some of the more prominent settings and their possible changes are outlined below;

> - **`image_size`**; depending on your device's capability, this metric can be changed to the closest approximation for your image-label pairs' relative size as a multiple of 2. The default is set to 512, though is can be lowered if the training is killed or increased if your machine can use a higher spec. or multiple GPUs. 
> - **`model:type`**; "U_Net", "U_Net_Plus_plus", "FPN", "DeepLabV3", "DeepLabV3_Plus" etc.; Training Model Architecture (U_Net is default)
> - **`model:encoder_name`**; "resnet34", "resnet50", "tu-convnextv2_base", "tu_convnext_large", "efficientnet-b3", "timm-resnest50d"*, "dinov2_vitb14" etc.; Encoder name for feature extraction of input data (tu-convnextv2_base is default)
> - **`loss_criterion`**; "CombinedCEDiceLoss", "BCELoss", "DiceLoss", "GeneralizedDiceLoss", etc.; Loss function for guideing pixel classification (CombinedCEDiceLoss is default)
> - **`num_cyc_frozen`**/**`num_cyc_unfrozen`**; number of frozen and unfrozen epochs used in the training. the higher the number of epochs, the more training time the model will incorporate; when looking at the *model_loss-Plot.png* of a successful model, the 2 plot-lines should be reduce towards a straight line at the x axis, however if the graph starts to rise again, consider reducing the epochs; this will mean the model is over-training and you will receive a less accurate prediction as a result. **The number of unfrozen epochs should be 2/3 the size of the frozen epochs; the default is 8/5**.
> - **`alpha/beta`** or **`ce_weight`**/**`dice_weight`**; Weighting for either the BCDDiceLoss (Alp/Bet) evaluation of binary classification (0.5/0.5 is default)(depreciated legacy criterion), or CombinedCEDiceLoss (ce/dice) evaluation of imbalance and overlap (0.2/0.8 is default)(default criterion)
> - **`eval_metric`**; "MeanIoU" or "DiceCoefficient"; Evaluation metric analysing errors in segmentation model

#### 2d_model_predict_settings.yaml

The settings within this file have comments as to their use next to their specific component, however, for further clarification for some of the more prominent settings and their possible changes are outlined below *(Normalisation, Augmentation, and 2.5D settings should be kept the same as those used in the train_settings.yaml file)*;

> - **`quality`**; Degree of prediction-image analysis and prediction view (Medium is default, predicting using all 3 axis, however this can be lowered or improved based on the linked GPU capability or required output needs). 
> - **`output_probs`**; tbc
> - **`output_entropy`**; tbc

> [!NOTE]
>*Further information regarding settings specifics, component options and choice is summarised in the [Advanced Usage and Functionality Documentation.](TBC)*

#### 2.5D Slicing and other Complex Features

Volume-segmantics supports *2.5D slicing*, which creates multi-channel images from specific slices in the volume as training data; this approach provides the model with spatial context from adjacent slices. This feature can be enabled by setting `use_2_5d_slicing: True` in the training settings file; where encoder adjusts to use the number of input channels specified by the `num_slices` parameter in *volseg-settings/2d_model_train_settings.yaml* (when 2.5D slicing is enabled). When using 2.5D slicing for training, you must also enable 2.5D prediction for inference by setting `use_2_5d_prediction: True` in your *2d_model_predict_settings.yaml* file and set the `num_slices` parameter to the same integer. The feature is shown to in some cases massively increase the output efficacy and quality of multi-label segmentation predictions alongside more complex and variable contrast image inputs.

> [!NOTE]
>*Further settings fine-tuning instructions and explanations regarding complex functionality is summarised in the [Advanced Usage and Functionality Documentation.](TBC)*

### RFI-developed Workflow and Utility documentation

A team within the AI&I dept. of the Rosalind Franklin Institute, Harwell Campus, UK, has been developing a workflow and efficient protocols for the use of this package relative to ongoing research grants. This comprises of utilising Jupyter Notebooks and Napari visualisation software, alongside in-house protocols and scripts written to streamline its utilities for efficient and higher accuracy segmentation outcomes. This information can be detailed in the [Advanced Usage and Functionality Documentation.](TBC)

## Supported Model Architectures and Encoders

There are 10 current model architectures compatible with the Volume Segmantics package. U-Net is the default and the most widely-tested architecture. The full architecture list includes;

- U-Net
- U-Net++
- FPN
- DeepLabV3
- DeepLabV3+
- MA-Net
- LinkNet
- PAN
- SegFormer
- Vanilla U-net

The pre-trained encoders that can be used with these architectures are: 

- ResNet34
- ResNet50
- ResNeXt-50_32x4d
- ConvNext_base
- ConvNext_large
- ConvNextV2_base
- ConvNextV2_large
- Efficientnet-b3
- Efficientnet-b4
- Efficientnet-b5
- Efficientnet-b7
- Resnest50d\*
- Resnest101e\*
- dinov2_vit(s/b/l/g)14
- dinov3_vit(b/l)16

\* Encoders with asterisk not compatible with PAN.

## Using the API

You can use the functionality of the package in your own program via the API, this is [documented here](https://diamondlightsource.github.io/volume-segmantics/). This interface is the one used by [SuRVoS2](https://github.com/DiamondLightSource/SuRVoS2), a client/server GUI application that allows fast annotation and segmentation of volumetric data. 

## Tutorial and example data

A tutorial is available that provides a walk-through of how to segment blood vessels from synchrotron X-ray micro-CT data collected on a sample of human placental tissue. It details the volume segmantics packages basic utility and geared towards non-coders and developmental users alike to help understand the primary settings inputs. 

> [!NOTE]
> The tutorial and the example data used in its example can be found [here](TBC)

## Contributing

We welcome contributions from the community. Please take a look at out [contribution guidelines](https://github.com/DiamondLightSource/volume-segmantics/blob/main/CONTRIBUTING.md) for more information.

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

Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A.A. (2020). Albumentations: Fast and Flexible Image Augmentations. Information 11. [https://doi.org/10.3390/info11020125](https://doi.org/10.3390/info11020125)

**Segmentation Models PyTorch**

Yakubovskiy, P. (2020). Segmentation Models Pytorch. [GitHub](https://github.com/qubvel/segmentation_models.pytorch)

**PyTorch-3dUnet**

Wolny, A., Cerrone, L., Vijayan, A., Tofanelli, R., Barro, A.V., Louveaux, M., Wenzl, C., Strauss, S., Wilson-Sánchez, D., Lymbouridou, R., et al. (2020). Accurate and versatile 3D segmentation of plant tissues at cellular resolution. ELife 9, e57613. [https://doi.org/10.7554/eLife.57613](https://doi.org/10.7554/eLife.57613)
