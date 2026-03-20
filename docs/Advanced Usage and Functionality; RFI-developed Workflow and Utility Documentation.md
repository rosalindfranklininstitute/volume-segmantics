# Advanced Usage and Functionality; RFI-developed Vol-Seg Workflow and Utility Documentation

The following summative documentation contains detailed explanations and walkthroughs for tried and tested methodologies, functionalities and the overall utility of the Volume-Segmentics PyTorch Deep-learning Segmentation Toolkit in conjunction with other packages. 

The information and instructions below have been compiled from initial and developmental users of the Volume-Segmentics Package within the AI&I dept. of the Rosalind Franklin Institute, Harwell Campus, UK, towards a number of segmentation projects and grants. The purpose behind building this separate documentation was not only for the creation of more efficient models with reduced time-frame constraints and implementation of critical workflows, but also to help users with minimal coding experience use the package in conjunction with other software for maximum utility. 

For an initial overview of the package functionalities, please refer to the *Main Vol-Seg Documentation page*, which can be found [here](TBC).

## Conda/Virtual Environments

When using Volume Segmantics (Vol-Seg), the recommended pathway to getting set up is through Anaconda (conda); an open source, user friendly and well-known package management tool. Conda is used primarily due to its versatility and compatibility with other packages most commonly used in conjunction with Vol-Seg (Napari, JupyterLab). Instructions on how to download conda can be found on the [main software page](https://www.anaconda.com/download), and instructions as to its user functionality can be found in the [main documentation site](https://www.anaconda.com/docs/getting-started/main). 

Once downloaded, open a terminal instance and create a virtual environment to house the installation of Vol-Seg. It is also advisable that you create a folder in an easily accessible space to store Vol-Seg and other future environments; an example would be `/home/'user_space'/envs`. Creating an environment is done through a command line terminal using the following format, alongside deactivating it once you are finished; 

>1 - Create an environment using the *'conda create'* command; *`conda create -y -p "path_to_env" python="py_version"`*
> - `"path_to_env"` refers to the full directory path of the environment; `/home/'user'/envs/"env_name"` etc.
> - `"env_name"` refers to the name of your environment; make the name easy to remember with no spaces e.g. 'VolSeg_env'
> - `"py_version"` refers to the version of Python used within the environment.
> 
>2 - Activate the environment using the 'conda activate' command; *`conda activate "path_to_env"`*
> - `"path_to_env"` refers to the name of your full environment path
> 
>3 - Deactivate your environment when you are finished using the *'conda deactivate'* command and close the terminal; `*conda deactivate path_to_env/example-env*`

```shell
conda create -y -p "path_to_env/" python="py_version" #Python version should be 3.9 or higher
conda activate "path_to_env/env_name"
conda deactivate "path_to_env/env_name"
```

A helpful selection of Python tutorials can be found at [W-3-Schools](https://www.w3schools.com/python/default.asp), and further information relating to Python documentation can be found [here](https://docs.python.org/3/) if you are not familiar with coding formats.

### Vol-Seg Installation; Walkthrough

After an environment has been created, make sure it is activated, then navigate to your user space and into a suitable housing directory using the terminal; you can then install the Vol-Seg package using *pip*. A recommended workflow for installation included the creation or use of a housing 'library' directory (*libs*), where the repository can be navigated to easily while viewing and altering its scripts. You can use a *'libs'* folder you currently use, or an alternate location that is memorable; one can be created from the command line using the following steps, or it can be manually created using your file explorer;

>1 - Navigate to your user space within your terminal using the *'cd'* command; `cd "users/'Individual_User'"`
>
>2 - Make a *'libs'* (library) directory; `mkdir "libs"`, or further navigate to an easily accessible directory where you wish to house the Vol-Seg installation using the *'cd'* command
> - You will need to navigate into the newly made "libs" directory if one is made; `cd "users/'Individual_User'/'libs'"`

```shell
cd /users/'Individual_User'  #Navigate to your user space/desired directory.
mkdir libs  #Make a 'libs' directory (if needed).
cd /users/'Individual_User'/libs  #Navigate to your housing/'libs' directory.
```

To install the Vol-Seg package, use the following command; it should then appear in the *'libs'* folder you have created/selected once finished;

```shell
pip install volume-segmantics
```

Once installed, you can then further navigate into the installed Vol-Seg directory and begin using its commands; the commands are only executable from within the Vol-Seg directory, relying on the settings within the installation.

```shell
cd volume-segmantics
```

- If a CUDA-enabled build of PyTorch is not being installed by pip, you can try the adaptation below; this particularity seems to be an issue on Windows.

```shell
pip install volume-segmantics --extra-index-url https://download.pytorch.org/whl
```

- If you require different versions of PyTorch/CUDA as per your computer build capabilities (GPU), these can further be modified to suit your needs; uninstall the current version of PyTorch, installed within the Vol-Seg installation, and reinstall the correct/preferred version. 

```shell
pip uninstall torch  #Uninstall previous torch dependencies.
pip install torch=='PyTorch_Version'  --index-url https://download.'full link'  #Install new torch dependencies.
```

> - Information as to the versions of PyTorch/CUDA available can be found using the python [get started](https://pytorch.org/get-started/locally/) link; the download path should be marked at the `--index-url` argument alongside the versions you require. Previous versions of PyTorch can also be found [here](https://pytorch.org/get-started/previous-versions/).

- *Opt.* To keep track of your segmentation history, you can optionally install *'Tensor board'*; a package that keeps track of your training and prediction outputs. Further information about using tensor board can be found [here](https://www.tensorflow.org/tensorboard).

```shell
pip install tensorboard
```

### Vol-Seg Installation; Advanced (Poetry vers.)

The packages needed to use Vol-Seg and manage releases, updates and amend software and package parameters, can also be installed through 'poetry'; a dependency and packaging tool. This installation is suited towards developmental users or those who wish to interact with installations on a deeper level. To install Vol-Seg in this way, you should first complete the steps outlined above; create a virtual environment and 'libs' folder (or choose and navigate to a "libs" space) to interact with the modelling scripts. After these are completed, you should then do the following; 

>1 - Make sure you have your Vol-Seg conda environment activated and are navigated into your "libs" directory from the terminal; `conda activate "path_to_env/env_name"` and `cd "users/'Individual_User'/'libs'"`
>
>2 - Clone the Volume-Segmentics package directly from the online repo using the *'git clone'* command; this will then appear as a 'volume-segmantics' directory within your libs folder: `"git clone --branch 'release_version' http://'Git_link'"`(TBC)
> - You will then be able to navigate into the cloned Vol-Seg directory using the cd command; `cd "users/'Individual_User'/'libs'/'volume-segmantics"`
>
>3 - Install the poetry tool using pip; `pip install poetry`
>
>4 - Install volume-segmantics using poetry (must be navigated within the 'libs'/'volume-segmantics directory); `poetry install`
> - The poetry tool will know the packages required for the installation by reading the cloned repo Vol-Seg files. 
> - You may also be required to run the `*'poetry lock'*` command, depending on your computer setup; this will be prompted as an error message when to try to execute poetry install.

```shell
conda activate "path_to_env/env_name"
cd "users/'Individual_User'/'libs'"
git clone --branch vs_'Version' https://github.com/'full_link'.git #Cloneing the Vol-Seg GitHub repo.
cd /ceph/users/'Individual_User'/libs/volume-segmantics #Navigating to the cloned Vol-Seg repo.
pip install poetry #Install Poetry 
poetry install  #Use Poetry (once within the Vol-Seg directory) to install the packages. 
```

## Additional Utility Packages

We recommend using two additional packages to make full use of Vol-Segs capabilities; these work directly in conjunction with the segmentation workflow and advanced utilities described within this documentation. These packages are Napari (data visualisation software) and Jupyter (script writing and editing tool). 

### Napari

Napari is a well-known interactive image viewer and editor for biological images and segmentation. It deals with 2D and 3D images and is the default for workflows and scripts developed within the RFI. Further information about using Napari can be found [here]([https://jupyter.org/](https://napari.org/stable/)).

#### Installation

Napari should be installed within a newly created virtual environment; if you wish to use multiple versions of Napari within your session, making use of different plugins or user preferences, you will need to make separate environments for each version to stop bugs and errors or overlap in utility. Python version 3.11 or higher is recommended for when making these environments. The 2 main versions of Napari that can be used are napari-4 and napari-5; if you are getting consistent bugs with the newer version of Napari, move to the older version (more stable with specific image types and editing protocols). 

> - Napari-5 version 0.5.5 is the most commonly used (fast, increased functionality)
> - Napari-4 version 0.4.19 is also very popular (very stable, more plugins)

A simplified format for new environment creation can be found below; an example for Napari-4 and Napari-5 is also provided. Examples for installing Napari-4 and Napari-5 are also provided separately.

```shell
conda create -y -p path_to_env/napari_4-env python=3.11
conda activate path_to_env/napari_4-env 
```
```shell
conda create -y -p path_to_env/napari_5-env python=3.11
conda activate path_to_env/napari_5-env 
```

> - To install napari, *conda forge* is required; `*install -c conda-forge*`
> - If working with .tiff images, the newer versions of tifffile (image storing and reading Python library) can throw errors when saving, opening large files and running specific plugins. It is recommended that you downgrade the tifffile version to a more stable version *'(2022.8.12)'* compatible with Napari-4 and Napari-5 before you start working; `*pip uninstall/install/ (tifffile version = 2022.8.12)*`
> - To open a session of Napari, use the `*'python -m'*` command 

```shell
conda install -c conda-forge napari==0.4.19 pyqt  #Install Napari-4
pip uninstall tifffile  #Uninstall newest version of tifffile
pip install tifffile==2022.8.12  #Install downgraded version of tifffile
python -m napari  #Open new napari-4 session window
```
```shell
conda install -c conda-forge napari==0.5.5 pyqt  #Install Napari-5
pip uninstall tifffile  #Uninstall newest version of tifffile
pip install tifffile==2022.8.12  #Install downgraded version of tifffile
python -m napari  #Open new napari-5 session window
```

After opening an instance of napari, you should be able to import and view your image-label pair files at will, and also import your segmentation predictions after their creation.

Further information relating to conda forge can be found [here](https://conda-forge.org/docs/).

#### Basic Functionality

When opening Napari for the first time, you can import images and previous label tiff layers into the viewer. It is important that you regularly save your work to avoid losing progress. An important plugin needed for segmentation work is the *napari-h5* plugin; it allows you to import *.h5 files* into the viewer and is required for adding and editing prediction labels. To install plugins, go to *plugins>install/Uninstall Plugins>* and use the search bar to find your required packages. you may need to shut down and open a new instance of Napari when you install a new plug-in before it becomes available. Other useful plugins include the *napari-animation* and *napari-chatgpt*; additional plugins can be found [here](https://napari-hub.org/). Further information relating to napari documentation can be found [here](https://napari.org/stable/tutorials/start_index.html), and other helpful guides for first time users can be found [here](https://napari.org/stable/howtos/index.html#how-tos).

### JupyterLab

JupyterLab is a well-known development environment for using coding notebooks, and it works hand-in-hand with the workflows and scripts developed within the RFI. Further information about using JupyterLab can be found [here](https://jupyter.org/).

#### Installation

JupyterLab can be installed within in your main virtual environment or one that has been newly created. Python version 3.9 or higher is recommended for overall utility, and a simplified format for new environment creation can be found below, using the format described above;

```shell
conda create -y -p path_to_env/juptyer-env python=3.11
conda activate path_to_env/jupyter-env
```

To install JupyterLab within your environment, you must first navigate to your user space, where you can then run a *'conda install'* command; this will download Jupyter to your user space. Once successfully installed, you can use the command line to open a Jupyter session in an internet browser, or run a session of Jupyter directly from its installation.; it is recommended that you further navigate to a specific document space to save your individual files once opened.

> - Navigate to your use space and optionally the storage directory you wish to save Jupyter files too using the *'cd'* command.
> - Use *`'JupyterLab'`* to open a new session. If Jupiter is installed within multiple environments, the session will only be able to make use of the packages installed within that environment. 

```shell
cd users/'Individual_User' #Navigate to user space
conda install jupyterlab #Install JupyterLab
jupyterlab #Open an instance of JupyterLab using a web browser.
```

#### Basic Functionality

When opening Jupyter for the first time, you can create a new file using the start-up page; it is important that you are able to save any work you do within an easily accessible folder. You can create, delete, view and move your files using the Jupyter directory interface; however to open previous notebooks, they must be accessible within the location you have opened the Jupyter session from. Further information relating to Jupyter documentation can be found [here](https://docs.jupyter.org/en/latest/), and a helpful cheat sheet for first-time users can be found [here](https://github.com/iAreful/python-quick-view/blob/main/jupyter-cheat-sheet.md).

## Segmentation Overview and Workflows

### Segmentation Workflow

Through legacy/developmental and previous versions of Vol-Seg, a workflow for efficient segmentation of 3D images has been established. When first segmenting large 3D data, using multiple segmentations on progressively larger images has proven to be much more rewarding both in terms of segmentation outcome (accuracy, detail, noise reduction), and time-related efficiency. The following diagram illustrates the full process;

![width=200](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/VolSeg2.5_Documentation_Update/images/RFI%20Segmentation%20Proccess%20-%20Documentation.jpg)

The workflow shows how you may segment a large cubic dataset without spending lengthy periods of time providing the trainer large quantities of label data initially. The methodology involves using much smaller ROIs of your original data and curating labels to suit your desired segmentation outcome on a small scale, before iterating the process of editing and predicting onto larger ROIs until you can then predict onto your original dataset. Creating multiple ground truths at multiple scales increases the model efficiency in terms of accuracy, lower noise creation and artifact mitigation and lowers the overall time-cost required to improve the model output quality. 

#### Where to start; 

If you are starting with a large cubic dataset with no labels, is it suggested that you first make several smaller ROIs (different sizes) that are representative of your data; this provide Vol-Seg with a good structure of inputs that will over time make your segmentations more accurate in your user-case. Starting with the smallest ROI, create your labels using the other tools provided in this documentation (Thresholding, Layer Manipulation, Erosion/Dilation and Median Filter alongside manual annotation painting), and then gradually predict over ROIs of increasing size; editing the labels as you go to mitigate noise, under-predicted and over-predicted areas. It is also recommended that you plan out your segmentation journey in detail before you execute any plans. A tutorial containing details of this workflow can be found [here](TBC).

### Using Vol-Seg

When you have successfully installed Vol-Seg into its own environment, you will have access to its full functionality and can begin your training and predictions. It is important to note that once you have completed a training model or prediction, you should rename and save your training models and predictions to a separate space away from the volume-segmantics directory; this is to stop you from overwriting your files with newly created ones and keep track of your segmentation history and used settings. 

> - Activate your Vol-Seg environment and Navigate to the Vol-Seg directory using the *'conda activate'* and *'cd'* commands.
> - The *'model-train-2d'* and *'model-predict-2d'* commands can only be used once you have navigated to your Vol-Seg directory. 
> - Please read the *ReadMe documentation* information regarding basic command use and *Basic Vol-Seg Settings* before using the command line. This can be found [here](TBC)

```shell
conda activate path_to_env/Vol_Seg-env
cd /users/'Individual_User'/libs/volume-segmantics

model-train-2d --data 'directory_location_image' --labels 'directory_location_labels'
  #'--mode=slicer' for splitting and executing the image slicing
  #'--mode=trainer' for executing training on the predetermined image slices from --mode=slicer

model-predict-2d 'directory_location_segmantics_training__model' 'directory_location_new_image'
```

## Overview; Training and Prediction settings fine-tuning

The training and prediction *.yaml* files are organised for easy navigation. The training file is further split into 2 sections; *Basic* and *Advanced*, with the following sub sections;

### - Basic;
> Image and Model Output; File allocation settings
- *`Image and Model Output`*; Names and paths of directories for storing segmentation data
- *`im_dirname`*: folder names created for where the image slices are kept, 
- *`model_output_fn`*: training model output filename (both can be changed for personal preference),
- *`hdf5_path`*: Internal paths used for data allocation.

> Normalisation Settings; ###
- *`clip_data`*: ### (if false, change minmax_norm to true), 
- *`st_dev_factor`*: ###, 
- *`use_imagenet_norm`*: ### (using pretrained weights for imagenet).

> Model Training Inputs; Overview of training settings
- *`training_axes`*: axes used for training perspective (should be kept to all, unless specific direction is needed), 
- *`image_size`*: size of data cubes used in training (should be less than the overall image/seg file voxel size; can be reduced and increased to improve model depending on image complexity)
- *`downsample`*: data is downsized if set to true; only if working with very large datasets for testing; data should be kept whole for best models)
- *`training_set_proportion`*: ratio used for training verses validation (15% validation is default, 85% for training) 
- *`cuda_device`*: graphics card designation (if computing setup has more than 1 (max 4), 0 is for 1 and first GPU)
- *`num_cyc_frozen`*: Number of frozen training cycles (Epochs), increase/decrease per prefered training periods wanted
- *`num_cyc_unfrozen`*: Number of unfrozen training cycles (Epochs), increase/decrease per preferred training periods wanted
- *`patience`*: number of epochs to wait if validation does not improve past a specific point; decreases chance for training to start degrading with continued training time.

> Model Architecture Settings; Workflow of data processing within the model trainer
- *`model Type`*: Choice of model type; 11 possible. U_Net is default and most stable (use Multitask_Unet for multitask advanced usage only)
- *`model encoder_name`*: Choice of encoder name; 19 possible. tu-convnextv2_base is default and most stable, tu-convnextv2_large is good for larger images, resnet can give better generalisations and performance on low contrast images. 
- *`model encoder_weights`*: 'imagenet' 
- *`model encoder_depth`*: ###

> Learning Rate Settings; ###
- *`lr_find_epochs`*: 
- *`lr_reduce_factor`*: 
- *`encoder_lr_multiplier`*: 

> Learning Rate Scheduler; ###
- *`pct_lr_inc`*: 
- *`starting_lr`*:
- *`end_lr`*: 

> Loss Function Selection; ###
- *`loss_criterion`*:
- *`alpha`*:
- *`beta`*:
- *`dice_weight_mode`*: 
- *`exclude_background_from_dice`*: 
- *`ce_weight`*:
- *`dice_weight`*: 

> Evaluation Metric; ###
- *`eval_metric`*:
- *`dice_averaging`*:

> Misc. Training Settings; ###
- *`plot_lr_graph`*: 
- *`use_sam`*: 
- *`adaptive_sam`*: 
- *`encoder_weights_path`*: 
- *`full_weights_path`*: 


### - Advanced;
> Augmentation Settings; Image-Label Data Training Library
- `augmentation_library`:` Choice between Albumentations (reliable and extensive), or MONAI (Medical/healthcare specific, use for multitask)
- `use_monai_datasets`: When MONAI is selected, it uses specific datasets from medical library (needed when MONAI is used)

> 2.5D Slicing Settings; 
- `use_2_5d_slicing`: Enables 2.5D functionality, either True of False.
- `num_slices`: Choice of the number of slices it separated before selecting a central training slice; must be an odd number (3/5/7/9 etc.)
- `slice_file_format`: File format choice for slices; use .png for 3 channels or .tiff for anything over 5 slices or above.
- `skip_border_slices`: If enabled, will ski the first and last slice when choosing enacting the num_slices choice. 

> Multi-task Learning settings; ###
- `use_multitask`: Enables Multitask functionality, either True of False.
- `num_tasks`: Number of tasks integrated into the multitask designator; includes 0 as image and 1 as label files. 
- `decoder_sharing`: ##
- `seg_loss_weight`: ##
- `boundary_loss_weight`: ##
- `task3_loss_weight`: ##
- `boundary_loss_type`: ##

> Semi-Supervised Learning Settings; ###
- `use_semi_supervised`: Enables Self-supervised functionality, either True of False.
- `unlabeled_batch_size`: ##
- `consistency_weight`: ##
- r`ampup_start`: ##
- r`ampup_end`: ##
- `ema_decay`: ##
- `mean_teacher_vis_epoch_interval`: ##

> Pseudo-labeling Semi-Supervised Learning Settings; ###
- `pseudo_label_confidence_threshold`: ##
- `pseudo_label_confidence_method`: ##
- `pseudo_label_min_pixels_per_class`: ##
- `pseudo_label_use_teacher`: ##
- `pseudo_label_weight`: ##
- `pseudo_label_rampup_start`: ##
- `pseudo_label_rampup_end`: ##
- `pseudo_label_threshold_schedule`: ##
- `pseudo_label_start_threshold`: ##
- `pseudo_label_target_acceptance_rate`: ##
- `pseudo_labeling_vis_epoch_interval`:  ##

~continued explanation of settings ?? (process for advanced settings explained in separate tutorial steps later)

### Batch Predict Script

If you are confident in your training settings, or are wanting to test your trained model on multiple comparable datasets, you can use the Batch_Predict script to run a single previously-trained model on a directory of image datasets. If the model requires the editing of the Prediction Settings YAML file to execute, editing this file prior to running this script is essential (e.g. 2.5D settings). 

The <ins>*'BatchModelPredict2d.py'*</ins> script can be found in the 'notebooks' folder alongside this documentation from the GitHub Repo [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/tree/sam-branch/VolSeg2.5_Documentation_Update/Notebooks), and should be copied into your Vol-Seg Directory to access it easily. Running the script requires executing it in a terminal while navigated into your Vol-Seg directory; providing the command with both the training model location alongside the directory of datasets you wish to predict using. It is recommended that you create a new directory in an easily accessible place and make a copy of desired datasets to place into it; check the directory prior to execution. 

>1 - Activate your Vol-Seg environment and Navigate to the Vol-Seg directory using the *'conda activate'* and *'cd'* commands.
> - Make sure the <ins>*'BatchModelPredict2d.py'*</ins> script is present in the folder you have navigated to and that you have opened the file (VisualStudioCode or equivalent file viewer) to check its contents. 
>
>2 - Use the command arguments *`'-mp'`* and *`'-fp'`* to designate the *Model-Path* (mp) and the *File-Path* (fp); they should be used after the execution of the *python command* running the script; the full command should have the format; `python BatchModelPredict2d.py -mp "Path_to_Training_Model" -fp "Path_to_Image_file_Directory"`
> - `"Path_to_Training_Model"` should be the full path to the training models location and `"Path_to_Image_file_Directory"` should show the path of the directory housing the data you wish to predict on; do not put a full data file path for any file in batch predict directory otherwse it will only predict on one file. 

```shell
conda activate path_to_env/Vol_Seg-env
cd /users/'Individual_User'/libs/volume-segmantics

python BatchModelPredict.py -mp 'Path_to_Training_Model' -fp 'Path_to_Image_file_Directory'
```

## Data Processing

### Initial Segmentation Data Preparation

*The following steps use Napari and Jupyter as the main viewing and altering packages for the data segmentation; other relative packages can be used in the same way and the methodologies extracted from the explanations provided.* 

Before you start segmenting, you will need to prepare your image; make sure the file type is correct and assess the image quality; tiff is the recommended file format, though .h5 files can also be used. *The better the quality of your image, the better your labels and eventual segmentation prediction will be*. You can import it into a napari window by dragging and dropping it into the workspace or using File>Open File(s). If your image is too large to import into napari, it is suggested that you make a "large-as-possible" crop to make it importable/viewable; go to the Crop_Downsize section of 'Data Processing: Scripts' for more information. Downsizing a large image can reducing component visibility and potential model efficacy and shouldn?t be used unless constrained by computing limitations. 

If your 3D image already has label data, napari tools can be used to edit pre-exiting data. Napari tools and options such as paint-brush/eraser/fill tools can be used to augment your desired label over your imported image (unpainted area is treated as the 'background' or label sub-layer '0', and painted areas as 'components' or label sub-layer '1'/'2' etc.). If no label data exists, it is recommended to follow the segmentation workflow outlines above; crop a small ROI, create your own labels from scratch and use this to train a starting model that can be imposed onto progressively larger images. Helpful tips and additional tools for creating your own labels can be found in the next section (thresholding). 

Within a label layer, you can have multiple sub-layers identifying different biological components; these will display as different colours. To check the number of sub-layers within your parent layer, you can use the *`'np.unique'`* command within the napari terminal. 

> - *'Label-layer'* refers to the parent label layer you wish to select; starting at lowest napari layer (layer 0) and moving upwards (3rd from the bottom would be '2'). 

```shell
#Sub-layer Callback
import numpy as np
np.unique('Label-layer')
```

If there is ever an issue with saving your label layer work or image crops (bug pop-up or error), you can also use the following script in the command terminal of napari to manually save your edited layer using the *`'imwrite'`* command from *tifffile*. This code may differ depending on the version of tifffile used, if changed previously. The code for a simple splice can be found below and can be copied straight into and amended in the Napari terminal; 

> - *'layer_Designation'* is the layer you wish to manually save
> - *'File_Location_with_Name'* requires a full path to the directory you wish to save into, and a file name; this should be in the format *'/ceph/users/'Individual_User'/.../Filename.tif* (.tif is required after your chosen file name).

```shell
#Manual Saving
import tifffile
layer = viewer.layers["Layer_Designation"]
tifffile.imwrite('File_Location_with_Name', layer.data, compression=('zlib'), compressionargs={'level': 1})
```

Further information and guidance for using napari can be found in the [viewer tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html#viewer-tutorial)

> [!NOTE]
> *A worked example using a high-definition human-placenta scan (WINSdata) shows the basic utility of some of the functions of this method and can be found [here](TBC).*

## Data Visualisation and Editing in Napari

Alongside the tools provided by Napari, developmental users have created a selection of code-based tools for processing your data. *Downsampling, Thresholding, Layer Manipulation and ROI cropping* and are more utilitarian, however *Erosion/Dilation, Voxel-median Filter* and *Voxel-delete Filter* can decrease your manual annotation time and help produce accurate training labels and final segmentations more efficiently. 

### Protocol: Downsampling (image and label)

If you have an image that is large in both size and data capacity, Napari may sometimes struggle with opening it. If this is the case, the <ins>'Crop_Downsize'</ins> Jupyter script can be used to process the file into an openable state. 

In other cases where the files are openable but to large to edit further or interact with, or where circumstances may require you to compress the image and/or labels, *downsampling through napari* can be used; it involves taking information away from the file, reducing the data size, without compromising its quality as much as possible. It is recommended that downsizing by 2 is enough for images, however there is an example code for downsampling by 4; these can be copied into the Napari terminal to use.

> - 2 is half of the original image's size and 4 is a quarter of the original image's size (in data size not spatial resolution). It is recommended that you use as much data as possible.
> - *'layer'* is the designated layer you wish manually downsize

```shell
#Downsampling by a Factor of 2
DSimage = viewer.layers['layer'].data  #Asign data 
DSfinal = DSimage[::2,::2,::2]  #Downsample data
viewer.add_labels(DSfinal)  #Create new layer 
```
```shell
#Downsampling by a Factor of 4
DSimage = viewer.layers['layer'].data  #Asign data 
DSfinal = DSimage[::4,::4,::4]  #Downsample data
viewer.add_labels(DSfinal)  #Create new layer 
```

### Protocol: Thresholding (image)

If you have image data with no label data, or wish to extract binary image information based on its grey-scale components, thresholding allows you to create a label layer only containing information you have decided on; 

Using the *Contrast Limits - Advanced Sidebar* (right click on contrast bar when an image is selected), you can use the slider to find an integer where the boundaries or specific wanted components are best visible and "snapshot" this, creating a label layer of observable features below that contract limit (higher integers will mean only brighter components will be taken into account). This can then be readded to your Napari window as an interactable label layer that you can further manipulate. The code for thresholding can be found below and can be copied straight into and amended in the Napari terminal;

> - *'I_layer'* refers to the image layer.
> - *'Threshold_Limit'* refers to the chosen threshold integer.

```shell
#Image Thresholding by Factor
img = viewer.layers['I_layer'].data
imgT = (img > 'Threshold_Limit') * 1
viewer.add_labels(imgT) 
```

### Utility; Layer Manipulation (labels)

Manipulating label data comes in many forms; *splicing, shifting, combining* and *subtracting* and are the most commonly used small-script inputs you can use on the command line to augment your layers to suit your purposes. 

~ A *splice* involves creating a new label layer containing a chosen sub-layer from a parent layer; if you have 2 sub-layer labels in one parent layer, a splice will take your chosen sub-layer from that parent layer, copy it, and create a new layer with only that sub-layer data within it. You may choose both which sub-layer the newly created layer is copied into, alongside which sub-layer from the parent layer is chosen to be spliced. The code for a simple splice can be found below and can be copied straight into and amended in the Napari terminal;

> - *'OLayer'* refers to the parent layer chosen.
> - *'SLayer'* refers to the sub-layer within the parent layer chosen.
> - *'SLD'* (refers to the 'Sub-Layer Designation' of the newly created layer from the splice.

```shell
#Splice/Shift
lA = viewer.layers['OLayer'].data  #'OLayer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1).
lA_specific = (lA == 'SLayer') * 1   #'SLayer'; index (starts as [0]) is layer within the parent layer (e.g. 1 = tissue).
lX = ((lA_specific) > 0) * 'SLD'  #'SLD'; the SLayer designation of the newly created copy layer (e.g. 2 = cell).
viewer.add_labels(lX)  #Creation the new layer.
```

The default for a *spliced* sub-layer should be the same sub-layer it was originally assigned to (where 0 is the background, or anything not painted)*(SLayer and SLD are the same)*, however the spliced data can instead be *shifted* into another sub-layer designation using another 'SLD' *(SLayer and SLD are different)*. 

> e.g. *`lA_specific = (lA == 1) * 1` -> `lX = ((lA_specific) > 0) * 2`* would splice the data from sub-layer 1 and copy it "shifted" into sub-layer 2 within the newly created label.

~ A *simple combination* involved the addition of 2 separate layers (both with a single sub-layer within them), forming a newly created layer; You may choose both layers to be combined. When executed, if the result produces additional/overlaps in layer data where the two combined layers were to mix/crossover; use the *complex combination* method and keep the sub-layer designations the same. The code for a simple combination can be found below and can be copied straight into and amended in the Napari terminal;

> - Both sub-layer designations within each parent layer must be the same; otherwise, the 2 layers will overlap and not merge. 

```shell
#Simple Combination
l1 = viewer.layers['Layer1'].data  #'Layer1'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
l2 = viewer.layers['Layer2'].data  #'Layer2'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.2).
lcombined = ((l1 + l2))  #Combination of data.
viewer.add_labels(lcombined)  #Creation the new layer.
```

~ A *complex combination* involves designating both the parent layer and sub-layer of the labels you wish to combine. You can choose both which parent layer, and the sub-layer within them you with to combine. With this code, you can also decide the sub-layer designation of the label data. The code for a complex combination can be found below and can be copied straight into and amended in the Napari terminal;

```shell
#Complex Combination
l1 = viewer.layers['Layer1'].data  #'Layer1'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1)
l2 = viewer.layers['Layer2'].data  #'Layer2'; index (starts as [0]) is the parent layer in napari (e.g. previous labels 1)
l1_specific = (l1 == 'sub_Layer1') * 1   #'sub_Layer1'; index (starts as [0]) is layer within the parent layer (e.g. 2=cells1)
l2_specific = (l2 == 'sub_Layer2') * 1   #'sub_Layer2'; index (starts as [0]) is layer within the parent layer (e.g. 1=previous label)
lcombined = ((l1_specific + l2_specific) > 0) * 'SLD' #Combination of data where 'SLD'; the SLayer designation of the newly created copy layer (e.g. 2=cell1).
viewer.add_labels(lcombined) #Creation of the new layer.
```

~ *Simple and complex combination* can be turned into *subtraction* alternatives by swapping the addition sign within the 'Combination of data' line (`lcombined`) to a subtraction. It should be noted that when you subtract, the second parent layer designated (l2) will be the data subtracted from the first layer designated (l1). 

> l2 will be subtracted from l1 *`'(l1_specific - l2_specific) > 0'`*

### Protocol; ROI cropping (image and label)

In napari, when you input an image, the default position is the Z-axis (you can view the data through the other axis using the axis shuffle button); if you are unsure about which axis you are currently viewing, refer to the number shown to the left side of the slice scrollbar (0=Z, 1=Y, 2=X). 

Taking an ROI crop of a larger 3D image or label layer requires you to know the 3D coordinates of the required cube in 3D space before you run the short-script. The co-ordinates needed are delineated by 6 numbers; the start (s) and end (e) point for each axis you require to form the necessary volume. 

> The 6-point co-ordinates are given by the formula; *"range_z", "range_y", "range_x"*, where the range has a start number and end number split between a colon (Zs:Ze, Ys:Ye, Xs:Xe)
> - This volume can be any dimension; however the script will throw an error if the numbers are outside of the image/labels designated space.
> - e.g. for a 700_voxel-cube within a 2000_voxel original image/label; `20:720, 130:730, 1200:1900`

The co-ordinates can be derived by using the Napari-builtins plug-in (already installed into your napari package at installation); dragging your mouse cursor over an image  will display the cursors current position in 3D space as a 6-point reference, found at the bottom of the napari window within a square bracket (without an image, the labels will have no anchor and therefore give incorrect co-ordinates). 

The recommended starting point for deciding these coordinates is viewing the image data from slice 0 on the Z-axis; a guide for finalising your ROI coordinate numbers can be found below though a demonstration of this visualisation is better illustrated in the [Vol-Seg Tutorial](TBC).

>1 - Inspect your main image in all 3 axis and choose the area of the image/label you wish to crop; be mindful of the size of the image you wish to create (if you have a set size in mind, make a rough note of the dimensions in the form of *"range_z", "range_y", "range_x"*).
>
>2 - Navigate to the Z-axis and scroll to the earliest slice (left-most) on the current plane view of which you wish to use (Zs); this is the starting Z-axis co-ordinate.
> 
>3 - Use your cursor, and visualise a square around your chosen crop area; place your cursor in the top-left-hand corner of this square and observe the numbers displayed in the napari builtins display.
> - Your cursor placement will show your current slice location (in the Z-axis position, which should not change when you move your cursor), alongside the position of the starting Y and X axis positions in the following format: (Zs, Ys, Xs).
> - *The Y coordinate will start from the top of the image and increase as you move down an image layer, whereas the X coordinate will start from the left and increase as you move right across the image*.
> - Make a note of these 3 numbers and input them into your 6-point coordinate reference.
>   
>4 - Scroll right along the Z-axis using the slice scroller until you view the last slice on the current plane view, which you wish to use (Ze); this is the ending Z-axis co-ordinate.
>   
>5 - Use your cursor, and again visualise the square around your chosen crop area; place your cursor in the bottom right-hand corner of this square and observe the numbers displayed in the napari builtins display.
> - Your cursor placement will show your current slice location (in the Z-axis position, which should not change when you move your cursor), alongside the position of the ending Y and X axis positions in the following format; (Ze, Ye, Xe).
> - Make a note of these 3 numbers and input them into your 6-point co-ordinate reference.
>   
>6 - **Note**; when forming the 6-point co-ordinate reference, the numbers should form 3 pairs, the difference in each pair will be the range of the length, width and breadth of your volume, and should match any previous ideas of set size (instruction 1). The size of the crop returned in the script and can be checked (the '.shape' command) within the cropping short-script. 

Once you have your co-ordinates, you can place them into the code formula for images and label layers respectively; it is important to note that for training, a crop from an image and label pair must be exactly the same size, and must superimpose over each other correctly. *Use the same co-ordinated for both the image and label layer.* 

The short-script will output a new layer containing your crop, without disturbing your original image and its size. The code should be run 2 times, once for your image data and once for your label data; your image should the cropped first to visualise your ROI before running the same co-ordinates on your label layer. The code for ROI cropping can be found below and can be copied straight into and amended in the Napari terminal;

> - *'Layer'* is the image or parent label layer you want to be cropped. 

```shell
#Image-layer cropping
img = viewer.layers['Layer'].data #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
img.shape  #View the shape of the original data volume

img_roi = img["range_z", "range_y", "range_x"]  #Designate crop shape and size; must be in the format [#:#, #:#, #:#]
img_roi.shape  #View and check the shape of the ROI data volume to be cropped

viewer.add_image(img_roi)  #Creation the new layer.
```

```shell
#Label-Layer cropping
l = viewer.layers['Layer'].data  #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
l.shape  #View the shape of the original data volume

l_roi = l["range_z", "range_y", "range_x"]  #Designate crop shape and size; must be in the format [#:#, #:#, #:#]
l_roi.shape  #View and check the shape of the ROI data volume to be cropped

viewer.add_labels(l_roi)  #Creation the new layer.
```

### Protocol; Erosion and Dilation (labels)

Erosion and Dilation are 2 useful tools for editing label layers; they edit the boundaries of a label, either expanding or decreasing its size by a specific factor. The factor is measured in 3D voxels, which can be specified and changed depending on the size of the erosion or dilation required. The voxel_imput requires a 3-digit size designation called a structure_element, where an image mask is used to calculate the area in the background to be filled or deleted relative to its 3D shape. 

This short-script uses the *binary_erosion* and *binary_dilation* functions of `scipy-ndimage` (*Running this code will produce an error if this package is not installed within your napari environment, it should be installed as standard though if not; it can be installed using pip from the terminal and restarting your Napari session*). The code for an erosion and dilation can be found below and can be copied straight into and amended in the Napari terminal;

> - *'Layer'* is the label layer you want to be eroded or dilated. 
> - *(x, x, x)* is the size of the voxel space chosen to erode or dilate by; 3 is a typical starting size and is displayed as (3, 3, 3) in 3D space.

```shell
#Erosion script
from scipy import ndimage
binary_mask = viewer.layers['Layer'].data  #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
binary_mask = binary_mask.astype(np.uint8)  #Set data to binary
structuring_element = np.ones(('x, x, x'))  #Set structuring_element parameter; where (x, x, x) is the size of the voxel space in 3D
eroded_mask = ndimage.binary_erosion(binary_mask, structure=structuring_element)  #Calculate erosion mask
viewer.add_labels(eroded_mask)  #Create new layer 
```
```shell
#Dilation script
from scipy import ndimage
binary_mask = viewer.layers['Layer'].data  #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
binary_mask = binary_mask.astype(np.uint8)  #Set data to binary
structuring_element = np.ones(('x, x, x'))  #Set structuring_element parameter; where (x, x, x) is the size of the voxel space in 3D
dilated_mask = ndimage.binary_dilation(binary_mask, structure=structuring_element)  #Calculate erosion mask
viewer.add_labels(dilated_mask)  #Create new layer 
```

### Protocol; Voxel-Component_Delete Filter (labels)

Prediction images can contain small false positives within the data background or around boundaries; shown as specs of label-layer material within the segmentation. This is likely due to noise, and even the best performing models will contain them. Segmentations will also often contain smaller unwanted artefacts that are separated from the main label components, and it can be time-consuming to erase manually. The Voxel-Component_Delete Filter deletes the unconnected smaller components of a label layer bellow a certain threshold, allowing for more consistent and polished post-processed label-layers. It searches for objects of a chosen 'min_size' and creates a mask that removes them from the label layer.

The short-script uses `scipy-ndimage` and creates a 'filter_components' function, where you can then define a minimum size of object to delete (*Running this code will produce an error if this package is not installed within your napari environment, it should be installed as standard though if not; it can be installed using pip from the terminal and restarting your Napari session*); this minimum size is represented as a single integer, the cubic volume of a 3d space (an object fully surrounded by background that fits into the cubic volume will be highlighted by the function as an unwanted component). A cubic volume of 5-5-5 (l/W/b) will be represented as a voxel_size of 125 (5x5x5). 

The size of the voxel can range from 2^3 onwards, though the Napari window will likely be killed above 30^3-voxels even with superior computing capabilities; if you run a command deleting many small components with a high-voxel component-delete filter, you may run the risk of killing the process; it is recommended that you run progressively larger component deletions to avoid this from happening (saving between each iteration), checking for unwanted component deletions as you progress. The code for the smoothing voxel-component delete filter can be found below and can be copied straight into and amended in the Napari terminal;

> - *'Layer'* is the label layer you want to be affected. 
> - The *'voxel-size'* is input as a single integer, with the increasing number size equating to a larger deletion factor; 3=9, 5=125, 9=729, 14=2744 etc..
> - The *`'def filter_componenets'`* command creates the filter that defines the data volume and the minimum size of objects wanted for deletion.

```shell
#Voxel-Component_Delete
import numpy as np
from scipy.ndimage import label, sum
def filter_components(volume, min_size):
    labeled_volume, num_features = label(volume)
    sizes = sum(volume, labeled_volume, index=range(1, num_features + 1))
    mask = np.isin(labeled_volume, np.where(sizes >= min_size)[0] + 1)
    filtered_volume = mask * volume
    return filtered_volume #Created filter_components function
seg = viewer.layers['Layer'].data  #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
del_comp = filter_components(seg, 'Voxel-size')  #Run deletion filter with a specific 'Voxel_size'; the cubic minimum size or deletion 5=125, 9=729 etc.
viewer.add_labels(del_comp)  #Create new layer 
```

### Protocol; Voxel-Median Filter (labels)

Predictions can often produce noisy or unwanted sharp boundaries in labels, and manual editing can sometimes create inconsistent edges and gaps in ground truth slices; smoothing these unwanted small imperfections on the boundaries and between slices can produce more favourable label layers for training. The Voxel-Median Filter acts as a smoothing code for rounding off labels and finding the best-fitting and consistent edge to a label layer. It uses a filtering size to define an averaged approximation of the label boundary between slices in 3D space along all 3 axes, and outputs the median overlapping option

The short-script uses `scipy-ndimage` and creates a *'median_filter function'*, where you can then choose the degree of smoothing and designate the layer you want to edit. It is recommended that you find the best smoothing_size by running iterations of the filter; the smoothing can impact the accuracy of your segmentation layer and reduce the detail you have previously put in if too high. 

A typical smoothing_size is **3** though can be higher, and it is suggested that you do not go above a smoothing_size of 9, as it tends to create straight lines and overfit the median average. The code for the smoothing voxel-median filter can be found below and can be copied straight into and amended in the Napari terminal;

> - *'Layer'* is the label layer you want to be affected. 
> - The smoothing size can be input from *'2'* onwards , with the increasing number size equating to a larger degree of smoothing.
> - The *`'def median_filter'`* command creating the filter takes the binary image data ('binary_image.astype(bool)') and returns a mask of the data that has been smoothed relative to the *'filter_size'*.

```shell
#Voxel-Median Filter
import numpy as np
from scipy import ndimage
from scipy import io
def median_filter(binary_image, filter_size='smoothing_size'):  #'smoothing_size'; the voxel-filter size or smoothing degree required
    binary_image = binary_image.astype(bool)
    filtered_image = ndimage.median_filter(binary_image, size=filter_size)
    return filtered_image * 1
seg = viewer.layers['Layer'].data  #'Layer'; index (starts as [0]) is the parent layer in napari (e.g. prediction model 1.1).
median_image = median_filter(seg) #Calculate median filter
viewer.add_labels(median_image) #Create new layer 
```

## Data Processing Scripts

More complex scripts that have been developed for further utilities that can not be used directly within napari; requiring multiple components and packages, they can instead be opens in Jupyter Lab; images and labels can be imported as data, manipulated and complex code executed seamlessly, where outputs can be opened again in Napari.

*Master versions* of these notebook scripts can be found alongside this documentation (GitHub Repo) and copied into your own user space, openable within your own Jupyter sessions. It is recommended that you copy and save these notebooks to an easily accessible place and that you fill out as much information as possible within the initial cells to keep track of what is being run and has already been completed. 

The *<ins>Crop_Downsize</ins>* and *<ins>Manual_Dicescore</ins>* scripts are more utilitarian, however the *<ins>Voxel-Fill_Holes</ins>* and *<ins>Flood-Fill_Hole</ins>* are very useful and can decrease your manual annotation time and help produce accurate training labels and final segmentations more efficiently.

### Script; <ins>Crop_Downsize</ins>: Cropping a large 3D file image and creating a downsized file

This script allows you to take a large .h5 or .tif file (usually obtained directly from a microscopy session or otherwise) and process the data in a Jupyter notebook if the data file is to large to be opened in Napari. The notebook allows you to crop the original data file into manageable ROIs and/or also allows you to downsize the full image and produce a .tif file of the data. The script has a full explanation of the code, written for non-coders, and duplicatable cells that use its functionality as intended.

The Jupyter file can be found in the 'Jupyter_notebooks' folder linked to this documentation [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/VolSeg2.5_Documentation_Update/Notebooks/Crop_Downsize.ipynb).

### Script; <ins>Manual_Dicescore</ins>: Calculation of Prediction accuracy vs a GroundTruth.

This script allows you to calculate the DiceScore of a prediction segmentation with respect to a 'ground truth' label or segmentation. The ground truth should be a label layer that is as near perfect to the required output as possible, whereby the DiceScore measures the accuracy of a prediction over 100% (represented as a 1.00 decimal). This script can be useful for evaluating prediction accuracy of multiple models numerically. The notebook assigns 2 .tif files as label data and then uses tensors within the MONAI package to calculate the shapes of both files relative to each other; it then prints a DiceScore as described in the notebook cells of which can be copied. *MONAI may need to be installed within your JupyterLab environment before the execution of this notebook*. The script has a full explanation of the code, written for non-coders, and duplicatable cells that use its functionality as intended.

The Jupyter file can be found in the 'Jupyter_notebooks' folder linked to this documentation [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/VolSeg2.5_Documentation_Update/Notebooks/Manual_DiceScore.ipynb).

### Script; <ins>Voxel-Fill_Holes</ins> Protocol (2D and 3D).

This script allows you to run a 'Fill-Holes' protocol, where a label layer is allocated and analysed, and then a script is run to infill small holes of a specific size in the label layer either in 3D and 2D to make noise easier to manage to decrease manual annotation time. The 2 methods (3D and 2D) have slightly different ways of finding and calculating the hole size that is to be infilled; the most utilitarian of the 2 methods is the 2D version. The notebook outputs a corrected .tif file of the labels layer with the holes of a specific size filled. The script has a full explanation of the code, written for non-coders, and duplicatable cells that use its functionality as intended.

The Jupyter file can be found in the 'Jupyter_notebooks' folder linked to this documentation [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/VolSeg2.5_Documentation_Update/Notebooks/Voxel-Fill_Holes.ipynb).

### Script; <ins>Flood-Fill_Hole</ins> Protocol 

~TBC (Dolapo)

The Jupyter file can be found in the 'Jupyter_notebooks' folder linked to this documentation [here](TBC).

## Advanced Training and Prediction Utility and Settings setup

The following section explains the more advanced capabilities for the toolkit
and gives instructions for Vol-Seg users on how to further edit the training a prediction settings .yaml files alongside guidance on inputting additional data to improve segmentation outcomes.

### Augmentation settings 

When you train a model, the trainer looks for comparable differences between what *is* (label-layer) and *is* not selected (background) from your data; it specifically looks for what is different. Augmentation is used to modify the data to expose this difference (contrast, blur, rotation, mirroring etc.) more clearly without changing the data itself. The Vol-Seg package has the capability to use 2 Augmentation libraries; **Albumentations** (widely used in industry and open-source projects) and **MONAI** (healthcare-imaging-specific framework for multi-dimensional image preprocessing). 

> [!NOTE]
> *Further information regarding [Albumentations](https://albumentations.ai/) and [MONAI](https://project-monai.github.io/index.html) can be found at the following links. 

Both packages work very well and have produced great outcomes on past and current projects. Users of Vol-Seg can select either library though may find that depending on their input images that one set of augmentations works better than the other. To switch between these libraries, use the '*training_settings_YAML*' to assign your preference prior to execution using the `augmentation_library` argument; further information can be found in the *Training and Prediction settings fine-tuning* section of this documentation.

### 2.5D training

~2.5D explanation~ include 3/5/7/9 choice explanation; AVERY for background and checks +++++

Implementing 2.5D can have a greater effect on larger datasets, those with more inner-label image complexity (where the image labels highlight a high variability in image detail (more differences for the trainer to measure)), wider image contrast threshold, or with datasets that contain a lower number of sub-layers (1-3 label layers). Datasets with instability in overall image contrast or a larger number of label sub-layers can often produce good segmentations with this method though it may also produce instances of hashing artefacts, inconsistencies in boundary identification across sub-layers and under/over segmented regions where the in-label area is possibly too complex. It is apparent that using ROI predictions from the 2.5D method, alongside those produced from non-2.5D Vol-Seg training, towards larger-image models can improve overall segmentation quality when following the recommended iterative segmentation workflow. 

To enable this feature, change the `use_2_5d_slicing` query to **True**, and then assign the `num_slices` to your preferred setting; the higher the number, the more data is chosen when the middle slice is selected. When choosing this number, also make sure you use the correct `slice_file_format` as per the directions in the '*training_settings_YAML*'. **It is very important to make sure you use the same settings in the '*predict_settings_YAML*' as the '*training_settings_YAML*'; training models using `use_2_5d_slicing: True` require the same argument when predicting using that model outcome.** *A model that incorporates data a from 2.5D prediction, but doesn?t use 2.5D when training the next model does not require a true argument when predicting.*

### Multi-task training

~Multi-task explanation~ MONAI only? ; AVERY for background and checks +++++

The multi-task utility allows additional data to be considered and processed during model *training* using multiple tasks on label data (decoder); segmentation using this method uses **3+** components/processing tasks; your standard *image* and *label layers*, alongside additional training data. Currently, this functionality includes a label *'boundary map'*, though there is also the capability of adding additional tasks in future updates or through developmental work. This boundary map is created from your original label data, following the edge of your label concisely; his label should be as complete as possible with respect to the original image as incomplete labels can lead to poor segmentation outcomes using this method. *The boundary map will output as a .tif file; to train a model using this map, the image and label files must also be .tif files*. To generate a boundary map, use the instructions bellow;

#### - Boundary map Creation;

To create a boundary map of your data, use the <ins>'*CalculateBoundaryMap.py*'</ins> script. This script can be found in the 'Jupyter_notebooks' folder linked to this documentation in the GitHub Repo [here](TBC); it should be copied and saved to your Vol-Seg directory or to an easily accessible folder in your user space. 

Open your Vol-Seg environment, navigate to your Vol-Seg Directory (or where the <ins>CalculateBoundaryMap</ins> script is saved to) and use the following command to formulate and execute the code and create your file; this map then can be opened and viewed in Napari. 

>1 - Activate your Vol-Seg environment and Navigate to the Vol-Seg directory using the *'conda activate'* and *'cd'* commands; `conda activate "path_to_env/env_name"` and `cd /users/'Individual_User'/libs/volume-segmantics` (or alternative location)
> - Make sure the <ins>*'CalculateBoundaryMap.py'*</ins> script is present in the folder you have navigated to, and that you have opened the file (VisualStudioCode or equivalent file viewer) to check its contents.
> Using this script may also require the additional installation of additional packages not installed when installing Vol-Seg; if this occurs, the package requirements will appear as error messages and *pip* can be used to install them to the Vol-Seg environment; the requirements can be found in the first part of the script.
>
>2 - Use the command arguments *`'--thickness'`* and *`'--min_component_size'`* to designate the *Boundary thickness* and the *solitary Component Size (within the boundary perimeter) you wish to remove from your map surrounding the label boundary(smaller than the integer allocated)*; they should be used after the execution of the *python command* running the script; the full command should have the format; `python CalculateBoundaryMap.py "Path_to_Image_File" --thickness 'Integer' --min_component_size 'Integer'`
> - `"Path_to_Image_File"` should be the full path to the label files location.
> If command arguments are not used, defaults for *`'--thickness'`* and *`'--min_component_size'`* will be used ('3' and '0' respectively); this is a good place to start if you are unsure about the initial integer inputs and what the boundary map produces. 

```shell
conda activate "path_to_env/Vol_Seg-env"
cd /users/'Individual_User'/libs/volume-segmantics

python CalculateBoundaryMap.py "Path_to_Image_File" --thickness 'Integer' --min_component_size 'Integer'
```

It is recommended that before you use your boundary map, you view it in Napari overlapping your original image and label file used to create the boundary map. If the outcome is unsatisfactory; alter the argument integers for the script (increase or decrease the default boundary thickness or component size to confine/redefine the boundary further). 

#### - Multitask training;

To run the Multitask training, use the same format for running a simple training model, but add an extra argument designating the boundary map as an extra task; in this case the task has the designation '2' ('0' is the image, and '1' is the labels) 

```shell
model-train-2d --data 'Path_to_Image_File' --labels 'Path_to_Label_File' --task2 'Path_to_BoundaryMap_File'
```

It is also possible to run a Multitask training model on multiple ROIs; you must the same number of boundary maps as image-labels pairs within the training execution command. List the image, label and boundary map files in the same order per argument, mirroring the same format as explained in the multiple image-label pair training instructions in the [ReadMe Documentation](TBC).

> An example command can be found below, where image_1, label_1 and BM_1 are one image-label-BM set, and image_2, label_2 and BM_2 are a second image-label-BM set.

```shell
model-train-2d --data image_1_path.h5 image-2_path.tiff --labels label_1_path.h5 label_2_path.tiff --task2 boundarymap_1_path.tif boundarymap_1_path.tif
```

Implementing Multitask training can have a greater affect on medium-sized datasets, those with complex boundaries or with datasets that contain a higher number of sub-layers; large datasets or those that require more ROIs may need larger computing capabilities. Depending on the number of sub-layers (large quantity) and the quality of both the boundary map, respective to the original segmentation, the outcome predictions may sometimes vary; changing the `loss_weights` in the *'Multi-task Learning settings'* can positively affect these outcomes depending on the issues that arise where favouring the boundary or segmentation data to suit can be experimented with. It is apparent that using this method on larger image iterations (rather than initial ROIS), where the ground truth is more evolved or complete, or where the segmentation boundaries take precedence in your data quality, can prove more effective and increase the final prediction quality when following the recommended iterative segmentation workflow. 

To enable this feature, change the `use_multitask` query to **True**, and then make sure the `num_tasks` reflects the number of tasks you will be using during the training setting; '2' will be image and label as default with the addition of an extra task ('2') for the boundary map.

`decoder_sharing` ??

`seg_loss_weight` ??
`boundary_loss_weight` ??
`task3_loss_weight` ??

`boundary_loss_type` ??


... *MONAI augmentations* should be selected when using Multitask training, making use of its libraries specifically designed to incorporate boundary maps alongside label data. Adding the boundary maps to your data (adding the extra task) will also not only increase the 'slicing' time required when running the model, but also may require a longer epoch number to allow for better training conditions depending on the number of label sub-layers connected to your image (more complex boundary map used during the multitask training). 

### Self-Supervised Training 

~Self-Supervised Training explanation~ 2 integrations?? ; AVERY for background and checks +++++

The Self-Supervision utility allows additional data to be considered during model *learning* using further unrelated image data (encoder); Segmentation using this method uses additional unlabelled image data to provide further support when looking for differences during label-image data comparison. *The unlabelled data should not be the same as that included in the main image inputs when training a model using this function; the unlabelled data should be from the same or a comparable scan but not from the same ROI.* This data must also be generated prior to the training models execution; to generate unlabelled data, use the instructions bellow:

#### - Generate Unlabelled Data; (use for both Mean Teacher and Pseudo-labelling)

To create your unlabelled data, you only require an image file; this should be either an ROI generated from the original image, that contains no data from the image you wish to use as 'labelled data' within the model, or an ROI from another image that contains the same image components in the same state as the labelled data (same contrast ratio and component realisation). The unlabelled data should also be smaller than the labelled image data by a factor of 40-70%; this is only a recommendation and can be outside of this range if required. 

To generate the unlabelled data, take your ROI image file and run Vol-Seg at default settings (no advanced settings or additions) on *'slicer mode '*, only allocating the `--data` argument (no labels). This will slice the ROI data into a *'data directory'* within your Vol-Seg directory that can then be saved to your user space in an easily accessible folder. 

```shell
conda activate "path_to_env/Vol_Seg-env"
cd /users/'Individual_User'/libs/volume-segmantics
model-train-2d --data 'Path_to_ImageROI_File' --mode=slicer
```

Once generated, it may be used for either/both *Mean Teacher* and *Pseudo-label* self-supervised training modes.

#### - Self-Supervised training;

*Mean Teacher* and *Pseudo-label* training settings are used to create self-supervised training models, allocating your unlabelled data to the argument `--unlabeled_data_dir` within the training execution and producing potentially better predictions depending on your input image and label quality and overall segmentation aims. Specific settings for the *Mean Teacher* and *Pseudo-label* settings can be found in the subsequent sub-section. 

> - An example command can be found below, where the usual command for model training can be used; with the additional allocation of unlabelled data. 

```shell
model-train-2d --data 'Path_to_Image_File' --labels 'Path_to_Label_File' --unlabeled_data_dir='Path_to_Unlabeled_Data_dir'
```

#### - Mean Teacher

~Mean teacher explanation~ ; AVERY for background and checks +++++

#### - Pseudo-label

~Pseudo-label~ ; AVERY for background and checks +++++
