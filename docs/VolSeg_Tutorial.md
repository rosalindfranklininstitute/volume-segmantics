# VolSeg Tutorial

## Training and Prediction of High-Resolution Human Placenta Segmentation Data (WINSdata)

The following tutorial explains and demonstrates an example use of the Volume-Segmantics Segmentation Toolkit using a 3D scan volume of the human placenta. The following instructions provided shows you how to view and inspect the data volume, crop a smaller Region of Interest (ROI), train a segmentation model using this ROI-crop and then use this model to obtain a prediction, finally calculating a DiceScore relative to its original GroundTruth representing the model's accuracy.

To use this tutorial, you will need a virtual environment with `Volume-Segmantics` installed, and a separate virtual environment with both `JupyterLab` and `Napari` packages installed. Further documentation detailing the installation of *Volume-Segmantics* can be found in the initial [ReadMe](https://github.com/rosalindfranklininstitute/volume-segmantics/blob/vs04b4/ReadMe.md) or associated documentation [here](https://github.com/rosalindfranklininstitute/volume-segmantics/tree/vs04b4/docs).

> **JupyterLab** is a well-known development environment for using coding notebooks, and runs seemlessly with the scripts and notebooks curated for this particular tutorial. Guidance surrounding the installation of [Jupyter](https://jupyter.org/) can be found [here](https://docs.jupyter.org/en/latest/), alongside a [helpful cheat sheet](https://github.com/iAreful/python-quick-view/blob/main/jupyter-cheat-sheet.md) for first-time users.
> - Within the environment you are using to run jupyterlab, to use the notebooks related to this tutorial you will also need to install the following packages dependencies in order to use their contents; this can be done after your installation of JupyterLab using pip and you will be reminded of this fact later on in the tutorial.
>> - pip install numpy
>> - pip install scipy
>> - pip install h5py
>> - pip install tifffile==2022.8.12 #Most stable version for usage.
>> - pip install monai
>> - pip install torch==2.5.1 #Minimum version or later.

> **Napari** is a well-known interactive image viewer and editor for biological data and segmentation, and is the default/recommended software for viewing images and label inputs while using VolSeg. Guidance surrounding the installation of [Napari](https://napari.org/stable/index.html) can be found [here](https://napari.org/stable/getting_started/installation.html); We recommend using version 0.5.5 or later for this tutorial. Once installed and opened, you will need to make sure you have the *napari builtins* plugin (dimensional viewer) installed alongside the *napari-h5* plugin (reader of .h5 files); 
> - To install or update/check the exsistence of your current plugins, Go to Plugins> Install/Uninstall Plugins tab within in an Napari Window and open the plugin manager; your installed Napari plugins will be present in the upper 'installed plugins' section of the pop-up window. 
>> - You will need to add both the *napari builtins* and *'napari-h5'* plugin to your installation by searching for the package in the top search bar of the pop-up window; plugin options relative to your search will appear in the lower section of the pop-up window and pressing the blue install button will install highlighted plugins. 
>> - You will likley need to reload your Napari session after installing plugins to affect the plugin changes. 
>> - Other useful plugins include *napari-animation* and *napari-chatgpt*; a list of additional plugins that may be of interest can be also found [here](https://napari-hub.org/).
> - Helpful guides for first time users can be found [here](https://napari.org/stable/howtos/index.html#how-tos).

## Step 1 - Viewing the image data;

After creating and loading your environment, you can look at the tutorial data. The dataset is a collection of files: a .tif image file containing a 700 cubic voxel 3D-slice of microCT scan data, and two .tif label files illustrating two biological components of the microCT scan data: the 'Villi' and 'Vessels'. *The original data is much larger (2000 cubic voxels), hence a section of tthe data has been taken for this tutorial.* The label-layer files are the best versions of the dataset currently available and are referred to as GroundTruth Images. These files can be found in the [Linked Tutorial Materials](https://github.com/SamK-RFI/volume-segmantics/tree/main/docs). 

> For the purposes of the tutorial, we will be refering only to the 'Vessel' label file, however the 'Villi' data can also be used as test data in its place if you wish to conduct further attempts. 

The diagram below shows screenshot examples of the files availible for this tutorial. 

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/WINSdata_Example.png" width="100%">
</div>


To view the data in Napari for yourselves, the Napari application needs to be opened using a terminal. The image and label files must be copied to your user space from the training materials to be usable; drag them into the open Napari window once they are in an easily accesable space.

> 1 - Open a Napari session instance, using your created Napari/Jupyter environment
> - A Napari terminal should open automatically though may take a second to load.
> 2 - Drag and drop the Image .tif file into the Napari window, followed by the Vessel label-layer .tif. 
> - *The label file is superimposed over the image as it is imported second, however the layer order can be changed by manually resorting the inputs with the layer list*. 
> - If the files have trouble opening using a drag-and-drop, you can also open the files using the file tab within the Napari window. 

You can use the Napari window and the diagram/key below to view the data in both 2D and 3D; inspect the data for quality, anomalies/inconsistencies, contrast variability, scan artefacts etc.

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/Napari_Ex.png" width="100%">
</div>

> - **(1)** Shuffle label colours **(2)** Erase tool **(3)** Paint tool **(4)** Bucket tool **(5)** Pan tool **(6)** Opacity bar **(7)** Label Selector **(8)** Tool applicator-size bar **(9)** Tool definition selector (2D or 3D) **(10)** New layer buttons (Points, Labels, Image) **(11)** Delete layer **(12)** Layer List, and Level selector **(13)** Command line **(14)** View shifter (2D/3D modes) **(15)** Change axis view (X, Y, Z) **(16)** Rotate plane (2D) **(17)** Navigator; Filename, Cursor place (3D), Current Axis view **(18)** Animate data and current data view (0-Z axis. 1-Y axis, 2-X axis) **(19)** Slice scroll bar **(20)** Slice viewer (current slice/overall slice total).

## Step 2 - Choosing your Training ROI;

To navigate numerically around your data within Napari in 3D space, use the *Navigator* (labelled as **17** on the napari diagram/key); the cursor's location when on top of the data in the viewer will show you a 3-point coordinate and your current acix view with the format `(Z, Y, X) 'axis view'`. 

Once you are familiar with the dataset, you can then select an ROI for your training model. Your ROI should be representative of the data, illustrating the main components and distinguishing features. *It is important to note that good quality models come from providing Volume-Segmantics with a good representation of numerous factors: realisation of background (non-label) vs your components (label), clear image and label feature recognition (high-quality image and accurate labels), and mitigation of observable artefacts (Avoiding scan noise and/or scan imperfections).*

The ROI for this tutorial should be no larger than *250 cubic voxels*; when choosing your ROI you should have a clear vision of where your volume starts and ends along each axis (Z, Y, X). Your ROI volume will be represented by 6-integer co-ordinate in 3D space with the format *(rangeZ, rangeY, rangeX)*, where each individual range is expressed as two integers; a starting slice and ending slice (Zstart:Zend, Ystart:Yend, Xstart:Xend) seperated by a ":", the difference between the two equating to the length, width and height. 

The recommended approach for notating your chosen ROI uses the following steps;

>1 - Inspect your main image in all 3 axis and choose the area of the image/label you wish to crop within the images boundaries; be mindful of the size of the image you wish to create (this volume can have any dimension however the script will throw an error if the area encompassing resides outside of the image/labels designated space). The dimensions should conform to the format *"range_z (Zs:Ze)", "range_y (Ys:Ye)", "range_x (Xs:Xe)"*.
>
>2 - Navigate to the Z-axis view and scroll (using the *Slice scroll bar*) to the earliest slice (left-most) on the current plane which you wish to use; this is the starting Z-axis co-ordinate (Zs).
> 
>3 - Use your cursor, and visualise a starting square around your chosen crop area; place your cursor in the top-left-hand corner of this square and observe the numbers displayed in the napari builtins display.
> - Your cursor placement in *Napari Builtins plugin* will show your current slice location (in the Z-axis position, which should not change when you move your cursor), alongside the position of the starting Y and X axis positions in the following format: (Zs, Ys, Xs).
> - *The Y coordinate will start from the top of the image and increase as you move down through the image/label layers in the Y-direction, whereas the X coordinate will start from the left and increase as you move right across the image/label layers in the X-direction*.
> - Make a note of these 3 numbers and input them into your 6-point coordinate reference.
>>>> *It can also be usefuly to imagine your Y-X square on this starting Z-axis plane during the step to help you identify the boundaries of your ROI more easily.*
>   
>4 - Scroll right along the Z-axis using the *Slice scroll bar* until you view the last slice on the current plane which you wish to use; this is the ending Z-axis co-ordinate (Ze). *This should be 250 voxels from the starting Z-slice for the purposes of this tutorial.*
>   
>5 - Use your cursor, and again visualise the square around your chosen crop area; place your cursor in the bottom right-hand corner of this square and observe the numbers displayed in the napari builtins display; *the distance from thair starting conterpart should be 250 voxels.*
> - Your cursor placement will show your current slice location (in the Z-axis position, which should not change when you move your cursor), alongside the position of the ending Y and X axis positions in the following format; (Ze, Ye, Xe).
> - Make a note of these 3 numbers and input them into your 6-point co-ordinate reference.

>  **Note**; when forming the 6-point co-ordinate reference, the numbers should form 3 pairs, the difference in each pair will be the range of the length, width and breadth of your volume, and should match any previous ideas of set size. The size of the crop will be checked numerically (the '.shape' command) within the cropping short-script in the next step. 

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/Napari_ROI.png" width="100%">
</div>

## Step 3 - Cropping your Data;

Once you have your ROI 6-integer co-ordinate noted, you can then make a crop of the original data files and save them as .tif files using the script below; the code should be input into the command line of your Napari window (labelled as **13** on the napari diagram/key. not the Napari terminal), interacting with your individual layers directly. The image and label ROI crops should superimpose over eachother exactly when using the same 6-integer co-ordinates to create a 'cut-out' of your image-label pair. 

*Use the code for creating your *image crop first*; this will also give you a chance to check your crop dimensions and test your intended visualisation of the crop before applying the same rationale to your labels layer.*

```shell
img = viewer.layers['Layer'].data
img.shape 
 
img_roi = img["range_z", "range_y", "range_x"]
img_roi.shape

viewer.add_image(img_roi)
```
> Assign the image layer to the first section of the code; if the image file was input first into Napari, this should be the lowest layer within your Napari layer list (0). **'Layer' replaced by 0**
> - This section of the code will also let you view the total shape of the original image data; *img.shape* which should return '[700, 700, 700]'

> Input your ROI 6-integer co-ordinate into the crop designation line *img["range_z", "range_y", "range_x"]*, with ranges seperating start and end integers with a ':', and check the shape of your crop *img_roi.shape*; This should return as [height, width, length] ([250, 250, 250] for the purposes of the tutorial).
> - If the crop shape is wrong, you can amend the axis ranges accordingly to the required size and run the lines in the terminal again using the *up-arrow* to show the previously listed commands.

> Create your crop using the *add_image* command; *'img_roi'* represents the original image *within the bounds* of your chosen ROI dictated by the *img["range_z", "range_y", "range_x"]* designation.
> - This will create a new layer which is viewed instantly within the napari window

> Save this new layer (clicking and highlighting the newly created layer first) to an easily accessible directory of your choosing; File>Save Selected Layers...

After confirming that your ROI 6-integer co-ordinate for the image is as you intended, use the same principle on your label layer; make sure you designate the label layer in the script (not the image layer).

```shell
l = viewer.layers['Layer'].data  
l.shape

l_roi = l["range_z", "range_y", "range_x"]  
l_roi.shape

viewer.add_labels(l_roi)
```

> Assign the label layer to the first section of the code; if input second, this should be the second lowest layer within your napari layer list (1) **'Layer' replaced by 1**
>>> - to distinguish between the Image data and Labekl data, we replace 'img' with 'l' within the script.

> Save this new layer (clicking and highlighting the newly created layer first) to an easily accessible directory of your choosing; File>Save Selected Layers...

## Step 4 - Creating your training model and prediction label;

Once you have your image-label ROI pair (2 X 250Cube files), you can then train a Volume-Segmantics model using these crops and then utilise this model to predict the vessel components of the original 700Cube image. Creating your model and prediction will take 3 steps; 

### 1 - Initialise a Volume-Segmantics Session:

To use Volume-Segmantics, you will need a new terminal running a your VolSeg working environment with the correct packages installed. Once the environment is activated, you will need to navigate to your Volume-Segmantics directory within this terminal before running any linked commands;

> Initiate Volume-Segmantics terminal instance;
> - Activate the Volume-Segmantics environment and navigate to your volume-segmantics directory; *conda activate..* *cd path_to_volume-segmantics-directory*

### 2 - Create Training Model:

To train your segmentation model, you will need to run the following command in the Volume-Segmantics terminal;

```shell
model-train-2d --data 'directory_location_image' --labels 'directory_location_labels'
```

The command is split into 3 parts: The training programme, *model-train-2d*, the data designation *--data*, and the label designation *--labels*. Using the 250 voxel ROIs you have created, you will asign the full file-path locations for both the image (data) and labels (labels) into the correct parts of the script; these paths must specify the exact file ending in .tif and will tell VolSeg what data to look at when running the model training rather than a directory. 

> 'directory_location_image' = 'path_to_image_ROI.tif'
> 'directory_location_labels' = 'path_to_labels_ROI.tif'

<div align="center">
  <img src="_static/Images/Training_script.png" width="80%">
</div>

Before you run your training model, you should first observe and confirm your training settings; to do this, navigate to the *volseg-settings* folder within the volume-segmantics directory. The *.yaml files* within this directory specify the conditions your model will be trained under; they will be set to default, however a good practice is to make a written/visual note or copy of the files into your project space before it is run to keep track of the model's conditions. The most important setting inputs can be found below;

> - clip_data: True,
> - training_axis: All
> - image_size: 256 (*this should be changed from 512 to suit new ROI size, though can be kept the same*)
> - downsample: False
> - cuda_device: 0, 
> - num_cyc_frozen: 8, 
> - num_cyc_unfrozen: 5, 
> - model: type: "U_Net" 
> - encoder_name: "tu-convnextv2_base"
> - encoder_weights: "imagenet"
> - loss_criterion: "CombinedCEDiceLoss"
> - eval_metric: "MeanIoU", 
> - use_sam: False
> - augmentation_library: albumentations
> - use_2_5d_slicing: False,
> - use_multitask: false
> - use_semi_supervised: False
> - use_pseudo_labeling: False

> More information surrounding further details for Volume-Segmantics training settings and documnetation as to their useage can be found [here](https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/Docs_Settings-Guide.md)

As the training is running, you will observe the image a label data being sliced seperatly, an outline of your settings and details of your training programe set to run, followed by a set of 8 frozen and 5 unfrozen training epochs. When the training has completed, the model will be saved to the volume-segmantics directory you are currently navigated to; the parent file, containing 4 files, should be cut and paste into a suitably named (with no spaces) directory in an easily accessible place (ideally alongside your ROI and original dataset). 

> A good way to initially instect your model is to open both the model_prediction_image.png and model_loss-Plot.png; visually inspect the model screenshots and the graphs for errors or inconsistencies.

### 3 - Generate Model Prediction:

You can now use your model (trained on 250cube data) to predict the vessel components of the original (700cube data) image; to do this, you will run the following command in the *same* Volume-Segmantics terminal;

```shell
model-predict-2d 'directory_location_segmantics_training_model' 'directory_location_new_image'
```

The command is again split into 3 sections: The prediction program; *model-predict-2d*, and the model and image designations (input concurrently after the prediction command with no additional defining arguments). You will use your *.pytorch* model-file as your model designation and the original 700cube image .tif file as the image designation, copying the file paths into the respective locations within the script; the files should be pasted seperated by a space.

These paths must specify the exact files ending in your saved .pytorch and .tif filenames, and will tell VolSeg what data to look at when running the model prediction.

> 'directory_location_segmantics_training_model' = 'path_to_training_model.pytorch'
> 'directory_location_new_image' = 'path_to_image-700CUBE.tif'

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/Prediction_script.png" width=80%">
</div>

Before you run your prediction, you should first observe and confirm your prediction settings; to do this, again navigate to the volseg-settings folder within the volume-segmantics directory. The *.yaml files* conditions will again be set to default however, a good practice is to make a note of or copy the file into the same area as your copied training settings before it is run to keep track of the model output conditions. The most important setting inputs can be found below;

> - quality: medium, 
> - clip_data: True (same as training settings),
> - cuda_device: 0 (same as training settings), 
> - downsample: False (same as training settings), 
> - prediction_axis: Z, 
> - output_size: 256 (same as training settings; *this should be changed from 512 to suit new ROI size, though can be kept the same*), 
> - use_2_5d_prediction: False (same as training settings),
> - use_sliding_window: false 

> More information surrounding the Volume-Segmantics prediction setting can be found [here](https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/Docs_Settings-Guide.md)

When the prediction has completed, the prediction file will be saved to the volume-segmantics folder you are currently navigated to; you should cut and paste this into the same space/directory as its corresponding model/data or in an easily accessible place. 
*Models left in the VolSeg directory run the risk of being overwriten if multiple are run conconrrently without additional setup.*

## Step 5 - Calculating your DiceScore;

The 700cube prediction you have created will be output as either a .h5 or .tif file (depending on your config.py setup, though the default is .tif), both of which can be opened within napari. *If a .h5 file is output by your terminal, and you cannot open the file in Napari, you may need to update your Napari plugin settings ('napari-h5' installed).*

After opening the prediction file in Napari and when attempting to view your data for the first time, the layer will likely be in 'image-mode'; you can use your left-mouse-click to convert the data to labels (*convert to Labels*). Once the data appears as labels, you will need to re-save this amended file as it is reccomended that you keep the original prediction saved seperatly.

Use the Layer list (labelled as **12** on the napari diagram/key) to toggle on and off your viewed layers, their opacity and arrangement to view your prediction layer with respect to the original 700cube GroundTruth; this will allow you to visually compare the prediction and its features with the original segmentation. 

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/Prediction_Accuracy.png" width="100%">
</div>

To produce a numerical representation of your model's effectiveness, you will use a manual DiceScore script to measure your predictions' labels relative to the original GroundTruth labels; it does this by calculating the space the 3D prediction labels occupies relatively. In order to calculate this, you are going to use a **Jupyter notebook** created specifically for the tutorial; DiceScore and other metrics are also mesured during the training and prediction process, however for comparing specific label files it is best to use this method.

A curated Jupyter notebook ('Manual_DiceScore_MASTER.ipynb') for calculating this DiceScore using a GroundTruth and a prediction segmentation can be found in the [Linked Tutorial Materials](https://github.com/SamK-RFI/volume-segmantics/tree/main/docs), and in order to use the notebook you must first copy the original file to your user space open a Jupyter session and open the file. 

To open a Jupyter session, you will need to use a new terminal and activate your created environment; this should be the same environment as that used to open your napari session. *You cannot use the same terminal as the one already open for running Napari, as using it for running additional code will kill the current Napari session*.

> 1 - Open a Juptyer session instance, using your created Napari/Jupyter environment
> - *Make sure you have the Jupyter package dependances installed within the environment before you open the file; the executable cells will not work correctly without these installations.*
> - A Jupyter browser should open automatically though may take a second to load.
> 2 - Use the Jupyter navigation tab to locate the copied Jupter file 'Manual_DiceScore_MASTER.ipynb' and open it using the File-Navigator (location [3]) on the left side of the Jupyter interface.
>>> - Double-clicking the file will then open it in the main window alongside the lauch menu. Once opened, the notebook can then be interacted with and run. 

<div align="center">
  <img src="https://github.com/SamK-RFI/volume-segmantics/blob/main/docs/source/_static/Images/Jupyter.png" width="100%">
</div>

> - **(1)** Run; run selected cells/run all cells, **(2)** Kernel; interrupt/Reconnect/Restart Kernel session, **(3)** File-Navigator, **(4)** Table of Contents, **(5)** Viewer panel, **(6)** Notebook tab, **(7)** Executable cell (code), **(8)** Note cell (markdown).

The notebook has 2 sections; the **first** is an explanatory set of cells and notes that give instructions on how to use the notebook's script (a cell followed by a notes section, 24 cells total), the **second** is a set blank and runnable cell-batches (12 cells total per batch); these can be copied and pasted concurrently into the same notebook if more are required. You can use the 'Table of Contents' tab (location [4]) on the left side of the Jupyter interface to navigate the notebook sections more easily.

- Read through the instructions in the master notebook and execute the cells using your own data files to calculate your DiceScore using one of the blank-cell batches. You can input the information regarding the date and description if you so choose. To run an individual cell, use *shift-enter*. 

- Once obtained (a decimal where 1.0 is 100% segmentation overlap with GroundTruth), make a note of your DiceScore mesurement to compare your models efficiency against others that you might produce in the future. 

> After completeing the tutorial, the same rationale can be used many times to improve on your existing model; using alternative or multiple ROIs, different/advanced model settings or alternative test data ('Villi'). 

> Due to to the quality of the original data scan and segmentation, the Dicescore calculated for this data could range highly depending on your chosen ROI; in reality, other datasets may produce a consistently higher or lower score relative to its original scan quality or intended outcome.
