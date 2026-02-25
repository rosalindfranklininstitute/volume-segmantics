# Tutorial
## High-Resolution: Human Placenta Segmentation (WINSdata)
The following tutorial explains and demonstrates an example of Volume-Segmantics Segmentation using a 3D scan volume of the human placenta. The following instructions provided show you how to view and inspect the data volume, crop a smaller Region of Interest (ROI), train a segmentation model using this ROI-crop and use this model to obtain a prediction, and finally calculate a DiceScore relative to its GroundTruth representing the model's accuracy.

To use this tutorial, you will need a virtual environment with Volume-Segmantics installed, and a separate environment with JupyterLab and napari installed. To install these packages and programs, further documentation surrounding *Volume-Segmantics* can be found [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/edit/sam-branch/README-Updated-SK.md), and more detailed guidance about *Jupyter* and *napari* can be found [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/edit/sam-branch/RFI-developed_Workflow_and_Utility.md). 

*``
For the 13.11.25 'Segmentation with Volume Segmantics' seminar/tutorial, environments for Volume-Segmantics and Jupyter/Napari have already been created. These environments have been previously created and distributed to your individual user spaces on the ceph server; you will find a copy of each environment on your home directory when you log into your workstation session. The directory path should be in the format '/ceph/users/'individual_user'/'. Outside of this seminar tutorial, personal versions of these environments must be created; the documentation for this can be found at https://github.com/rosalindfranklininstitute/aiimg_scripts/edit/sam-branch/RFI-developed_Workflow_and_Utility.md.  
``*

## Step 1 - Viewing the image data;

The dataset a set of files: a .tif image file containing a 700-slice cube of microCT scan data, and two .tif label files illustrating two biological components of the microCT scan data: the 'Villi' and 'Vessels'. The label-layer files are the best versions of the dataset currently available and referred to as GroundTruth Images. These files can be found in the 'COMPUTER-VISION_TUTORIAL_DATA directory that has also been added to your individual user space. The label files can be superimposed over the microCT image and viewed in 2D and 3D; the diagram below shows screenshot examples of the files viewed in Napari.  

![width=200](_static/images/WINSdata-ViewFigure.png)

*``
For this tutorial, we will only be using the GT-Vessel Data. 
``*

To view the data in Napari for yourselves, the Napari application needs to be opened using a terminal, the image and label files must be copied to your user space within Ceph, and then drag them into the open Napari window.

> 1 - Open napari;
> - 1.1 Open a command line terminal and initiate a conda module session, *'module load miniconda'*
> - 1.2 Activate the working environment using the *'conda activate'* command; this has been named 'napari-jupyter'
> - 1.3 Navigate to your user space using the *cd'* command and the directory path
> - 1.4 Open a napari session using python (simple command); *'napari'*

```
module load miniconda
conda activate 'path_to_napari-jupyter-env'
cd /ceph/users/'individual_user'
napari

# A napari terminal should open automatically though may take a second to load. 
```

> 2 - Drag and drop the Image .tif file into the napari window, followed by the Vessel label-layer .tif. *The label file is superimposed over the image as it is imported second, however the layer order can be changed by manually resorting the inputs with the layer list*. If the files have trouble opening using a drag-and-drop, you can also open the files using the file tab within the napari window. 

You can use the napari window and the diagram/key below to view the data in both 2D and 3D; inspect the data for quality, anomalies/inconsistencies, contrast variability, scan artefacts etc.

![width=150](_static/images/napari_figure1.png)

> - **(1)** Shuffle label colours **(2)** Erase tool **(3)** Paint tool **(4)** Bucket tool **(5)** Pan tool **(6)** Opacity bar **(7)** Label Selector **(8)** Tool applicator-size bar **(9)** Tool definition selector (2D or 3D) **(10)** New layer buttons (Points, Labels, Image) **(11)** Delete layer **(12)** Layer List, and Level selector **(13)** Command line **(14)** View shifter (2D/3D modes) **(15)** Change axis view (X, Y, Z) **(16)** Rotate plane (2D) **(17)** Napari Builtins plugin; Filename, Cursor place (3D), Current Axis view **(18)** Animate data and current data view (0-Z axis. 1-Y axis, 2-X axis) **(19)** Slice scroll bar **(20)** Slice viewer (current slice/overall slice total).

## Step 2 - Choosing your ROI;

Once you are familiar with the dataset, you can then select an ROI for your training model. Your ROI should be representative of the data, illustrating the main components and distinguishing features. *It is important to note that good quality models come from providing Volume-Segmantics with a good representation of numerous factors: realisation of background (non-label) vs your components (label), clear image and label feature recognition (high-quality image and accurate labels), and mitigation of observable artefacts.*

The ROI should be no larger than 250^3 in volume; to choose your ROI, have a clear vision of where your volume starts and ends along each axis (X, Y, Z). Your volume will be represented by co-ordinates in 3D space with the format *(rangeZ, rangeY, rangeX)*, where each individual range is expressed as 2 integers; a starting slice and ending slice (the difference between the 2 equating to the length, width and height at ~250 voxels). 

To navigate numerically around your data in 3D space, use the *Napari Builtins plugin* (labelled as **17** on the napari diagram/key); the cursor's location when on top of the data in the viewer will show you a 3-point coordinate with the format (Z, Y, X). The recommended approach for notating your chosen ROI can be found in the *'Protocol; ROI cropping (image and label)'* section of the [RFI-developed_Workflow_and_Utility documentation](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/RFI-developed_Workflow_and_Utility.md); following these steps should produce a 6-integer co-ordinate of your ROI represented as *(Z-Z, Y-Y, X-X)*, an example of which, from a cube corner of the 700cube groundtruth *(0:250, 0:250, 0:250)*, is shown below.

> Do not use the '0:250, 0:250, 0:250' co-ordinates for the tutorial!

![width=150](_static/images/ROI_Example.png)

## Step 3 - Cropping your ROI;

Once you have your ROI 6-integer co-ordinate written down, you can then make a crop of the original data files and save them as .tif files using the concluding script instructions found in *'Protocol; ROI cropping (image and label)'* section of the [RFI-developed_Workflow_and_Utility documentation](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/RFI-developed_Workflow_and_Utility.md). *This code has been copied below for ease of access and explaination purposes*. 

The code should be input into the command line of your Napari window, interacting with your individual layers directly. The ROIs image and labels should superimpose over eachother exactly (using the same 6-integer co-ordinates) creating a smaller version of your 700CUBE image-label pair. Use this code for creating your image crop first; this will also give you a chance to check your crop dimensions and test the intended visualisation of your crop before applying the same rationale to your labels layer.

> Assign the image layer to the code; if the image file was input first into Napari, this should be the lowest layer within your Napari layer list (0) *img = viewer etc.*

> Input your ROI 6-integer co-ordinate into the crop designation line *img[~]*, and check the shape of your crop *img_roi.shape*; This should return as [250, 250, 250] if this is your intended crop shape.
> - If the crop shape is wrong, you can amend the axis ranges accordingly to the required size and run the lines in the terminal again using the *up-arrow* to show the previously listed commands.

> Create your crop using the *add_image* command; img_roi represents the original image within the bounds of your chosen ROI dictated by the img[~] designation.
> - This will create a new layer which can be viewed instantly within the napari window

> Save this new layer (clicking and highlighting the newly created layer first) to an easily accessible directory of your choosing; File>Save Selected Layers...

```
img = viewer.layers['Layer'].data
img.shape 
 
img_roi = img["range_z", "range_y", "range_x"]
img_roi.shape

viewer.add_image(img_roi)
```

After confirming that your ROI 6-integer co-ordinate for the image, use the same principle on your label layer; make sure you designate the label layer in the script (not the image layer).

> Assign the labels layer to the code; if input second, this should be the second lowest layer within your napari layer list (1) *img = viewer etc.*
 
> Save this new layer to the same directory as your image crop.

```
l = viewer.layers['Layer'].data  
l.shape

l_roi = l["range_z", "range_y", "range_x"]  
l_roi.shape

viewer.add_labels(l_roi)
```

## Step 4 - Creating your training model and prediction label;

Once you have your image-label layer ROI pair (250Cube), you can then train a Volume-Segmantics model using these crops and then utilise this model to predict the vessel components on the original 700Cube image; effectively remaking the original 700CUBE labels file using your own segmentation model. Creating your prediction will take 3 steps; 

#### 1 Initialise a Volume-Segmantics Session:

To use Volume-Segmantics, you will need a terminal running a specific working environment with the correct packages installed. This environment has already been created for this tutorial, and the files connected to that environment can be viewed in the *'volume_segmantics'* directory that has been copied into your individual user space. Once the environment is activated, you will need to navigate to this directory in order to run its packages.

You will run the packages from within the command terminal, so keep it open and in view. 

> Initiate Volume-Segmantics;
> - 1.1 Open a command line terminal and initiate a conda module session, *'module load miniconda'*
> - 1.2 Activate the Volume-Segmantics environment using the *'conda activate'* command; this has been named 'Volume-Segmantics-env' 
> - 1.3 Navigate to the volume-segmantics directory, from which you can run the training packages; *cd path_to_volume-segmantics-directory*

```
module load miniconda
conda activate 'path_to_Volume-Segmantics-env'
cd /ceph/users/'individual_user'/'path_to_volume-segmantics-directory'
```

#### 2 Create Training Model:

To train your segmentation model, you will run the following command in the Volume-Segmantics terminal;

```
model-train-2d --data 'directory_location_image' --labels 'directory_location_labels'
```

The command is split into 3 parts: The training programme, *model-train-2d*, the data designation *--data*, and the label designation *--labels*. Using your 250CUBE~ ROIs you have created, you will copy the full file-path locations for both the image (data) and labels (labels) into the correct parts of the code; these paths must specify the exact file ending in .tif and will tell Volume-Segmantics what data to look at when running the model training rather than a directory. 

> 'directory_location_image' = /ceph/'user'/'individual_user'/'path_to_image_ROI.tif'

> 'directory_location_labels' = /ceph/'user'/'individual_user'/'path_to_labels_ROI.tif'

![width=150](_static/images/training_script.png)

Before you run your training model, you should first observe and confirm your training settings; to do this, navigate to the *Volume-Segmantics-settings* folder within the volume-segmantics directory. The *.yaml files* within this directory specify the conditions your model will be trained towards; they will be set to default however, a good practice is to make a written/visual note of or copy the file into another area before it is run to keep track of the model's conditions. The most important setting inputs can be found below.

*``
A seperate copy of these settings has also been copeid to your indiviual user space within the COMPUTER-VISION_TUTORIAL_DATA directory; these are not the designated settings files that will be used for the training (you have have these files imbedded into the volume_segmantics directory), these are simply a backup in case either the main directory has a bug, or if you need to reset your settings back to default and need a reference. 
``*

> clip_data: True, use_2_5d_slicing: False, image_size: 256, cuda_device: 0, num_cyc_frozen: 8, num_cyc_unfrozen: 5, loss_criterion: "DiceLoss", eval_metric: "MeanIoU", encoder_weights_path: False, model: type: "U_Net" encoder_name: "resnet50" encoder_weights: "imagenet"
> - More information surrounding the Volume-Segmantics training setting can be found [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/README-Updated-SK.md)

As the training is running, you will observe the data being sliced, a training epoch being run, and then a set of 8 frozen and 5 unfrozen training epochs working. When the training has completed, the model will be saved to the volume-segmantics folder you are currently located in; this will contain 4 files, of which you should cut and paste into a suitably named (no spaces) directory in an easily accessible place. 

> **1** *.pytorch file*; the segmentation model created by the train command,
> **2** *model_loss-Plot.png*; a graph showing the training and validation loss over the course of the training epochs (a graphical representation of the model's success),
> **3** *model_prediction_image.png*; showing specific slices of the image and labels used within the model, and the resultant prediction test outcomes for that slice (a visual representation model's success),
> and **4** *stats.csv*; a record of the training loss, validation loss and evaluation score; accuracy of the test prediction generated versus the original label per epoch.

*``
A good way to initially instect your model is to open both the model_prediction_image.png and model_loss-Plot.png; visually inspect the model screenshots and the graph where if the components seem to be correct and the 2 lines on the graph do not cross, the model will have been created as per its settings.
``*

#### 3 Generate Model Prediction:

You can now use your model (trained on 250CUBE) to predict the vessel components of the original 700CUBE image; to do this, you will run the following command in the *same* Volume-Segmantics terminal;

```
model-predict-2d 'directory_location_segmantics_training__model' 'directory_location_new_image'
```

The command is again split into 3 sections: The prediction program; *model-predict-2d*, the model designation, and an image designation. You will use your *.pytorch* model-file as your model designation and the original 700CUBE image .tif file as the image designation, copying the file paths into the respective locations within the code; there are no argument attribute designations and the files should be pasted seperated by a space. 

These paths must specify the exact files ending in your saved .pytorch and .tif filenames, and will tell Volume-Segmantics what data to look at when running the model prediction.

> 'directory_location_segmantics_training__model' = /ceph/'user'/'path_to_training_model.pytorch'

> 'directory_location_new_image' = /ceph/'user'/'path_to_image-700CUBE.tif'

![width=150](_static/images/prediciton_script.png)

Before you run your prediction, you should first observe and confirm your prediction settings; to do this, again navigate to the Volume-Segmantics-settings folder within the volume-segmantics directory. The *.yaml files* conditions will again be set to default however, a good practice is to make a note of or copy the file into the same area as your copied training settings before it is run to keep track of the model output conditions. The most important setting inputs can be found below.

> quality: medium, clip_data: True, cuda_device: 0, downsample: False, prediction_axis: Z, output_size: 512, use_2_5d_prediction: False
> - More information surrounding the Volume-Segmantics prediction setting can be found [here](https://github.com/rosalindfranklininstitute/aiimg_scripts/blob/sam-branch/README-Updated-SK.md)

When the prediction has completed, a .tif file will be saved to the volume-segmantics folder you are currently located in; you should cut and paste this into the same space/directory as its corresponding model or in an easily accessible place.

## Step 5 - Calculating your DiceScore;

The 700CUBE prediction you have created will be output as either a .h5 or .tif file (depending on the Volume-Segmantics version), both of which can be opened within napari. If a .h5 file is output by your terminal, and you cannot open the file in Napari, you will need to amend your Napari plugin settings.

> Go to Plugins> Install/Uninstall Plugins... and open the plugin manager
> Your installed Napari plugins will be present in the upper 'installed plugins' section of the pop-up window;
> - You will need to add the *'napari-h5'* plugin to your installation by searching for the package in the top search bar of the pop-up window
> - Plugin options relative to your search will appear in the lower section of the pop-up window; press the blue install button to install the required plugin and wait until it confirms the update.

> You will need to reload your Napari session after installing this plugin to affect the plugin changes.
> - Go into the terminal where your Napari session is being run and use 'Ctrl-C' to cancel the Napari session. You may then use the 'up-arrow' to bring up the start code and initialise a new napari session *'python -m napari'*.

After opening the prediction file and when attempting to view your data for the first time, the layer will likely be in 'image-mode'; you can use your left-mouse-click to convert the data to labels (*convert to Labels*). Once the data appears as labels, you will need to re-save the file as it is reccomended that you keep the original prediction saved as a seperate file; the name of this file can be the same as the names previously or different, though you must only use and make changed to this new saved file moving forwards. 

Use the Layer list (labelled as **12** on the napari diagram/key) to toggle on and off your viewed layers, their opacity and arrangement to view your prediction layer with respect to the original 700CUBE GroundTruth

- This will give you a complete visual representation of your model and predictions effectiveness.

![width=150](_static/images/prediction_napari.png)

To produce a numerical representation of your model's effectiveness, we use a DiceScore to measure your predictions' labels relative to the original GroundTruth labels; it does this by calculating the space the 3D prediction labels occupy relatively. In order to calculate this, we are going to use a **Jupyter notebook** linked to the console's computing power to speed up the process. 

A "master Jupyter notebook" for calculating a DiceScore using a GroundTruth (the original 700CUBE) and a segmentation of the same size (your prediction labels, 700CUBE) can be found in the [Jupyter_Notebooks directory](https://github.com/rosalindfranklininstitute/aiimg_scripts/tree/sam-branch/Jupyter_Notebooks), and in order to use the notebook, you must first open a Jupyter session. 

To open a Jupyter session, you will need to use a new terminal and activate the same working environment as the one used to open Napari; *you cannot use the same terminal as the one already open for running Napari, as using it for running additional code will kill the current Napari session*;

> 1 - Open Jupyter;
> - 1.1 Open a command line terminal and initiate a conda module session, *'module load miniconda'*
> - 1.2 Activate the working environment using the *'conda activate'* command; this has been named 'napari-jupyter'
> - 1.3 Navigate to your user space using the *'cd'* command and the directory path
> - 1.4 Open a Jupyter session using a simple command; *'Jupyter Lab'*

```
module load miniconda
conda activate 'path_to_working_env'
cd /ceph/users/'individual_user'
jupyter lab

# A Jupyter session should open automatically, though make take time to open on its first instance.
```

Once your Jupyter session is open, navigate to the'Jupyter_Notebooks' folder on the GIThub repo where a copy the *'Manual_DiceScore_MASTER.ipynb'* file is stored and download it; this will then appear in your downloads folder in your home space. Navigate to the same file within your Jupyter session using the File-Navigator (location [3]) on the left side of the Jupyter interface; double-clicking the file will then open it in the main window alongside the lauch menu. Once opened, the notebook can then be interacted with and run. 

![width=150](_static/images/jupyter.png)

> - **(1)** Run; run selected cells/run all cells, **(2)** Kernel; interrupt/Reconnect/Restart Kernel session, **(3)** File-Navigator, **(4)** Table of Contents, **(5)** Viewer panel, **(6)** Notebook tab, **(7)** Executable cell (code), **(8)** Note cell (markdown).

*``
The first cell of the notebook contains installation information regarding how to set up an environment to use the notebook in question; the nessisary packages have already been installed into the environment you currently have activated for the purposes of this tutorial. IF you wante to run this notebook again, you will need to use an environment with these packages installed to usilize the notebook.
``*

- The notebook has 2 sections; the **first** is an explanatory set of cells and notes that give instructions on how to use the notebook's code (a cell followed by a notes section, 24 cells), the **second** is a set blank and runnable cell-batches (12 cells total per batch). You can use the 'Table of Contents' tab (location [4]) on the left side of the Jupyter interface to navigate the notebook sections more easily.

- Read through the instructions in the master notebook and execute the cells using your own data to calculate your DiceScore using one of the blank-cell batches. You can input the information regarding the date and description if you so choose. To run an individual  cell, use *shift-enter*. 

- Once obtained, make a not eof your DiceScore to compare your model's efficiency against others that you might produce. 

 > More information regarding Jupyter and Napari utility and functionality can be found on the links within the [RFI-developed_Workflow_and_Utility documentation](https://github.com/rosalindfranklininstitute/aiimg_scripts/edit/sam-branch/RFI-developed_Workflow_and_Utility.md).
