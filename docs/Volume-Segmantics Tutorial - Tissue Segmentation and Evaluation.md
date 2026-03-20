# Volume-Segmantics_Working Example - High-Res. Human Placenta Segmentation (WINSdata)

The following tutorial explains and demonstrates an example of Vol-Seg Segmentation using a 3D scan volume of the human placenta; the image data has been selected and curated from the open source 'WINSdata' dataset, and the labels have been created by a team from the AI&I dept. of the Rosalind Franklin Institute, Harwell Campus, UK. The following instructions provided show you how to view and inspect the data volume, crop a smaller Region of Interest (ROI), train a segmentation model using this ROI-crop and use this model to obtain a prediction, and finally calculate a Dice Score relative to its Ground Truth representing the model's accuracy.

To use this tutorial, you will need a virtual environment with Volume-Segmantics (Vol-Seg) installed, and a separate environment with JupyterLab and napari installed (either as a single environment or separate environments for each package). To install these packages and programs, further documentation surrounding *Vol-Seg* can be found [here](TBC), and more detailed guidance about *Jupyter* and *napari* can be found [here](TBC). 

## Step 1 - Viewing the image data;

The dataset is a set of files: a .tif image file containing a 700-slice cube of microCT scan data, and two .tif label files illustrating two biological components of the microCT scan data: the 'Villi' and 'Vessels'. The label-layer files are the best versions of the dataset currently available and referred to as Ground Truth Images. These files can be found in the data directories linked to this tutorial [here](TBC). The label files can be superimposed over the microCT image and viewed in 2D and 3D; the diagram below shows screenshot examples of the files viewed in Napari.  

![width=200](TBC) #WINSdata figure

> - For first time users, only the Vessels data should be used. 

To view the data in Napari for yourselves, the Napari application needs to be opened using a terminal, the image and label files must be copied to an easily accessible directory in your user space, and then either dragged into the open Napari window or opened using the File>Open file navigator tabs.

> 1] - Open napari;
> - 1.1] Open a command line terminal and Activate the working environment using the *'conda activate'* command. 
> - 1.2] Navigate to your user space using the *'cd'* command and the directory path
> - 1.3] Open a napari session using python (simple command); *'napari'*

```shell
conda activate 'path_to_napari-jupyter-env'
cd /users/'individual_user'
napari
```

> 2] - Once opened, drag and drop the Image .tif file into the napari window, followed by the Vessel label-layer .tif. *The label file is superimposed over the image as it is imported second, however the layer order can be changed by manually resorting the inputs with the layer list*. If the files have trouble opening using a drag-and-drop, you can also open the files using the file tab within the napari window. 

You can use the napari window and the diagram/key below to view the data in both 2D and 3D; inspect the data for quality, anomalies/inconsistencies, contrast variability, scan artefacts etc.

![width=150](TBC) #Napari figure 1

> - **(1)** Shuffle label colours **(2)** Erase tool **(3)** Paint tool **(4)** Bucket tool **(5)** Pan tool **(6)** Opacity bar **(7)** Label Selector **(8)** Tool applicator-size bar **(9)** Tool definition selector (2D or 3D) **(10)** New layer buttons (Points, Labels, Image) **(11)** Delete layer **(12)** Layer List, and Level selector **(13)** Command line **(14)** View shifter (2D/3D modes) **(15)** Change axis view (X, Y, Z) **(16)** Rotate plane (2D) **(17)** Napari Builtins plugin; Filename, Cursor place (3D), Current Axis view **(18)** Animate data and current data view (0-Z axis. 1-Y axis, 2-X axis) **(19)** Slice scroll bar **(20)** Slice viewer (current slice/overall slice total).

## Step 2 - Choosing your ROI;

Once you are familiar with the dataset, you can then select an ROI for your training model. Your ROI should be representative of the data, illustrating the main components and distinguishing features. *It is important to note that good quality models come from providing Vol-Seg with a good representation of numerous factors: realisation of background (non-label) vs your components (label), clear image and label feature recognition (high-quality image and accurate labels), and mitigation of observable artefacts.*

The ROI should be no larger than 250^3 in volume (250-slice cube); to choose your ROI, have a clear vision of where your volume starts and ends along each axis (X, Y, Z). Your volume will be represented by co-ordinates in 3D space with the format *(rangeZ, rangeY, rangeX)*, where each individual range is expressed as 2 integers; a starting slice and ending slice (the difference between the 2 equating to the length, width and height at ~250 voxels). 

To navigate numerically around your data in 3D space, use the *Napari Builtins plugin* (labelled as **17** on the napari diagram/key); the cursor's location when on top of the data in the viewer will show you its 3-point coordinate location with the format (Z, Y, X). The recommended approach for notating your chosen ROI can be found in the *'Protocol; ROI cropping (image and label)'* section of the [RFI-developed_Workflow_and_Utility documentation](TBC); following these steps should produce a 6-integer co-ordinate of your ROI represented as *(Z-Z, Y-Y, X-X)*.  An example from a cube corner of the 700^3 cube ground truth would be *(0:250, 0:250, 0:250)*, is shown below.

> Do not use the '0:250, 0:250, 0:250' co-ordinates for the tutorial!

![width=150](TBC) #ROI example

## Step 3 - Cropping your ROI;

Once you have your ROI 6-integer co-ordinate written down, you can then make a crop of the original data files and save them as .tif files using the concluding script instructions found in *'Protocol; ROI cropping (image and label)'* section of the [RFI-developed_Workflow_and_Utility documentation](TBC). *This code has been copied below for ease of access and explanation purposes*. 

The code should be input into the command line of your Napari window, interacting with your individual layers directly. The ROIs image and labels should superimpose over each other exactly (using the same 6-integer co-ordinates) creating a smaller version of your 700CUBE image-label pair only smaller. Use this code for creating your image crop first; this will also give you a chance to check your crop dimensions and test the intended visualisation of your crop before applying the same rationale to your labels layer.

> Assign the image layer to the code; if the image file was input first into Napari, this should be the lowest layer within your Napari layer list (0) *img = viewer etc.*

> Input your ROI 6-integer co-ordinate into the crop designation line *img[~]*, and check the shape of your crop *img_roi.shape*; This should return as [250, 250, 250] if this is your intended crop shape.
> - If the crop shape is wrong, you can amend the axis ranges accordingly to the required size and run the lines in the terminal again using the *up-arrow* to show the previously listed commands.

> Create your crop using the *add_image* command; img_roi represents the original image within the bounds of your chosen ROI dictated by the img[~] designation.
> - This will create a new layer which can be viewed instantly within the napari window

> Save this new layer (clicking and highlighting the newly created layer first) to an easily accessible directory of your choosing; File>Save Selected Layers...

```shell
img = viewer.layers['Layer'].data
img.shape 
 
img_roi = img["range_z", "range_y", "range_x"]
img_roi.shape

viewer.add_image(img_roi)
```

After confirming that your ROI 6-integer co-ordinate for the image, use the same principle on your label layer; make sure you designate the label layer in the script (not the image layer).

> Assign the labels layer to the code; if input second, this should be the second lowest layer within your napari layer list (1) *img = viewer etc.*
 
> Save this new layer to the same directory as your image crop.

```shell
l = viewer.layers['Layer'].data  
l.shape

l_roi = l["range_z", "range_y", "range_x"]  
l_roi.shape

viewer.add_labels(l_roi)
```

After creating both your Image and Label 250-slice ROIs, be sure to save them to an easily accessible directory within your user space; Select the layer and File>Save Selected Layers. 

## Step 4 - Creating your training model and prediction label;

Once you have your image-label layer ROI pair, you can then train a VolSeg model using these crops and then utilise this model to predict the vessel components on the original image; effectively remaking the original 700CUBE labels file using your own segmentation model. Creating your prediction will take 3 steps; 

### 1] Initialise a Vol-Seg Session:

To use Vol-Seg, you will need a terminal running a specific working environment with the correct packages installed. This environment has already been created for this tutorial, and the files connected to that environment can be viewed in the *'volume_segmantics'* directory that has been copied into your individual user space. Once the environment is activated, you will need to navigate to this directory in order to run its packages.

You will run the packages from within the command terminal, so keep it open and in view. 

> Initiate Vol-Seg;
> - 1.1] Open a command line terminal and Activate the Vol-Seg environment using the *'conda activate'* command
> - 1.2] Navigate to the volume-segmantics directory, from which you can run the training packages; *cd path_to_volume-segmantics-directory*

```shell
conda activate 'path_to_Vol-Seg-env'
cd users/'individual_user'/'path_to_volume-segmantics-directory'
```

### 2] Create Training Model:

To train your segmentation model, you will run the following command in the Vol-Seg terminal;

```shell
model-train-2d --data 'directory_location_image' --labels 'directory_location_labels'
```

The command is split into 3 parts: The training programme, *model-train-2d*, the data designation *--data*, and the label designation *--labels*. Using the annotated ROI you have created, you will copy the full file-path locations for both the image (data) and labels (labels) into the correct parts of the code; these paths must specify the exact file ending in .tif and will tell Vol-Seg what data to look at when running the model training rather than a directory. 

> 'directory_location_image' = /'user'/'individual_user'/'path_to_image_ROI.tif'

> 'directory_location_labels' = /'user'/'individual_user'/'path_to_labels_ROI.tif'

![width=150](TBC) #Training script

Before you run your training model, you should first observe and confirm your training settings; to do this, navigate to the *volseg-settings* folder within the volume-segmantics directory. The *.yaml files* within this directory specify the conditions your model will be trained towards; they will be set to default however, a good practice is to make a written/visual note of or copy the file into another area before it is run to keep track of the model's conditions. Please keep the Vol-seg training and predictions settings as default for the tutorial if attempting this for the first time. These can be altered to help you test out setting functionailty at a later date if you so wish. 

> - More information surrounding the volseg training setting can be found [here](TBC)

As the training is running, you will observe the data being sliced, a training epoch being run, and then a set of 8 frozen and 5 unfrozen training epochs working. When the training has completed, the model will be saved to the volume-segmantics folder you are currently located in; this will contain 4 files, of which you should cut and paste into a suitably named (no spaces) directory in an easily accessible place. It is recommended that a new folder is made and these 4 files are placed into it to make it easier to keep track of model creation and setting used.


> - A good way to initially inspect your model is to open both the model_prediction_image.png and model_loss-Plot.png; visually inspect the model screenshots and the graphs, where if the components seem to be correct and the 2 graph lines on the graph do not cross, the model will have been created as per its settings.


### 3] Generate Model Prediction:

You can now use your trained model to predict the vessel components of the original '700CUBE' image; to do this, you will run the following command in the *same* VolSeg terminal;

```shell
model-predict-2d 'directory_location_segmantics_training__model' 'directory_location_new_image'
```

The command is again split into 3 sections: The prediction program; *model-predict-2d*, the model designation, and an image designation (no separate argument allocations). You will use your created *.pytorch* model-file as your model designation and the original '700CUBE' image .tif file as the image designation, copying the file paths into the respective locations within the code; there are no argument attribute designations and the files should be pasted separated by a space. 

These paths must specify the exact files ending in your saved .pytorch and .tif filenames, and will tell Vol-Seg what data to look at when running the model prediction.

> 'directory_location_segmantics_training__model' = /ceph/'user'/'path_to_training_model.pytorch'

> 'directory_location_new_image' = /ceph/'user'/'path_to_image-700CUBE.tif'

![width=150](TBC) #Prediction code

Before you run your prediction, you should first observe and confirm your prediction settings; to do this, again navigate to the volseg-settings folder within the volume-segmantics directory. The *.yaml files* conditions will again be set to default. Please keep the Vol-seg training and predictions settings as default for the tutorial if attempting this for the first time. These can be altered to help you test out setting functionality at a later date if you so wish. 

> - More information surrounding the vol-seg prediction setting can be found [here](TBC)

When the prediction has completed, a .tif file will be saved to the volume-segmantics folder you are currently located in; you should cut and paste this into the same space/directory as its corresponding model or in an easily accessible place; rename the file as necessary.

## Step 5 - Calculating your DiceScore;

The '700CUBE' prediction you have created will be output as either a .h5, of which can be opened within napari. If you cannot open the file in Napari, you will need to amend your Napari plugin settings.

> Go to Plugins> Install/Uninstall Plugins... and open the plugin manager
> Your installed Napari plugins will be present in the upper 'installed plugins' section of the pop-up window;
> - You will need to add the *'napari-h5'* plugin to your installation by searching for the package in the top search bar of the pop-up window
> - Plugin options relative to your search will appear in the lower section of the pop-up window; press the blue install button to install the required plugin and wait until it confirms the update.

> You will need to reload your Napari session after installing this plugin to affect the plugin changes.
> - Go into the terminal where your Napari session is being run and use 'Ctrl-C' to cancel the Napari session. You may then use the 'up-arrow' to bring up the session opening code and initialise a new napari session *'python -m napari'*.

After opening the prediction file and when attempting to view your data for the first time, the layer will likely be in 'image-mode'; you can use your left-mouse-click to convert the data to labels (*convert to Labels*). Once the data appears as labels, you will need to re-save the file as it is recommended that you keep the original prediction saved as a separate file; the name of this file can be the same as the names previously or different, though you must only use and make changed to this new saved file moving forwards. 

Use the Layer list (labelled as **12** on the napari diagram/key) to toggle on and off your viewed layers, their opacity and arrangement to view your prediction layer with respect to the original '700CUBE' Ground Truth

- This will give you a complete visual representation of your model and predictions effectiveness.

![width=150](TBC) #Prediction visual accuracy

To produce a numerical representation of your model's effectiveness, we use a Dice Score to measure your predictions' labels relative to the original Ground Truth labels. In order to calculate this, we are going to use **Jupyter notebook**.

A notebook for calculating a DiceScore using a Ground Truth (the original '700CUBE') and a segmentation of the same size (your prediction labels, 700CUBE) can be found in the [Notebooks directory](TBC); this should be copied to your used space in an easily accessible location. In order to use the notebook, you must first open a Jupyter session. 

To open a Jupyter session, you will need to use a new terminal and activate the environment in which it is installed (possibly the same one as napari depending on your setup); *you cannot use the same terminal as the one already open for running Napari, as using it for running additional code will kill the current Napari session*;

> 1] - Open Jupyter;
> - 1.1] Open a command line terminal and Activate the working environment using the *'conda activate'* command
> - 1.2] Navigate to your user space using the *'cd'* command and the directory path
> - 1.3] Open a Jupyter session using a simple command; *'Jupyter Lab'*

```shell
conda activate 'path_to_working_env'
cd /users/'individual_user'
jupyter lab

# A Jupyter session should open automatically, though make take time to open on its first instance.
```

Once your Jupyter session is open, navigate to the file within your Jupyter sessions File-Navigator (location [3]) on the left side of the Jupyter interface; double-clicking the file will then open it in the main window alongside the launch menu. Once opened, the notebook can then be interacted with and run. 

![width=150](TBC) #Jupyter session

> - **(1)** Run; run selected cells/run all cells, **(2)** Kernel; interrupt/Reconnect/Restart Kernel session, **(3)** File-Navigator, **(4)** Table of Contents, **(5)** Viewer panel, **(6)** Notebook tab, **(7)** Executable cell (code), **(8)** Note cell (markdown).

> - The first cell of the notebook contains installation information regarding how to set up an environment to use the notebook; it may be necessary to install these packages using the terminal and environment used to open Jupyter before it can be used. If this is the case, close your Jupyter session, use pip install to make sure these packages and then restart your Jupyter session.

- The notebook has 2 sections; the **first** is an explanatory set of cells and notes that give instructions on how to use the notebook's code (a cell followed by a notes section, 24 cells in total), the **second** is a set blank and runnable cell-batches (12 cells total per use). You can use the 'Table of Contents' tab (location [4]) on the left side of the Jupyter interface to navigate the notebook sections more easily. if you require more runnable cells; blank ones can be copied and pasted beneath the current ones. 

- Read through the instructions in the notebook and execute the cells using your own data to calculate your DiceScore using one of the blank-cell batches. You can input the information regarding the date and description if you so choose. To run an individual cell, use *shift-enter* whilst the cell is selected. 

- Once obtained, make a note of your DiceScore to compare your model's efficiency against others that you might produce. 

 > More information regarding Jupyter and Napari utility and functionality can be found on the links within the [RFI-developed_Workflow_and_Utility documentation](TBC).
