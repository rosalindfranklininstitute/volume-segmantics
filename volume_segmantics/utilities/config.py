"""Data to be shared across files.
"""
# Parser strings
TRAIN_DATA_ARG = "data"
LABEL_DATA_ARG = "labels"
MODEL_PTH_ARG = "model"
PREDICT_DATA_ARG = "data"
DATA_DIR_ARG = "data_dir"
PREDICT_DATADIR_ARG = "data_image_dir"
OUTPUT_DATA_DIR_ARG = "output"
# File extensions
TIFF_SUFFIXES = {".tiff", ".tif"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".nxs"}
ZARR_SUFFIXES = {".zarr"}
PNG_SUFFIXES = {".png"}
TRAIN_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
LABEL_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES}
MODEL_DATA_EXT = {".pytorch", ".pth"}
PREDICT_DATA_EXT = {*HDF5_SUFFIXES, *TIFF_SUFFIXES, *PNG_SUFFIXES, *ZARR_SUFFIXES}
# TODO Required settings - check required keys are in settings files
# Logging format
LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"
# Settings yaml file locations
SETTINGS_DIR = "volseg-settings"
TRAIN_SETTINGS_FN = "2d_model_train_settings.yaml"
PREDICTION_SETTINGS_FN = "2d_model_predict_settings.yaml"

TQDM_BAR_FORMAT = "{l_bar}{bar: 30}{r_bar}{bar: -30b}"  # tqdm progress bar format

OUTPUT_FORMAT = "tif" # tif or hdf
HDF5_COMPRESSION = "gzip"

BIG_CUDA_THRESHOLD = 16 # GPU Memory (GB), above this value batch size is increased
BIG_CUDA_TRAIN_BATCH = 16 # Size of training batch on big GPU
BIG_CUDA_PRED_BATCH = 4 # Size of prediction batch on big GPU
SMALL_CUDA_BATCH = 8 # Size of batch on small GPU
NUM_WORKERS = 8 # Number of parallel workers for training/validation dataloaders
PIN_CUDA_MEMORY = True # Whether to pin CUDA memory for faster data transfer
IM_SIZE_DIVISOR = 32 # Image dimensions need to be a multiple of this value
MODEL_INPUT_CHANNELS = 3 # Use 1 for grayscale input images, 3 for RGB (2.5D)
USE_ALL_GPUS = False

DEFAULT_MIN_LR = 0.00075 # Learning rate to return if LR finder fails
LR_DIVISOR = 3 # Divide the automatically calculated learning rate (min gradient) by this magic number

IMAGENET_MEAN = 0.449 # Mean value for single channel imagnet normalisation
IMAGENET_STD = 0.226 # Standard deviation for single channel imagenet normalisation

IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD = [0.229, 0.224, 0.225]

def get_model_input_channels(settings=None):
    if settings and getattr(settings, 'use_2_5d_slicing', False):
        return getattr(settings, 'num_slices', 3)  # Return number of slices for 2.5D
    return MODEL_INPUT_CHANNELS  # Default to 1 channel for 2D slicing

def get_imagenet_normalization(settings=None):
    if settings and getattr(settings, 'use_2_5d_slicing', False):
        num_channels = getattr(settings, 'num_slices', 3)
        # For 2.5D, use single channel normalization repeated for all channels
        return [IMAGENET_MEAN] * num_channels, [IMAGENET_STD] * num_channels
    return IMAGENET_MEAN, IMAGENET_STD  # Single channel normalization for 2D
