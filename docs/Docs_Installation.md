# VolSeg Documentation; Installation and Setup Guides

The following guides detail both walkthroughs designed for users with limited coding expeirence, and developmental users. This documentation covers the setting up of virtual enviroments, a recomended procedure for installing the Volume-Segmentics package, and alternative instalation instructions for more advanced users using *poetry*. 

## Conda/Virtual Environments

When using Volume Segmantics (Vol-Seg), the recommended pathway to getting set up witn an environment is through Anaconda (conda); an open source, user friendly and well-known package management tool. Conda is used primarily due to its versatility and compatibility with other packages most commonly used in conjunction with Vol-Seg (Napari, JupyterLab). Instructions on how to download conda can be found on the [main software page](https://www.anaconda.com/download), and instructions as to its user functionality can be found on the [main documentation site](https://www.anaconda.com/docs/getting-started/main). 

Once downloaded, open a terminal instance and create a virtual environment to house your installation of VolSeg. It is also advisable that you create a folder in your user space in an easily accessible place to store your Vol-Seg and other environments you may create in the future; an example would be `/home/'user_space'/envs`. Creating an environment is done through a command line terminal using the following format, alongside deactivating it once you are finished; 

>1 - Navigate to istallation location using the *'cd'* command; e.g. *`cd /home/'user_space'/'envs`*
> - This can be done in stages using individual directory names until your reach the required location.
>2 - Create an environment using the *'conda create'* command; *`conda create "path_to_env" python="py_version"`*
> - `"path_to_env"` refers to the full directory path of the environment; `/home/'user'/envs/"env_name"` etc.
>> - `"env_name"` refers to the name of your environment; make the name easy to remember with no spaces e.g. 'VolSeg_env'
> - `"py_version"` refers to the version of Python used within the environment; this should be version 3.10 or higher
>
>2 - Activate the environment using the 'conda activate' command; *`conda activate "path_to_env"`*
> - `"path_to_env"` refers to the name of your full environment path including the environment name
>
>3 - Deactivate your environment when you are finished using the *'conda deactivate'* command and close the terminal; *`conda deactivate "path_to_env"`*

```shell
cd "/home/'user_space'/'envs'"
conda create -y -p "path_to_env/" python="py_version" #Python version should be 3.10 or higher
conda activate "path_to_env"
conda deactivate "path_to_env/"
```

A helpful selection of Python tutorials can be found at [W-3-Schools](https://www.w3schools.com/python/default.asp), and further information relating to Python documentation can be found [here](https://docs.python.org/3/) if you are not familiar with coding formats.

## VolSeg Installation; Default Walkthrough (PyPI)

After an environment has been created, make sure it is activated, then navigate back to your user space (*cd "/home/'user_space'* or similar). The VolSeg Package should be installed in a directory seperate from your environment. 

It is advisable that you create a folder in your user space an easily accessible space to store the VolSeg Installation, where the repository can be navigated to easily while viewing and altering its scripts; an example would be `/home/'user_space'/libs`, where "libs" refers to a package "libruary" designation. You can use a *'libs'* folder (or equivilent) you currently use, or an alternate location that is memorable; one can be created from the command line using the following steps, or it can be manually created using your file explorer; you can then install the Vol-Seg package using *pip*.  

>1 - Navigate to your user space within your terminal using the *'cd'* command; `cd "home/'user_space'"`
>
>2 - Make a *'libs'* (library) directory; `mkdir "libs"`, or further navigate to an easily accessible directory where you wish to house the Vol-Seg installation using the *'cd'* command
> - You will need to navigate into the newly made "libs" directory if one is made; `cd "users/'Individual_User'/'libs'"`
>
>3 - Install the Vol-Seg package using *pip*; it should then appear in the *'libs'* folder you have created/selected once finished;


```shell
cd /users/'Individual_User'  #Navigate to your user space/desired directory.
mkdir libs  #Make a 'libs' directory (if needed).
cd /users/'Individual_User'/libs  #Navigate to your housing/'libs' directory.

pip install volume-segmantics
```

Once installed, you will need to navigate into your installed VolSeg directory to begin using its commands; the commands are only executable from within the this directory, relying on the settings within the installation to execute.

```shell
cd volume-segmantics
```

If a CUDA-enabled build of PyTorch is not being installed by pip, you can use the adaptations below;

```shell
pip install volume-segmantics --extra-index-url https://download.pytorch.org/whl
```

- If you require different versions of PyTorch/CUDA as per your computer build capabilities (GPU), these can further be modified to suit your needs; uninstall the current version of PyTorch installed within the Vol-Seg installation, and reinstall the correct/preferred version. 

```shell
pip uninstall torch  #Uninstall previous torch dependencies.
pip install torch=='PyTorch_Version'  --index-url https://download.'full link'  #Install new torch dependencies.
```

> - Information as to the versions of PyTorch/CUDA available can be found using the python [get started](https://pytorch.org/get-started/locally/) link; the download path should be marked with the argument`--index-url` alongside the versions you require. Previous versions of PyTorch can also be found [here](https://pytorch.org/get-started/previous-versions/).

**Opt.** To keep track of your segmentation history, you can optionally install *'Tensorboard'*; a package that keeps track of your training and prediction outputs. Further information about using tensorboard can be found [here](https://www.tensorflow.org/tensorboard).

```shell
pip install tensorboard
```

## VolSeg Installation; Advanced version (Poetry)

The packages needed to use Vol-Seg and manage releases, updates and amend software and package parameters, can also be installed alternativly through 'poetry'; a dependency and packaging tool. This installation method is suited towards developmental users or those who wish to interact with installations on a deeper level. 

To install Vol-Seg in this way, you should first complete the initial steps outlined above; create and activate a virtual environment and create or navigate to a 'libs' folder (or alternative user directory /space) using a terminal. You can then follow the subsequant steps to install VolSeg;

>1 - Make sure you have your Vol-Seg conda environment activated and are navigated into your "libs" directory from the terminal; `conda activate "path_to_env/env_name"` and `cd "users/'Individual_User'/'libs'"`
>
>2 - Clone the Volume-Segmentics package directly from the online Git repo using the *'git clone'* command; this will then appear as a 'volume-segmantics' directory within your chosen or equivilent 'libs' folder: `"git clone --branch 'release_version' http://'Git_link'`
>
>3 - Navigate into the cloned Vol-Seg directory using the *cd* command; `cd "users/'Individual_User'/'libs'/volume-segmantics"`
>
>4 - Install the poetry tool using pip; `pip install poetry`
>
>5 - Install volume-segmantics using poetry; `poetry install`
> - The poetry tool will know the packages required for the installation by reading the cloned repo VolSeg file dependancies.
>
> - You may also be required to run the `*'poetry lock'*` command, depending on your computer setup; this will be prompted as an error message when to try to execute poetry install, where the *'poetry install'* command can be rerun after *'poetry lock'* has finished. 

```shell
conda activate "path_to_env/env_name" #Activate virtual environment
cd "users/'Individual_User'/'libs'" #Navigate to suitable user space
git clone --branch vs_'Version' https://github.com/'full_link'.git #Clone the VolSeg GitHub repo.
cd /ceph/users/'Individual_User'/libs/volume-segmantics #Navigate into the cloned VolSeg directory.
pip install poetry #Install Poetry 
poetry install  #Use Poetry to install VolSeg. 
```



