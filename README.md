# 02461-Exam-Project

## Virtual Environment

In this project we use a virtual enviroment built with Python 3.11.7 to ensure compability between operating systems. To set up this environment you can either use Python or VSCode:

* Python: `py -m venv venv`
* VSCode:
  1. Open `Command Palette` with ctrl+shift+p
  2. Select `Python: Create Environment` and use `venv`

When the virtual environment is set up, use following code in terminal to ensure correct python path:

```powershell
py -c "import os, sys; print(os.path.dirname(sys.executable))"
```

If you're getting an error, try using `python` instead of `py`.

When all this is done, use the following code in terminal to install package requirements.

```powershell
py -m pip install -r requirements.txt
```

## Cuda

To utilize the GPU for training the model you must install a CUDA Toolkit that is supported by your GPU, as well as a CUDA-supported version of [PyTorch](https://pytorch.org/get-started/locally/). If installed correctly, the initializer of ModelHandler should print out "cuda".

## Usage

Use the `-h` argument to get a list of all arguments.

# Model files

Model files (.pt) are saved in our [OneDrive](https://dtudk-my.sharepoint.com/:f:/g/personal/s234843_dtu_dk/EkReAW9hH-xErYLYcxcNkbcBD6G0a7i_TMyv7vxg9BtvWQ?e=cquPLM "Requires DTU email to access") due to storage limit on GitHub.
