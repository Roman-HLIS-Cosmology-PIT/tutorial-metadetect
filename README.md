# tutorial-metadetect

Tutorial on metacalibration and metadetection.


# Installation

## Conda
Make conda environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

## Pip
To install the requirements using pip:
- Step 1: create your python environment with python and pip (tested with python>=3.12)
- Step 2: we nned to install some packages manually first:
```bash
pip install --upgrade setuptools wheel numpy
```
- Step 3: we can now install the rest of the libraries:
```bash
pip install -r requirements-pip.txt
```
