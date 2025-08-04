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
- Step 1: create your python environment with python and pip (tested with python>=3.13). You can use `pipenv` as follows:
```bash
pip install --upgrade pipenv
python -m pipenv install
python -m pipenv shell
```
- Step 2: we can now install the rest of the libraries: `pip install -r requirements-pip.txt`
