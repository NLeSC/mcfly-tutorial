<p align="left">
  <img src="mcflylogo.png" width="200"/>
</p>

[![Build Status](https://travis-ci.org/NLeSC/mcfly-tutorial.svg?branch=master)](https://travis-ci.org/NLeSC/mcfly-tutorial)

This repository contains notebooks that show how to use the [mcfly](https://github.com/NLeSC/mcfly) software. Mcfly is a deep learning tool for time series classification.

The tutorial can be found in the notebook [notebooks/tutorial/tutorial.ipynb](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb). This tutorial will let you train deep learning models with mcfly on the [PAMAP2 dataset for activity recognition](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring).

Prerequisites:
- Python 2.7 or >3.5
- Have the following python packages installed:
  - jupyter
  - mcfly

## Installation jupyter
The tutorials are provided in Jupyter notebooks, which can be found in the folder notebooks.
To use a notebook, first install Jupyter, for instance through pypi:

`pip install jupyter`

This also installs other dependencies like Matplotlib and Numpy.
For more documentation on Jupyter: See the [official documentation](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/).

## Installation mcfly
Mcfly can be installed through pypi:

`pip install mcfly`

See https://github.com/NLeSC/mcfly for alternative installation instructions


## Running the notebooks
The tutorials can be run using Jupyter. From the tutorial root folder run:

`jupyter notebook`

There are two versions of the tutorial. The [standard tutorial](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb) is for self-learning. There is also a [version for workshops](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/workshop.ipynb) which is only expected to be used with the aid of an instructor.
