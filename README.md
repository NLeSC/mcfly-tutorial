<p align="left">
  <img src="mcflylogo.png" width="200"/>
</p>

[![Build Status](https://travis-ci.org/NLeSC/mcfly-tutorial.svg?branch=master)](https://travis-ci.org/NLeSC/mcfly-tutorial)

This repository contains notebooks that show how to use the [mcfly](https://github.com/NLeSC/mcfly) software. Mcfly is deep learning tool for time series classification..

## Tutorials
Currently we here offer two tutorials. 
Our main tutorial can be found in the notebook [notebooks/tutorial/tutorial.ipynb](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb). This tutorial will let you train deep learning models with mcfly on the [PAMAP2 dataset for activity recognition](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring).  

A comparable, slightly quicker tutorial can be found in the notebook [notebooks/tutorial/tutorial_quick.ipynb](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial_quick.ipynb). This tutorial will let you train deep learning models with mcfly on the [RacketSports dataset for activity recognition](http://www.timeseriesclassification.com/description.php?Dataset=RacketSports).

Prerequisites:
- Python 3.5, 3.6, or 3.7
- Have the following python packages installed:
  - mcfly
  - jupyter
 

## Installation mcfly
Mcfly can be installed through pypi:

`pip install mcfly`

See https://github.com/NLeSC/mcfly for alternative installation instructions

## Installation jupyter
The tutorials are provided in Jupyter notebooks, which can be found in the folder notebooks.
To use a notebook, first install Jupyter:

`pip install jupyter`

For more documentation on Jupyter: See the [official documentation](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/)

## Installation on Windows
Windows users can best use [Anaconda 3.6](https://www.anaconda.com/download). 
* Create a new environment (Environments > Create…)
* Click the play button next to your environment and select ‘Open terminal’
* Type `conda install numpy scipy jupyter` and then `pip install mcfly`
* Click the play button again and select ‘open with Jupyter notebook’
* Navigate to the directory where you cloned this repository, where you can open the notebooks


## Running the notebooks
The tutorials can be run using Jupyter. From the tutorial root folder run:

`jupyter notebook`

There are two versions of the tutorial. The [standard tutorial](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb) is for self-learning. There is also a [version for workshops](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/workshop.ipynb) which is only expected to be used with the aid of an instructor.
