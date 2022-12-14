<p align="left">
  <img src="mcflylogo.png" width="200"/>
</p>

[![CI Build](https://github.com/NLeSC/mcfly-tutorial/workflows/CI%20Build/badge.svg)](https://github.com/NLeSC/mcfly-tutorial/actions)

This repository contains notebooks that show how to use the [mcfly](https://github.com/NLeSC/mcfly) software. Mcfly is deep learning tool for time series classification.

## Tutorials
Currently we offer two tutorials here. 
Our main tutorial can be found in the notebook [notebooks/tutorial/tutorial.ipynb](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb). This tutorial will let you train deep learning models with mcfly on the [PAMAP2 dataset for activity recognition](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring).  

A comparable, slightly quicker tutorial can be found in the notebook [notebooks/tutorial/tutorial_quick.ipynb](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial_quick.ipynb). This tutorial will let you train deep learning models with mcfly on the [RacketSports dataset for activity recognition](http://www.timeseriesclassification.com/description.php?Dataset=RacketSports).

Prerequisites:
- Python 3.7 and above
- The following python packages have to be installed (also specified in requirements.txt file):
  - mcfly
  - jupyter
  - pandas
  - matplotlib
  - scipy
  - numpy

## Installation

 ```shell
python3 -m venv env
. env/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

## Running the notebooks
The tutorials can be run using Jupyter notebook. From the tutorial root folder run:

`jupyter notebook`

There are two versions of the tutorial. The [standard tutorial](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/tutorial.ipynb) is for self-learning. There is also a [version for workshops](https://github.com/NLeSC/mcfly-tutorial/blob/master/notebooks/tutorial/workshop.ipynb) which is only expected to be used with the aid of an instructor.
