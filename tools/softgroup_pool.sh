#!/bin/bash

conda update conda -y
conda env remove -n softgroup

conda create -n softgroup python=3.9 -y
source activate softgroup

conda install -y -c conda-forge cudatoolkit-dev=11.3
conda install -y pytorch cudatoolkit=11.3 -c pytorch
pip install spconv-cu102
pip install -r requirements.txt
conda install -y -c conda-forge sparsehash

# python setup.py build_ext develop