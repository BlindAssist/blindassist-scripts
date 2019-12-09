#!/bin/bash

brew install python@2
sudo pip install setuptools

sudo pip install tensorflow==2.0.0
sudo pip install tqdm
sudo pip install sympy

./install_coremltools.sh

python ./download.py
python ./convert.py
