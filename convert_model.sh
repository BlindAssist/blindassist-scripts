#!/bin/bash

brew install python@2
sudo pip install setuptools

sudo pip install coremltools==3.1
sudo pip install tensorflow==2.0.0
sudo pip install tqdm
sudo pip install sympy

python ./download.py
python ./convert.py
