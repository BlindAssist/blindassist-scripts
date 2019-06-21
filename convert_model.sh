#!/bin/bash

brew install python@2

sudo pip install coremltools==3.0b1
sudo pip install tensorflow==2.0.0a0
sudo pip install setuptools==41.0.1
sudo pip install keras==2.2.4
sudo pip install tqdm

python ./download.py
python ./convert.py
