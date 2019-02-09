#!/bin/bash

brew install python@2

sudo pip install coremltools==2.0
sudo pip install tqdm==4.28.1
sudo pip install tensorflow==1.5.0
sudo pip install keras==2.1.6

python ./download.py
python ./convert.py
