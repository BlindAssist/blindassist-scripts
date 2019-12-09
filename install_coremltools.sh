#!/bin/bash

brew install cmake

git clone https://github.com/apple/coremltools.git
git checkout master

cd coremltools
mkdir build
cd build

cmake ../
make dist
cd dist

sudo pip install wheel
sudo pip install coremltools-3.1-cp27-none-macosx_10_15_intel.whl

cd ../../../
