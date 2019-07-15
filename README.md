# BlindAssist model scripts
[![Build Status](https://travis-ci.com/BlindAssist/blindassist-scripts.svg?branch=develop)](https://travis-ci.com/BlindAssist/blindassist-scripts)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

These scripts will download the pretrained DeepLabv3+ model based on MobileNetv2 from the Tensorflow model
zoo. Since they trained their models on MobileNetV2, I decided to use their model as the base model for the
BlindAssist app.

# Download and convert the model to CoreML (macOS)
- Clone this repo
- Run `./convert_model.sh`

After that the model has been generated. Then add `cityscapes.mlmodel` to the BlindAssist application.

This scripts are based on the work of @bonlime which created DeepLabv3+ for Keras and @seantempesta who
made DeepLabv3+ working on CoreML.

Original source: https://github.com/bonlime/keras-deeplab-v3-plus

*   DeepLabv3+:

```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

*   MobileNetv2:

```
@inproceedings{mobilenetv22018,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```
