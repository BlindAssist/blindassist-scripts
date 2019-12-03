from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from model import Deeplabv3
from model import relu6
import coremltools

import tensorflow as tf
from tensorflow.python.keras.utils import CustomObjectScope

MODEL_DIR = './models'
CHECKPOINT_LOCATION = './checkpoint.h5'
MLMODEL_NAME = 'cityscapes.mlmodel'

print('Instantiating an empty Deeplabv3+ model...')
tf_model = Deeplabv3(input_shape=(384, 384, 3), classes=19)

WEIGHTS_DIR = 'weights/mobilenetv2'
print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(tf_model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)

print('Loaded weights. Summary of model:\n')
tf_model.summary();
print('')

print('Saving model...')
tf_model.save(CHECKPOINT_LOCATION)

print('Converting...')
with CustomObjectScope({'relu6': relu6}):
    coreml_model = coremltools.converters.tensorflow.convert(CHECKPOINT_LOCATION,
                        input_names=['input_1'],
                        image_input_names='input_1', 
                        output_names='lambda_3')

coreml_model.author = 'Giovanni Terlingen'
coreml_model.license = 'MIT License'
coreml_model.short_description = 'Produces segmentation info for urban scene images.'

coreml_model.save(MLMODEL_NAME)
print('Model converted, optimizing...')

# Load a model, lower its precision, and then save the smaller model.
model_spec = coremltools.utils.load_spec(MLMODEL_NAME)
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, MLMODEL_NAME)

print('Done.')
