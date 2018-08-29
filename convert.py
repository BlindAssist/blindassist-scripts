from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from model import Deeplabv3
import coremltools
from coremltools.proto import NeuralNetwork_pb2

MODEL_DIR = 'models'
MLMODEL_NAME = 'cityscapes.mlmodel'

print('Instantiating an empty Deeplabv3+ model...')
keras_model = Deeplabv3(input_shape=(384, 384, 3), classes=19)

WEIGHTS_DIR = 'weights/mobilenetv2'
print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(keras_model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)

# CoreML model needs to normalize the input (by converting image bits from (-1,1)), that's why
# we are defining the image_scale, red, green, and blue bias
print('converting...')
coreml_model = coremltools.converters.keras.convert(keras_model,
                        input_names=['input_1'],
                        image_input_names='input_1', 
                        output_names='up_sampling2d_2',
                        image_scale=2/255.0,
                        red_bias=-1,
                        green_bias=-1,
                        blue_bias=-1)

coreml_model.author = 'Giovanni Terlingen'
coreml_model.license = 'GPLv3 License'
coreml_model.short_description = 'Produces segmentation info for urban scene images.'

coreml_model.save(MLMODEL_NAME)
print('model converted, optimizing...')

# Load a model, lower its precision, and then save the smaller model.
model_spec = coremltools.utils.load_spec(MLMODEL_NAME)
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, MLMODEL_NAME)

print('Done.')
