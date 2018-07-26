from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file

def get_mobilenetv2_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('MobilenetV2_', '')
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # convert tensor name into the corresponding Keras layer weight name and save
        filename = get_mobilenetv2_filename(key)

        if (filename):
            path = os.path.join(output_folder, filename)
            arr = reader.get_tensor(key)
            np.save(path, arr)
            print("tensor_name: ", key)

CKPT_URL = 'http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
MODEL_DIR = 'models'
MODEL_SUBDIR = 'deeplabv3_mnv2_cityscapes_train'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

checkpoint_tar_mobile = get_file(
    'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    CKPT_URL,
    extract=True,
    cache_subdir='',
    cache_dir=MODEL_DIR)

checkpoint_file = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'model.ckpt')

extract_tensors_from_checkpoint_file(checkpoint_file, 
    output_folder='weights/mobilenetv2')
