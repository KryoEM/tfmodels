# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# def random_flip_polarity(image):
#     sel = tf.random_uniform([], maxval=2, dtype=tf.int32)
#     return tf.cond(tf.equal(sel,0),lambda:image,lambda:-image)

def preprocess_common(image, height, width):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    channels = image.get_shape()[2]
    tf.summary.image('initial_image', tf.expand_dims(image, 0))
    image.set_shape([height, width, channels])
    # normalize std
    image = tf.image.per_image_standardization(image)
    # normalize image mean
    # image -= tf.reduce_mean(image)
    return image

def preprocess_for_train(image, height, width, scope=None):
    """ """
    with tf.name_scope(scope, 'distort_image', [image, height, width]):
        image = preprocess_common(image, height, width)
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(image)
        # Randomly distort the colors. There are 4 ways to do it.
        #distorted_image = random_flip_polarity(distorted_image)
        tf.summary.image('final_distorted_image',tf.expand_dims(distorted_image, 0))
        return distorted_image

def preprocess_for_eval(*args):
    return preprocess_common(*args)

def get_preprocess_image(is_training=False):
    """ """
    if is_training:
        return preprocess_for_train
    else:
        return preprocess_for_eval


