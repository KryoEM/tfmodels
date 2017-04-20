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
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
# from   tools.io import get_filenames_and_classes
import numpy as np
import random
from   scipy import misc
#from   tools.image import ImageReader
#from six.moves import urllib
import tensorflow as tf

slim = tf.contrib.slim

LABELS_FILENAME = 'labels.txt'

def get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.
  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  directories = []
  class_names = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)
  filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      filenames.append(path)
  return filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir,split_name,shard_id,num_shards):
    output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name,shard_id,num_shards)
    return os.path.join(dataset_dir, output_filename)

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

def write_label_file(labels_to_class_names, dataset_dir,filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))

def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))

def read_meta_params(split_name,dataset_dir):
    metafile = os.path.join(dataset_dir, split_name)
    return np.load(metafile + '.npy').tolist()

def get_dataset(split_name,dataset_dir,channels=1,reader=None):

  #if split_name not in SPLITS_TO_SIZES:
  #  raise ValueError('split name %s was not recognized.' % split_name)

  file_pattern = '%s_*.tfrecord'

  assert split_name in ['train', 'test']
  params = read_meta_params(split_name,dataset_dir)

  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  items_to_descriptions = {
      'image': 'An image of varying size.',
      'label': 'A single integer between 0 and 1',
  }

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      # ask to provide channels as in the file
      'image': slim.tfexample_decoder.Image(channels=channels),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      #'height':params['nsamples'],
      #'width':params['width'],
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)

  # labels_to_names = None
  # if has_labels(dataset_dir):
  #   labels_to_names = read_label_file(dataset_dir)

  return slim.dataset.Dataset(data_sources=file_pattern,reader=reader,decoder=decoder,
                              num_samples=params['nsamples'],items_to_descriptions=items_to_descriptions,
                              num_classes=params['nclasses']) #labels_to_names=labels_to_names


def convert_dir_classes(data_in_dir,data_out_dir,split_name,num_shards):
  '''
  Converts directory structure with different classes of images stored in different directories to
  a TFRecord file structure
  :param data_in_dir:  input directory with subdirectories with classes
  :param data_out_dir: output directory for writing TFRecord files
  :param split_name:   identifier for selecting training/validation/test selecting
  :param num_shards:   number of shards to store for parallel processing
  :return: None
  '''
  assert split_name in ['train', 'validation']
  fnames,classes      = get_filenames_and_classes(data_in_dir)
  nfiles              = len(fnames)
  #shuf_idxs           = np.arange(nfiles)
  random.shuffle(fnames)
  #fnames,classes      = np.array(fnames)[shuf_idxs],np.array(classes)[shuf_idxs]
  #fnames,classes      = get_filenames_and_classes(os.path.join(data_in_dir,split_name))
  num_per_shard       = int(np.ceil(nfiles) / float(num_shards))
  class_names_to_ids  = dict(zip(classes, range(len(classes))))
  im = misc.imread(fnames[0])
  height,width=im.shape[0:2]
  # save image size and number of samples params
  np.save(os.path.join(data_out_dir,split_name),
          {'height':height,'width':width,'nsamples':len(fnames),'nclasses':len(classes)})
  #write_label_file(dict(zip(range(len(classes)), classes)),data_out_dir)
  with tf.Graph().as_default():
      with tf.Session('') as sess:
          for shard_id in range(num_shards):
              file_out = _get_dataset_filename(data_out_dir,split_name,shard_id,num_shards)
              with tf.python_io.TFRecordWriter(file_out) as tfrecord_writer:
                  #tfrecord_writer = tf.python_io.TFRecordWriter(file_out)
                  start_ndx = shard_id * num_per_shard
                  end_ndx   = min((shard_id + 1) * num_per_shard, len(fnames))
                  for i in range(start_ndx, end_ndx):
                      sys.stdout.write('\r>> Converting image %d/%d shard %d of %d' % \
                                       (i+1,len(fnames),shard_id,num_shards))
                      sys.stdout.flush()
                      image_data = tf.gfile.FastGFile(fnames[i], 'r').read()
                      # image       = image_reader.decode_png(sess, image_data)
                      class_name  = os.path.basename(os.path.dirname(fnames[i]))
                      class_id    = class_names_to_ids[class_name]
                      example     = image_to_tfexample(image_data, 'png', height, width, class_id)
                      tfrecord_writer.write(example.SerializeToString())
