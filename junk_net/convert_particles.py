##
import tensorflow as tf
from   data_utils import pour_mrcs2pngs
from   datasets.dataset_utils import convert_dir_classes
from   fileio import filetools as ft
from   utils import tprint

# import numpy as np
# from   fileio import mrc
# from   myplotlib import imshow,clf
# import os

_NUM_SHARDS = 5

data_png_dir = '/jasper/data/train_data_sets/junk_detection/png/'
tfrecord_dir = '/jasper/data/train_data_sets/junk_detection/tfrecord/'

tprint('Cleaning %s ...' % data_png_dir)
ft.rmtree_assure(data_png_dir)
ft.mkdir_assure(data_png_dir)

# Split Junk mrc files into pngs
pour_mrcs2pngs('/jasper/result/PKM2_WT/Extract/bad_6840_101/Movies/',data_png_dir,'junk')
# Split Good mrc files into pngs
pour_mrcs2pngs('/jasper/result/PKM2_WT/Extract/good_78505/Movies/',data_png_dir,'good',use_only=6840)

tprint('Cleaning %s ...' % tfrecord_dir)
ft.rmtree_assure(tfrecord_dir)
ft.mkdir_assure(tfrecord_dir)

convert_dir_classes(data_png_dir+'/train',tfrecord_dir,'train',_NUM_SHARDS)
convert_dir_classes(data_png_dir+'/test',tfrecord_dir,'test',_NUM_SHARDS)

tprint('Finished TFRecord Conversions!')

########

# _NUM_SHARDS = 5
#
# ##
import numpy as np

def pixel_size_2_io_box_sizes(psize,cls_psize=3.0,part_d=160):
    bin          = cls_psize/psize
    out_box_size = 1.5*part_d/cls_psize
    return np.round(bin*out_box_size),out_box_size

print pixel_size_2_io_box_sizes(0.429109,3.0,160)
print pixel_size_2_io_box_sizes(1.3,3.0,160)
##
#
# def int64_feature(values):
#   if not isinstance(values, (tuple, list)):
#     values = [values]
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
#
#
# def bytes_feature(values):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
#
# def _get_dataset_filename(dataset_dir,split_name,shard_id,num_shards):
#     output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name,shard_id,num_shards)
#     return os.path.join(dataset_dir, output_filename)
#
# def image_to_tfexample(image_data, image_format, height, width, class_id):
#   return tf.train.Example(features=tf.train.Features(feature={
#       'image/encoded': bytes_feature(image_data),
#       'image/format': bytes_feature(image_format),
#       'image/class/label': int64_feature(class_id),
#       'image/height': int64_feature(height),
#       'image/width': int64_feature(width),
#   }))
#
#
# print pixel_size_2_io_box_sizes(psize=0.318,cls_psize=3.0,part_rad=160)
#
#
# ##
#
# im = np.int8(mrc.load('/jasper/result/PKM2_WT/Extract/bad_6840_101/Movies/PKM2_WT_20151210_0953_movie_avg.mrcs'))
#
# imshow(im[0])
# ##
#
# test_prop    = 0.1
# data_out_dir = '/jasper/data/train_data_sets/junk_particles/train'
#
# shard_id     = 0
# class_ids    = np.zeros(im.shape[0],dtype='int8')
# h,w          = im.shape[1:3]
# fidx         = 0
# tf.gfile.MakeDirs(data_out_dir)
# file_train  = _get_dataset_filename(data_out_dir, 'train', shard_id, _NUM_SHARDS)
# file_test   = _get_dataset_filename(data_out_dir, 'test', shard_id, _NUM_SHARDS)
#
# #with tf.Graph().as_default():
# g = tf.Graph()
# #    with tf.Session('') as sess:
# sess = tf.InteractiveSession()
#         # with tf.python_io.TFRecordWriter(file_train) as tfrecord_writer:
# tfrecord_writer = tf.python_io.TFRecordWriter(file_train)
#
# image_data = tf.Variable(im[fidx], name="image_data")
# ##
# example    = image_to_tfexample(image_data, 'png', h, w, class_ids[fidx])
# ##
# tfrecord_writer.write(example.SerializeToString())
#
# ##
#
