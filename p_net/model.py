"""
Contains definition of a variant of inception-v4 on 39x19 pedestrian images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from    model import Model,fire_module,fire_reduction,split_module,prelu,tiny_darknet_module,darknet_module
from    squeezenet.tfrecorder import create_dataset

WEIGHT_DECAY = 1e-8
ALPHA_DECAY  = 1e-8
DD           = 4

def create_instance(split_name,data_dir):
    return PModel(split_name,data_dir)

class PModel(Model):
    def __init__(self,split_name,data_dir):
        super(PModel, self).__init__(split_name,data_dir)

    def create_dataset(self,split_name,data_dir):
        return create_dataset(split_name,data_dir)

    def arg_scope(self,is_training,**kwargs):
        with slim.arg_scope(super(PModel, self).arg_scope(is_training,WEIGHT_DECAY,use_batch_norm=True,
                                                                   batch_norm_decay=0.999,**kwargs)):
            with slim.arg_scope([prelu],alpha_decay=ALPHA_DECAY) as sc:
                return sc

    def network(self,images,is_training=False,**kwargs):
        nclasses = self._dataset.example_meta['nclasses']
        with slim.arg_scope(self.arg_scope(is_training,**kwargs)):
            with tf.variable_scope('darknet', values=[images]) as sc:
                end_point_collection = sc.original_name_scope + '_end_points'
                with slim.arg_scope([fire_module,slim.conv2d,slim.fully_connected,fire_reduction,slim.max_pool2d,
                                     tiny_darknet_module,darknet_module],
                                    outputs_collections=[end_point_collection]):

                    with slim.arg_scope([fire_module,split_module],kernel_size=[3,3]):
                        depth = 16
                        net = slim.conv2d(images, depth, [3, 3], scope='conv1_0')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_0')
                        net = slim.max_pool2d(net, [2, 2], scope='maxpool2')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_1')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_0')
                        net = slim.max_pool2d(net, [2,2], stride=[2,2], scope='maxpool3')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_2')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_0')
                        net = slim.max_pool2d(net, [2,2], stride=[2,2], scope='maxpool4')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_2')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_0')
                        net = slim.max_pool2d(net, [2,2], stride=[2,2], scope='maxpool5')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_2')

                        depth *=2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_0')
                        net = slim.max_pool2d(net, [2, 2], scope='maxpool6')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_1')

                        net = slim.avg_pool2d(net, net.get_shape()[1:3], stride=net.get_shape()[1:3], scope='avgpool4')
                        net = slim.flatten(net)
                        logits = slim.fully_connected(net, nclasses,
                                                      activation_fn=None,
                                                      normalizer_fn=None,
                                                      scope='fc1')

                        return logits, end_point_collection

    def data_batch(self,batch_size,reader_threads):
        ''''Return tensor with data batch for training '''
        provider = slim.dataset_data_provider.DatasetDataProvider(self._dataset,shuffle=True,num_readers=reader_threads,
                                                                  common_queue_capacity=10*batch_size,
                                                                  common_queue_min=4*batch_size)
        [image, label] = provider.get(['image', 'label'])
        image = self._image_preprocessing(image)
        images,labels = tf.train.batch([image,label],batch_size=batch_size,
                                        num_threads=reader_threads,capacity=10*batch_size)
        return images,labels

    def add_loss_and_metrics_train(self,labels,logits,end_point_collection):

        #### ADD LOSS ##########
        tf.losses.sparse_softmax_cross_entropy(labels,logits)

        ##### ADD METRICS ######
        predictions = tf.argmax(logits, 1)
        tf.contrib.metrics.streaming_accuracy(predictions,labels,
                                               name='accuracy',
                                               metrics_collections=['metrics'],
                                               updates_collections=tf.GraphKeys.UPDATE_OPS)


    def get_metrics_eval(self,labels,logits):
        predictions = tf.argmax(logits, 1)
        cid = tf.constant(self._dataset.example_meta['classes']['/2_good/'], dtype=predictions.dtype)
        return slim.metrics.aggregate_metric_map({
            'accuracy': tf.contrib.metrics.streaming_accuracy(predictions,labels,
                                                               name='accuracy',
                                                               metrics_collections=['metrics'],
                                                               updates_collections=tf.GraphKeys.UPDATE_OPS),
            'precision': tf.contrib.metrics.streaming_precision(tf.equal(predictions,cid),tf.equal(labels,cid),
                                                               name='precision',
                                                               metrics_collections=['metrics'],
                                                               updates_collections=tf.GraphKeys.UPDATE_OPS)
        })

    def _preprocess_common(self,image):
        image = tf.cast(image, tf.float32)
        image -= tf.reduce_mean(image)
        image = tf.image.per_image_standardization(image)
        tf.summary.image('initial_image', tf.expand_dims(image, 0))
        return image

    def _image_preprocessing_train(self,image):
        image = self._preprocess_common(image)
        distorted_image = tf.image.random_flip_left_right(image)
        tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
        # return image
        return distorted_image

    def _image_preprocessing_test(self,image):
        return self._preprocess_common(image)


############################ JUNK ##############################






