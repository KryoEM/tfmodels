"""
Contains definition of a variant of inception-v4 on 39x19 pedestrian images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from    model import Model,prelu,tiny_darknet_module,darknet_module
from    autopick.tfrecorder import create_dataset
from    autopick.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from    autopick import cfg
import  os
from   myplotlib import imshow,clf

WEIGHT_DECAY = 1e-6
ALPHA_DECAY  = 1e-6
DD           = 4
# equal to
FEAT_STRIDE  = [32,]

def create_instance(split_name,data_dir):
    return AutopickModel(split_name,data_dir)

def anchor_target_layer(rpnshape,coords,part_rad,imshape,feat_stride,name):
    # def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
    with tf.variable_scope(name) as scope:
        rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,[rpnshape,coords['clean'],part_rad,imshape,feat_stride],
                       [tf.float32,tf.float32,tf.float32,tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

class AutopickModel(Model):
    def __init__(self,split_name,data_dir):
        super(AutopickModel, self).__init__(split_name,data_dir)

    def create_dataset(self,split_name,data_dir):
        return create_dataset(split_name,data_dir)

    def arg_scope(self,is_training,**kwargs):
        with slim.arg_scope(super(AutopickModel, self).arg_scope(is_training,WEIGHT_DECAY,use_batch_norm=False,
                                                                batch_norm_decay=0.999,**kwargs)):
            with slim.arg_scope([prelu],alpha_decay=ALPHA_DECAY) as sc:
                return sc

    def network(self,data,is_training=False,**kwargs):

        images  = data['image']
        coords  = {key:value for key,value in data.items() if key is not 'image'}
        imshape = tf.shape(images)

        nclasses = self._dataset.example_meta['nclasses']
        with slim.arg_scope(self.arg_scope(is_training,**kwargs)):
            with tf.variable_scope('autopick', values=data) as sc:
                end_point_collection = sc.original_name_scope + '_end_points'
                with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d,
                                     tiny_darknet_module,darknet_module],
                                    outputs_collections=[end_point_collection]):

                    with slim.arg_scope([slim.max_pool2d],stride=[2,2]):
                        depth = 32
                        net = slim.conv2d(images, depth, [3, 3], scope='conv1_0')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_0')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_1')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_0')
                        net = slim.max_pool2d(net, [2,2], scope='pool2')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_2')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_0')
                        net = slim.max_pool2d(net, [2,2], scope='pool3')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_2')

                        depth *= 2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_0')
                        net = slim.max_pool2d(net, [2,2], scope='pool4')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_2')

                        depth *=2
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_0')
                        net = slim.max_pool2d(net, [2,2], scope='pool5')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_1')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_2')
                        net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_3')

                        #=================== RPN =================
                        ## detection score
                        rpn_conv  = tiny_darknet_module(net,depth // DD, depth, scope='rpn_conv')
                        rpn_score = slim.conv2d(rpn_conv, 3*2, [1,1], padding='VALID',activation_fn=None, scope='rpn_cls_score')
                        rpnshape  = tf.shape(rpn_score)

                        ## data for rpn loss
                        rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                            anchor_target_layer(rpn_score,coords,cfg.PART_R,images,FEAT_STRIDE, 'anchor_target')

                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        glob_init = tf.global_variables_initializer()
                        loc_init = tf.local_variables_initializer()
                        with tf.Session().as_default() as sess:
                           with tf.Graph().as_default() as g:
                               with g.device('/cpu:0'):
                                    sess.run(loc_init)
                                    sess.run(glob_init)
                                    coord = tf.train.Coordinator()
                                    tf.train.start_queue_runners(coord=coord)
                                    rpn_score_py, coords_py, rpnshape_py, imshape_py = sess.run([rpn_score, coords, rpnshape, imshape])

                                    # res = sess.run([rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights])

                                    res = anchor_target_layer_py(rpnshape_py, coords_py['clean'], cfg.PART_R, imshape_py, FEAT_STRIDE)



                                    # ld = [v for v in coords.values()]
                                    # ld.append(rpn_score)

                                    #data_py =  sess.run(data)
                                    # for k in range(10):
                                    #     ld_py = sess.run(ld)
                                    #     print([len(v) for v in ld_py])

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

        image = provider.get(['image'])[0]

        # image,shape = provider.get(['image','shape'])
        #
        # # reshape image
        # shape   = tf.reshape(shape,(2,))
        # image   = tf.reshape(image,shape)
        # specify single channel
        image   = tf.expand_dims(image,axis=2)
        image   = self._image_preprocessing(image)
        # specify size of 1
        image   = tf.expand_dims(image,0)

        classes = self._dataset.example_meta['classes']
        coords  = provider.get(classes)

        # reshape coordinate matrices
        # coords  = [tf.reshape(c,[-1,2]) for c in coords]

        # construct batch dictionary
        data = {'image': image}
        data.update(dict(zip(classes,coords)))
        return data

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
        image = tf.image.random_flip_left_right(image)
        distorted_image = tf.image.random_flip_up_down(image)
        tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
        # return image
        return distorted_image

    def _image_preprocessing_test(self,image):
        return self._preprocess_common(image)


############################ JUNK ##############################






