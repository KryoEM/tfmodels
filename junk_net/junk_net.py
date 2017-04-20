from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  tensorflow as tf
from    tensorflow.contrib.framework import add_arg_scope
from    tensorflow.contrib.layers.python.layers import utils
from    tensorflow.contrib.layers.python.layers import initializers

slim = tf.contrib.slim

@add_arg_scope
def batch_activate(inputs,
                   normalizer_fn,
                   activation_fn,
                   scope=None,
                   reuse=None,
                   outputs_collections=None):
    """ Applies batch_normalization and activation only """
    with tf.variable_scope(scope, 'batch_activate', [inputs], reuse=reuse) as sc:
        if normalizer_fn is not None:
          inputs = normalizer_fn(inputs)

        if activation_fn is not None:
            outputs = activation_fn(inputs)

        return utils.collect_named_outputs(outputs_collections,sc.original_name_scope,outputs)

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope,
                                           outputs)

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
            e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
            e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat(axis=3, values=[e1x1, e3x3])

@add_arg_scope
def fire_module_resnet(inputs,
                       squeeze_depth,
                       expand_depth,
                       scale=0.1,
                       reuse=None,
                       scope=None,
                       outputs_collections=None):
    with tf.variable_scope(scope,'fire_resnet',[inputs],reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,batch_activate],
                            outputs_collections=None):
            tower   = batch_activate(inputs)
            tower   = squeeze(tower,squeeze_depth)
            tower   = expand(tower,expand_depth)
            outputs = inputs + scale*tower
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope,
                                           outputs)

@add_arg_scope
def fire_reduction(inputs,
                   squeeze_depth,
                   expand_depth,
                   kernel_size,
                   stride,
                   padding,reuse=None,scope=None,outputs_collections=None):
    with tf.variable_scope(scope,'fire_reduction',[inputs],reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                            outputs_collections=None):
            squeeze_net = squeeze(inputs,squeeze_depth)
            # maxpool branch
            with tf.variable_scope('Branch_0'):
                tower_maxpool = slim.max_pool2d(inputs,kernel_size,stride=stride,scope='mpool',padding=padding)
            # expand branch
            with tf.variable_scope('Branch_1'):
                tower_fire0  = slim.conv2d(squeeze_net,expand_depth,kernel_size=kernel_size,stride=stride,
                                           padding=padding,scope='expand0')
            # expand branch with larger kernel
            with tf.variable_scope('Branch_1'):
                tower_fire1  = slim.conv2d(squeeze_net,squeeze_depth,3,padding='SAME',scope='expand1_0')
                tower_fire1  = slim.conv2d(tower_fire1,expand_depth,kernel_size,stride=stride,
                                           padding=padding,scope='expand1_1')
        outputs = tf.concat(axis=3, values=[tower_maxpool,tower_fire0,tower_fire1])
        return utils.collect_named_outputs(outputs_collections,sc.original_name_scope,outputs)

def inference(images,num_classes,is_training=True):

    with slim.arg_scope(squeezenet_arg_scope(is_training)):
        with tf.variable_scope('squeezenet', values=[images]) as sc:
            end_point_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, batch_activate, fire_reduction,
                                 slim.max_pool2d, slim.avg_pool2d],
                                outputs_collections=[end_point_collection]):

                net = slim.conv2d(images, 64, [3, 3], scope='conv1')
                #net = slim.avg_pool2d(net, [2,2], stride=[2,2],scope='avgpool1')
                net = slim.max_pool2d(net, [2, 2], scope='maxpool1')

                net = fire_module(net, 16, 64, scope='fire2')
                net = fire_module_resnet(net,16,64,scale=0.1,scope='fire3_0')
                net = batch_activate(net,scope = 'activate3')

                # net = slim.avg_pool2d(net, [2,2], stride=[2,2],scope='avgpool4')
                # net = slim.max_pool2d(net, [2, 2], stride=[2,2], scope='maxpool4')
                net = fire_reduction(net,32,128,[2,2],stride=[2,2],padding='VALID',scope='reduction4')

                net = fire_module(net, 32, 128, scope='fire5')
                net = fire_module_resnet(net,32,128,scale=0.1,scope='fire6_0')
                net = batch_activate(net, scope = 'activate6')

                # net = slim.avg_pool2d(net, [2,2], stride=[2,2],scope='avgpool8')
                # net = slim.max_pool2d(net, [2, 2], scope='maxpool8')
                net = fire_reduction(net,64,256,[2,2],stride=[2,2],padding='VALID',scope='reduction8')

                net = fire_module(net, 64, 256, scope='fire9')
                net = fire_module_resnet(net,64,256,scale=0.1,scope='fire10_0')
                net = batch_activate(net, scope = 'activate10')

                # net = slim.avg_pool2d(net, [2,2], stride=[2,2],scope='avgpool10')
                # net = slim.max_pool2d(net, [2, 2], scope='maxpool10')
                net = fire_reduction(net,128,512,[2,2],stride=[2,2],padding='VALID',scope='reduction10')

                net = fire_module(net, 128, 512, scope='fire11')
                net = fire_module_resnet(net,128,512,scale=0.1,scope='fire12_0')
                net = batch_activate(net, scope = 'activate12')

                # net = slim.avg_pool2d(net, [2,2], stride=[2,2],scope='avgpool12')
                #net = slim.max_pool2d(net, [2, 2], scope='maxpool12')
                net = fire_reduction(net,256,512,[2,2],stride=[2,2],padding='VALID',scope='reduction12')

                net = fire_module(net, 256, 1024, scope='fire13')
                net = fire_module_resnet(net,256,1024,scale=0.1,scope='fire14_0')
                net = batch_activate(net, scope = 'activate14')

                # net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool14')
                #net = slim.max_pool2d(net, [2, 2], scope='maxpool14')
                net = fire_reduction(net,512,2048,[2,2],stride=[2,2],padding='VALID',scope='reduction14')

                net = slim.conv2d(net, num_classes, [1, 1],activation_fn=None,
                                  normalizer_fn=None,scope='conv14')

                logits = tf.squeeze(net, [1, 2], name='logits')
                logits = utils.collect_named_outputs(end_point_collection,sc.name + '/logits',logits)

            end_points = utils.convert_collection_to_dict(end_point_collection)
    return logits, end_points

def squeezenet_arg_scope(is_training,
                   weight_decay=0.00001,
                   use_batch_norm=False,
                   batch_norm_decay=0.999):

    normalizer_fn = slim.batch_norm if use_batch_norm else None
    with slim.arg_scope([slim.conv2d, slim.fully_connected, batch_activate],
                         activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.fully_connected],
                              weights_regularizer = slim.l2_regularizer(weight_decay),
                              weights_initializer = initializers.xavier_initializer()):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer = slim.l2_regularizer(weight_decay),
                                weights_initializer = initializers.xavier_initializer_conv2d()):
                with slim.arg_scope([slim.batch_norm],
                                    is_training = is_training,
                                    decay = batch_norm_decay):
                    with slim.arg_scope([slim.conv2d, batch_activate],  # slim.fully_connected
                                        normalizer_fn = normalizer_fn) as sc:
                        return sc

###################################################################################
# def squeezenet_arg_scope_old(is_training, decay=0.999):
#     with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
#         with slim.arg_scope([batch_activate],
#                             normalizer_fn=slim.batch_norm,
#                             activation_fn=tf.nn.relu):
#             with slim.arg_scope([slim.batch_norm],
#                              is_training=is_training,
#                              decay=decay) as sc:
#                 return sc



########################### GARBAGE ######################################

# def inference(images,num_classes,is_training=True):
#
#     with slim.arg_scope(squeezenet_arg_scope(is_training)):
#         with tf.variable_scope('squeezenet', values=[images]) as sc:
#             end_point_collection = sc.original_name_scope + '_end_points'
#             with slim.arg_scope([slim.conv2d, batch_activate, fire_reduction,
#                                  slim.max_pool2d, slim.avg_pool2d],
#                                 outputs_collections=[end_point_collection]):
#
#                 net = slim.conv2d(images, 96, [3, 3], scope='conv1')
#                 #net = slim.conv2d(net, 96, [3, 3], scope='conv2')
#                 #net = slim.conv2d(net, 96, [3, 3], scope='conv3')
#
#                 net = slim.max_pool2d(net, [2, 2], scope='maxpool1')
#                 net = fire_module(net, 16, 64, scope='fire2')
#
#                 net = fire_module_resnet(net,16,64,scale=0.1,scope='fire3_0')
#                 net = fire_module_resnet(net,16,64,scale=0.1,scope='fire3_1')
#                 net = batch_activate(net,scope = 'activate3')
#
#                 net = fire_module(net, 32, 128, scope='fire4')
#                 net = batch_activate(net,scope = 'activate4')
#
#                 net = fire_reduction(net,48,192,[2,2],stride=[2,2],padding='VALID',scope='maxpool4')
#                 #net = slim.max_pool2d(net, [4, 2], stride=[4,2], scope='maxpool4')
#
#                 net = fire_module(net, 48, 192, scope='fire5')
#
#                 net = fire_module_resnet(net,48,192,scale=0.1,scope='fire6_0')
#                 net = fire_module_resnet(net,48,192,scale=0.1,scope='fire6_1')
#                 net = batch_activate(net, scope = 'activate6')
#
#                 net = fire_module(net, 64, 256, scope='fire8')
#                 net = batch_activate(net, scope = 'activate8')
#
#                 net = fire_reduction(net,64,256,[2,2],stride=[2,2],padding='VALID',scope='maxpool8')
#
#                 net = fire_module(net, 64, 256, scope='fire9')
#
#                 net = fire_module_resnet(net,64,256,scale=0.1,scope='fire10_0')
#                 net = fire_module_resnet(net,64,256,scale=0.1,scope='fire10_1')
#                 net = batch_activate(net, scope = 'activate10')
#
#                 net = fire_module(net, 80, 360, scope='fire11')
#                 net = batch_activate(net, scope = 'activate11')
#
#                 net = fire_reduction(net,80,360,[2,2],stride=[2,2],padding='VALID',scope='maxpool11')
#
#                 net = fire_module(net, 96, 384, scope='fire12')
#                 net = batch_activate(net, scope = 'activate12')
#
#                 # Reversed avg and conv layers per 'Network in Network'
#                 net = slim.avg_pool2d(net,[5,5], scope='avgpool10') #net.get_shape()[1:3]
#                 net = slim.conv2d(net, num_classes, [1, 1],
#                                   activation_fn=None,
#                                   normalizer_fn=None,
#                                   scope='conv10')
#                 logits = tf.squeeze(net, [1, 2], name='logits')
#                 logits = utils.collect_named_outputs(end_point_collection,
#                                                      sc.name + '/logits',
#                                                      logits)
#             end_points = utils.convert_collection_to_dict(end_point_collection)
#     return logits, end_points
