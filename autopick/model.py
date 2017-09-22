"""
Contains definition of a variant of inception-v4 on 39x19 pedestrian images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from    model_generic import Model,prelu,enet_pool,enet,enet_unpool,modified_smooth_l1
from    autopick.tfrecorder import create_dataset
from    autopick import cfg
import  numpy as np
import  os

from   myplotlib import imshow,clf

WEIGHT_DECAY = 1e-5
ALPHA_DECAY  = 1e-5
DD           = 8

def create_instance(split_name,data_dir):
    return AutopickModel(split_name,data_dir)

class AutopickModel(Model):
    def __init__(self,split_name,data_dir):
        super(AutopickModel, self).__init__(split_name,data_dir)

    def create_dataset(self,split_name,data_dir):
        return create_dataset(split_name,data_dir)

    @staticmethod
    def get_rpn_scores(net, scope, nclasses, reuse=False):
        depth = net.shape[-1]._value
        # Feature extractor part is shared for all heads
        with tf.variable_scope(scope, reuse=reuse):
            # head processing
            rpn_conv = enet(net, depth // DD, depth, scope='rpn_conv_0', reuse=reuse)
            rpn_conv = enet(rpn_conv, depth // DD, depth, scope='rpn_conv_1', reuse=reuse)

            # score for classification
            rpn_score = slim.conv2d(rpn_conv, nclasses, [1, 1], activation_fn=None, scope='rpn_cls_score',reuse=reuse)
            # coordinate correction prediction
            dxy_pred  = slim.conv2d(rpn_conv, 2, [1, 1], activation_fn=None, scope='dxy_pred',reuse=reuse)
            # auxilliary channels for classification
            cls_aux   = slim.conv2d(rpn_conv, cfg.N_CLS_AUX_CHANNELS, [1, 1], scope='cls_aux',reuse=reuse)

            # focused detection score #!!
            cls_conv  =  tf.concat(axis=3, values=[rpn_score,dxy_pred,cls_aux])

            # extra processing for focused detection
            depth     = cfg.CLS_CHANNELS
            cls_conv  = slim.conv2d(cls_conv, depth, [1, 1], scope='cls_conv_0',reuse=reuse)
            cls_conv  = enet(cls_conv, depth // DD, depth, scope='cls_enet_1', reuse=reuse)
            cls_conv  = enet(cls_conv, depth // DD, depth, scope='cls_enet_2', reuse=reuse)
            cls_conv  = enet(cls_conv, depth // DD, depth, scope='cls_enet_3', reuse=reuse)
            cls_score = slim.conv2d(cls_conv, 2, [1, 1], activation_fn=None, scope='cls_score',reuse=reuse)
            return rpn_score,dxy_pred,cls_score

    @staticmethod
    def get_batch_image_indices(data, rpn_shape, label):
        batch = rpn_shape[0]
        idxs  = data[label] - 1
        # obtain location of batch padding
        non_ignore = tf.where(tf.greater(idxs, 0))
        # convert to indice into batched image
        # generate batch indices
        bidxs = idxs + tf.reshape(tf.range(batch) * rpn_shape[1] * rpn_shape[2], tf.cast([batch, 1], tf.int32))
        # select non-padded indices
        return tf.gather_nd(bidxs, non_ignore),non_ignore

    @staticmethod
    def labels_for_display(rpn_prob, imidxs):
        pshape  = tf.shape(rpn_prob)
        rpn_prob_reshape = tf.reshape(rpn_prob, [-1,pshape[-1]])
        rpshape = tf.cast(tf.shape(rpn_prob_reshape)[:-1], tf.int64)
        # bring background to 0
        labels = tf.reshape(tf.sparse_to_dense(imidxs, rpshape, tf.ones(tf.shape(imidxs), dtype=tf.int32),
                                               validate_indices=False),pshape[:-1])
        return labels

    def scores2losses(self,logits, data, reuse=False, scope=''):
        rpn_score = logits['rpn_score']
        dxy_pred  = logits['dxy_pred']
        cls_score = logits['cls_score']

        nclasses = self._dataset.example_meta['nclasses']
        classes  = self._dataset.example_meta['classes']
        with tf.variable_scope(scope, reuse=reuse):
            # ========= RoI Proposal ============
            im_shape = tf.cast(tf.shape(rpn_score)[:-1], tf.int64)

            # ##### LOSS functions #######
            rpn_prob = tf.nn.softmax(rpn_score, name='rpn_prob')
            cls_prob = tf.nn.softmax(cls_score, name='cls_prob')

            rpn_score_reshape = tf.reshape(rpn_score, [-1, nclasses])
            dxy_pred_reshape  = tf.reshape(dxy_pred, [-1, 2])
            cls_score_reshape = tf.reshape(cls_score, [-1, 2])
            # rpn_prob_reshape  = tf.reshape(rpn_prob, [-1, nclasses])

            # ##### SUMMARY ###########
            tf.summary.image('zinput_image', tf.expand_dims(data['image'][-1, :, :, :1], axis=0))

            # collect foreground scores
            rpn_loss = 0.0
            for idx in range(nclasses):
                c        = classes[idx]
                imidxs,_ = self.get_batch_image_indices(data, im_shape, c)
                score_gather = tf.reshape(tf.gather(rpn_score_reshape, imidxs), [-1, nclasses])
                npart  = tf.shape(score_gather)[0]
                labels = idx*tf.ones((npart,), tf.int32)
                ce     = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score_gather, labels=labels)
                # collect probablity of the current class
                # pt     = tf.reshape(tf.gather(rpn_prob_reshape, imidxs), [-1, nclasses])[...,idx]
                # pt     = tf.gather(rpn_prob_reshape[...,idx], imidxs)
                # calculate focused cross entropy
                # fce    = (((1.-pt)/(1.-1./nclasses))**cfg.GAMMA)*ce
                # normalize cross entropy by the number of particles
                rpn_loss = rpn_loss + tf.reduce_sum(ce)/tf.cast(npart + np.int32(1), tf.float32)

                # ##### SUMMARY ###########
                # create labels -needed for summary only
                dsp_labels = tf.cast(AutopickModel.labels_for_display(rpn_prob, imidxs), tf.float32)
                tf.summary.image(c + '_label', tf.expand_dims(tf.expand_dims(dsp_labels[-1], axis=0), axis=3),max_outputs=1000)
                if c == 'clean':
                    clean_prob = rpn_prob[-1,:,:,idx]
                tf.summary.image(c + '_prob', tf.expand_dims(tf.expand_dims(rpn_prob[-1,:,:,idx], axis=0), axis=3),max_outputs=1000)

            ##############################################################################
            # add loss for coordinate regression and classification
            neibidxs,nnig_neib = self.get_batch_image_indices(data, im_shape, 'dxyidxs')
            centidxs,nnig_cent = self.get_batch_image_indices(data, im_shape, 'clean')
            nneib       = tf.shape(neibidxs)[0]
            ncent       = tf.shape(centidxs)[0]
            nneib1      = tf.cast(nneib + np.int32(1), tf.float32)
            ncent1      = tf.cast(ncent + np.int32(1), tf.float32)
            neib_dxy_gather  = tf.reshape(tf.gather(dxy_pred_reshape, neibidxs), [-1, 2])
            cent_dxy_gather  = tf.reshape(tf.gather(dxy_pred_reshape, centidxs), [-1, 2])
            neib_cls_gather  = tf.reshape(tf.gather(cls_score_reshape, neibidxs), [-1, 2])
            cent_cls_gather  = tf.reshape(tf.gather(cls_score_reshape, centidxs), [-1, 2])

            # assign classification labels
            neiblabels = tf.zeros((nneib,), tf.int32)
            centlabels = tf.ones((ncent,), tf.int32)
            zero_dxy   = tf.zeros((ncent,2), tf.float32)

            # l1 loss for coordinate corrections
            dxy = tf.reshape(tf.gather_nd(data['dxy'],nnig_neib),(-1,2))
            neib_smooth_l1 = modified_smooth_l1(cfg.L1_SIGMA, neib_dxy_gather, dxy)
            cent_smooth_l1 = modified_smooth_l1(cfg.L1_SIGMA, cent_dxy_gather, zero_dxy)

            # cross entropy loss for classification
            neibce         = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neib_cls_gather, labels=neiblabels)
            centce         = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cent_cls_gather, labels=centlabels)

            dxy_loss       = (tf.reduce_sum(neib_smooth_l1)/nneib1 + tf.reduce_sum(cent_smooth_l1)/ncent1)/2.0
            cls_loss       = tf.reduce_sum(neibce)/nneib1 + tf.reduce_sum(centce)/ncent1

            # add classification result to display
            # neib_map = tf.cast(AutopickModel.labels_for_display(cls_prob, neibidxs), tf.float32)
            # tf.summary.image('zneib_map', tf.expand_dims(tf.expand_dims(neib_map[-1], axis=0), axis=3),max_outputs=1000)
            tf.summary.image('yclean_cls', tf.expand_dims(tf.expand_dims(cls_prob[-1, :, :, -1], axis=0), axis=3),max_outputs=1000)
            tf.summary.image('ycombined_cls', tf.expand_dims(tf.expand_dims(clean_prob*cls_prob[-1, :, :, -1], axis=0), axis=3),max_outputs=1000)

            # os.environ['CUDA_VISIBLE_DEVICES'] = ''
            # glob_init = tf.global_variables_initializer()
            # loc_init = tf.local_variables_initializer()
            # with tf.Session().as_default() as sess:
            #    with tf.Graph().as_default() as g:
            #        with g.device('/cpu:0'):
            #             sess.run(loc_init)
            #             sess.run(glob_init)
            #             coord = tf.train.Coordinator()
            #             tf.train.start_queue_runners(coord=coord)
            #             # d = sess.run([data,neibidxs])
            #             # neib_dxy_py, dxy_py = sess.run([neib_dxy_gather, dxy])
            #             rpn_loss_py, dxy_loss_py, cls_loss_py = sess.run([rpn_loss, dxy_loss, cls_loss])
            #             # image_py,imidxs_py,dsp_labels_py,prob_py = sess.run([data['image'],imidxs,dsp_labels,rpn_prob])

            # bw = np.zeros((16*64*64),dtype=np.float32)
            # bw[imidxs_py] = 1
            # bw = np.reshape(bw, (16,64,64,1))
        return rpn_loss,dxy_loss,cls_loss

    def arg_scope(self,is_training,**kwargs):
        with slim.arg_scope(super(AutopickModel, self).arg_scope(is_training,WEIGHT_DECAY,use_batch_norm=True,
                                                                batch_norm_decay=0.997,**kwargs)):
            with slim.arg_scope([prelu],alpha_decay=ALPHA_DECAY) as sc:
                return sc

    def network(self,data,is_training=False,**kwargs):

        images   = data['image']
        nclasses = self._dataset.example_meta['nclasses']
        with slim.arg_scope(self.arg_scope(is_training,**kwargs)):
            with tf.variable_scope('autopick', values=data) as sc:
                end_point_collection = sc.original_name_scope + '_end_points'
                with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d,
                                     enet, enet_pool, enet_unpool],
                                    outputs_collections=[end_point_collection]):

                    #### Encoder convolution layers #############
                    depth  = cfg.CONV0_CHANNELS
                    enc1_0 = slim.conv2d(images, depth, kernel_size=[3, 3], scope='conv1_0')
                    enc1_1 = enet_pool(enc1_0, depth // DD, depth, scope='enet1_1')
                    enc1   = enet(enc1_1, depth // DD, depth, scope='enet1_2')
                    # enc1   = tf.concat(axis=3, values=[enc1_1,enc1_2])

                    depth *= 2
                    enc2_0 = enet_pool(enc1, depth // DD, depth, scope='enet2_0')
                    enc2   = enet(enc2_0, depth // DD, depth, scope='enet2_1')
                    # enc2   = tf.concat(axis=3, values=[enc2_0,enc2_1])

                    depth *= 2
                    enc3_0 = enet_pool(enc2, depth // DD, depth, scope='enet3_0')
                    enc3   = enet(enc3_0, depth // DD, depth, scope='enet3_1')
                    # enc3   = tf.concat(axis=3, values=[enc3_0,enc3_1])

                    depth *= 2
                    enc4_0 = enet_pool(enc3, depth // DD, depth, scope='enet4_0')
                    enc4   = enet(enc4_0, depth // DD, depth, scope='enet4_1')
                    # enc4   = tf.concat(axis=3, values=[enc4_0,enc4_1])

                    depth *= 2
                    enc5_0 = enet_pool(enc4, depth // DD, depth, scope='enet5_0')
                    enc5   = enet(enc5_0, depth // DD, depth, scope='enet5_1')
                    # enc5   = tf.concat(axis=3, values=[enc5_0,enc5_1])

                    depth *= 2
                    enc6_0 = enet_pool(enc5, depth // DD, depth, scope='enet6_0')
                    enc6_1 = enet(enc6_0, depth // DD, depth, scope='enet6_1')
                    enc6_1 = tf.concat(axis=3, values=[enc6_0,enc6_1])
                    enc6_2 = enet(enc6_1, depth // DD, depth, scope='enet6_2')
                    enc6   = tf.concat(axis=3, values=[enc6_1,enc6_2])

                    depth *= 2
                    enc7_0 = enet_pool(enc6, depth // DD, depth, scope='enet7_0')
                    enc7_1 = enet(enc7_0, depth // DD, depth, scope='enet7_1')
                    enc7_1 = tf.concat(axis=3, values=[enc7_0,enc7_1])
                    enc7_2 = enet(enc7_1, depth // DD, depth, scope='enet7_2')
                    enc7   = tf.concat(axis=3, values=[enc7_1,enc7_2])

                    #### Unpooling layers ##############
                    depth //= 2
                    dec6_0 = enet_unpool(enc7, enc6, depth // DD, depth, scope='uenet6_0')
                    dec6   = enet(dec6_0, depth // DD, depth, scope='uenet6_1')
                    # dec6   = tf.concat(axis=3, values=[dec6_0,dec6_1])

                    depth //= 2
                    dec5_0 = enet_unpool(dec6, enc5, depth // DD, depth, scope='uenet5_0')
                    dec5   = enet(dec5_0, depth // DD, depth, scope='uenet5_1')
                    # dec5   = tf.concat(axis=3, values=[dec5_0,dec5_1])

                    depth //= 2
                    dec4_0 = enet_unpool(dec5, enc4, depth // DD, depth, scope='uenet4_0')
                    dec4   = enet(dec4_0, depth // DD, depth, scope='uenet4_1')
                    # dec4   = tf.concat(axis=3, values=[dec4_0,dec4_1])

                    depth //= 2
                    dec3_0 = enet_unpool(dec4, enc3, depth // DD, depth, scope='uenet3_0')
                    dec3   = enet(dec3_0, depth // DD, depth, scope='uenet3_1')
                    # dec3   = tf.concat(axis=3, values=[dec3_0,dec3_1])

                    # connect detection head that calculates detection score
                    rpn_score,dxy_pred,cls_score = AutopickModel.get_rpn_scores(dec3, 'rpn_score3', nclasses, reuse=False)

                    logits = {'rpn_score':rpn_score,
                              'dxy_pred': dxy_pred,
                              'cls_score': cls_score}

                    return logits, end_point_collection

    def data_batch(self,batch_size,reader_threads):
        ''''Return tensor with data batch for training '''
        provider = slim.dataset_data_provider.DatasetDataProvider(self._dataset,shuffle=True,num_readers=reader_threads,
                                                                  common_queue_capacity=10*batch_size,
                                                                  common_queue_min=4*batch_size)


        shape   = (cfg.PICK_WIN,)*2 # self._dataset.example_meta['shape']
        classes = self._dataset.example_meta['classes']
        keys    = ['image'] + classes + ['dxy','dxyidxs']
        data    = provider.get(keys)
        # reshape image
        image   = tf.reshape(data[0],shape+(1,))
        data[0] = self._image_preprocessing(image)

        databatch = tf.train.batch(data,batch_size=batch_size,
                                   num_threads=reader_threads,
                                   dynamic_pad=True,
                                   capacity=10*batch_size)

        return dict(zip(keys,databatch))

    def data_single(self,image):
        image = self._image_preprocessing_test(image)
        return  tf.expand_dims(image,axis=0)

    def add_loss_and_metrics_train(self,data,logits,end_point_collection):
        rpn_loss, dxy_loss, cls_loss = self.scores2losses(logits,data,scope='rpn_head')
        loss = rpn_loss + dxy_loss + cls_loss
        #### ADD LOSS ##########
        tf.losses.add_loss(loss)

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
        # image -= tf.reduce_mean(image)
        image = tf.image.per_image_standardization(image)
        # tf.summary.image('initial_image', tf.expand_dims(image, 0))
        return image

    def _image_preprocessing_train(self,image):
        image = self._preprocess_common(image)
        # image = tf.image.random_flip_left_right(image)
        # distorted_image = tf.image.random_flip_up_down(image)
        # tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
        # return image
        return image

    def _image_preprocessing_test(self,image):
        return self._preprocess_common(image)


############################ JUNK ##############################

    # @staticmethod
    # def reshape_rpn_score(input, name):
    #     # bxHxWxN
    #     sz = tf.shape(input)
    #     # bxHxW*N
    #     return tf.reshape(input, [sz[0], sz[1], sz[2] * sz[3]])

    # @staticmethod
    # def score2prob(rpn_score):
    #     sz = tf.shape(rpn_score)
    #     # bxHxWxN
    #     # rpn_score_reshape = tf.reshape(rpn_score, [sz[0], sz[1], sz[2] * sz[3]])
    #     # bxHxW*N - select only positive class prob
    #     rpn_prob_reshape = tf.nn.softmax(rpn_score, name='rpn_cls_prob')
    #     # convert back to bxHxWxAx2 and take only positive class probability
    #     return tf.reshape(rpn_prob_reshape, np.int32(rpn_score.shape))

    # def anchor_target_layer(rpnshape,coords,part_rad,imshape,feat_stride,name):
    #     # def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
    #     with tf.variable_scope(name) as scope:
    #         rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
    #             tf.py_func(anchor_target_layer_py,[rpnshape,coords['clean'],part_rad,imshape,feat_stride],
    #                        [tf.float32,tf.float32,tf.float32,tf.float32])
    #
    #         rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
    #         rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
    #         rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
    #         rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')
    #         return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    # depth *= 2
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_0')
    # net = slim.max_pool2d(net, [2, 2], scope='pool1')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny2_1')
    #
    # depth *= 2
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_0')
    # net = slim.max_pool2d(net, [2,2], scope='pool2')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_1')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny3_2')
    #
    # depth *= 2
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_0')
    # net = slim.max_pool2d(net, [2,2], scope='pool3')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_1')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny4_2')
    #
    # depth *= 2
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_0')
    # net = slim.max_pool2d(net, [2,2], scope='pool4')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_1')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny5_2')
    #
    # depth *=2
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_0')
    # net = slim.max_pool2d(net, [2,2], scope='pool5')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_1')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_2')
    # net = tiny_darknet_module(net, depth // DD, depth, scope='tiny6_3')


    # # image,shape = provider.get(['image','shape'])
        # #
        # # # reshape image
        # # shape   = tf.reshape(shape,(2,))
        # # image   = tf.reshape(image,shape)
        # # specify single channel
        # image   = tf.expand_dims(image,axis=2)
        # image   = self._image_preprocessing(image)
        # # specify size of 1
        # image   = tf.expand_dims(image,0)
        #
        # classes = self._dataset.example_meta['classes']
        # coords  = provider.get(classes)
        #
        # # reshape coordinate matrices
        # # coords  = [tf.reshape(c,[-1,2]) for c in coords]
        #
        # # construct batch dictionary
        # data = {'image': image}
        # data.update(dict(zip(classes,coords)))
        # return data


    # #=================== RPN =================
    # ## detection score
    # rpn_conv  = tiny_darknet_module(net,depth // DD, depth, scope='rpn_conv')
    # rpn_score = slim.conv2d(rpn_conv, 3*2, [1,1], padding='VALID',activation_fn=None, scope='rpn_cls_score')
    # rpnshape  = tf.shape(rpn_score)
    #
    # ## data for rpn loss
    # rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
    #     anchor_target_layer(rpn_score,coords,cfg.PART_R,images,FEAT_STRIDE, 'anchor_target')
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # glob_init = tf.global_variables_initializer()
    # loc_init = tf.local_variables_initializer()
    # with tf.Session().as_default() as sess:
    #    with tf.Graph().as_default() as g:
    #        with g.device('/cpu:0'):
    #             sess.run(loc_init)
    #             sess.run(glob_init)
    #             coord = tf.train.Coordinator()
    #             tf.train.start_queue_runners(coord=coord)
    #             rpn_score_py, coords_py, rpnshape_py, imshape_py = sess.run([rpn_score, coords, rpnshape, imshape])
    #
    #             # res = sess.run([rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights])
    #
    #             res = anchor_target_layer_py(rpnshape_py, coords_py['clean'], cfg.PART_R, imshape_py, FEAT_STRIDE)
    #
    #
    #
    #             # ld = [v for v in coords.values()]
    #             # ld.append(rpn_score)
    #
    #             #data_py =  sess.run(data)
    #             # for k in range(10):
    #             #     ld_py = sess.run(ld)
    #             #     print([len(v) for v in ld_py])
    #
    # net = slim.avg_pool2d(net, net.get_shape()[1:3], stride=net.get_shape()[1:3], scope='avgpool4')
    # net = slim.flatten(net)
    # logits = slim.fully_connected(net, nclasses,
    #                               activation_fn=None,
    #                               normalizer_fn=None,
    #                               scope='fc1')





