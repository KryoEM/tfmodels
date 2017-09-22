import os
import tensorflow as tf
from   model_generic import Model
import numpy as np
# from PIL import Image, ImageDraw, ImageFont, ImageOps
from arg_parsing import load_config
from gpu import pick_gpus_lowest_memory
from myplotlib import imshow,clf
from rpn.generate_anchors import generate_anchors
from rpn.bbox.bbox_transform import bbox_transform_inv,sized_to_4coords,revert_xy
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from   scipy.ndimage.measurements import label
from   utils import col_set_diff
from   fileio import filetools as ft
from   relion_params import path2psize,parse_ctf_star,path2part_diameter
from   fileio import mrc
import image
import cv2
import utils
from   skimage.measure import label,regionprops

# import glob

from autopick import cfg


slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model_path', 'horizon.model',
                           'Path to the custom implementation of model')
tf.app.flags.DEFINE_string('tfrecord_dir', '/home/yonatan/data/horizon/tfrecord',
                           'Directory for tfrecord input')
tf.app.flags.DEFINE_string('output_dir', '/home/yonatan/data/horizon/logs',
                           'Path to directory where summaries will be saved.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/yonatan/data/horizon/ckpts',
                           'Path to the directory where summaries and checkpoints will be saved.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size for model training.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            'The duration for the program to sleep before awaiting a new checkpoint to evaluate..')
tf.app.flags.DEFINE_integer('reader_threads', 4, 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_string('eval_device', '/cpu:0', 'Device to use for evaluation.')
tf.app.flags.DEFINE_string('use_gpu', False, 'if to use gpu')

FLAGS = load_config(tf.app.flags.FLAGS, 'eval')

tf.logging.set_verbosity(tf.logging.INFO)

if FLAGS.use_gpu:
    gpuid = pick_gpus_lowest_memory(1, 0)[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuid
    device = '/gpu:%d' % gpuid
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = '/cpu:0'

##
def _load_checkpoint():
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # load model
    ckpt_path = ckpt.model_checkpoint_path
    saver     = tf.train.Saver()
    # saver = tf.train.import_meta_graph(ckpt_path+'.meta',clear_devices=True)
    saver.restore(sess, ckpt_path)

def calc_single_score_per_particle(cls_prob, rpn_prob, dxy_pred):
    '''Combines region proposal, classification and particle centering scores into one map with one score
       per particle.
       cls_prob - probability that this particle is isolated
       rpn_prob - probability that there is a particle
       dxy_pred - predicted location corrections around the particle'''
    bw = np.logical_and(cls_prob >= cfg.PROB_THRESH, rpn_prob >= cfg.PROB_THRESH)
    # predicted distance from partice
    dxy = np.sqrt(np.sum(dxy_pred ** 2, axis=2))

    # particle radius in feature map
    fr = np.int32(cfg.PART_D_PIXELS / (2.0 * cfg.STRIDES[0]))
    # create circular mask
    x, y = image.cart_coords2D((np.int32(2 * fr) + 3,) * 2)
    r = np.sqrt(x ** 2 + y ** 2)
    H = np.logical_and(r >= fr - 2, r <= fr)

    # obtain cmap with score that describes how is the particle isolated from neighbors values in [0,1]
    bwsz = bw.shape
    cmap = np.zeros(bwsz, np.float32)
    hx, hy = np.where(H)
    xs, ys = np.where(bw)
    rh = r.flat[np.ravel_multi_index([hx, hy], H.shape)]
    for x, y in zip(xs, ys):
        xhx = np.minimum(np.maximum(hx + x - fr - 1, 0), bwsz[0] - 1)
        yhy = np.minimum(np.maximum(hy + y - fr - 1, 0), bwsz[1] - 1)
        # calc mean normalized radius
        pmean = np.mean((dxy.flat[np.ravel_multi_index([xhx, yhy], bwsz)] - dxy[x, y]) / rh)
        # count how many pixels exceed half of the expected radius
        pcount = np.sum(dxy.flat[np.ravel_multi_index([xhx, yhy], bwsz)] - dxy[x, y] >= rh / 2.0) / float(rh.size)
        cmap[x, y] = pcount * pmean

    # analyse region properties and leave only the strongest cmap scores per particle
    bw = np.logical_and(bw, cmap >= cfg.PERIM_THRESH)
    cmap[np.logical_not(bw)] = 0
    regions = regionprops(label(bw))
    for region in regions:
        coords = region['Coordinates']
        idxs = np.ravel_multi_index(coords.transpose(), bwsz)
        # obtain switch off indices
        offidxs = idxs[cmap.flat[idxs] < cmap.flat[idxs].max()]
        # switch off non-maxima
        cmap.flat[offidxs] = 0
    return cmap

def image_generator(ctfstar):
    micros = parse_ctf_star(ctfstar)
    psize  = path2psize(ctfstar,'CtfFind')
    # particle diameter in pixels
    D      = path2part_diameter(ctfstar) / psize
    # calcualte micrographs binning factor
    bn     = D / cfg.PART_D_PIXELS
    psizebn = psize*bn
    for micro in micros:
        im      = mrc.load(micro)[0]
        szbn    = utils.np.int32(np.round(np.float32(im.shape) / bn))
        imbn    = cv2.resize(im, tuple(szbn[::-1]), interpolation=cv2.INTER_AREA)
        szcrop  = utils.prevmult(szbn,128)
        imcrop  = image.crop2D(imbn,szcrop)
        # remove bad pixels
        imcrop = image.unbad2D(imcrop, thresh=10, neib=3)
        imff   = micros[micro]['ctf'].phase_flip_cpu(imcrop, psizebn)
        # add one channel
        imff   = imff.reshape(imff.shape+(1,))
        yield(imff)
##

ctfstar = '/jasper/result/GPCR_GI/CtfFind/job002/micrographs_ctf.star'
# ctfstar = '/jasper/result/Nucleosome_20170427_1821/CtfFind/job005/micrographs_ctf.star'
# ctfstar = '/jasper/result/Braf_20170526_1206/CtfFind/job007/micrographs_ctf.star'


savpath = FLAGS.output_dir
# clean result directory
ft.rmtree_assure(savpath)
ft.mkdir_assure(savpath)

im_gen = image_generator(ctfstar)
# im = image_generator(ctfstar)

with tf.Graph().as_default() as g:
    with g.device(FLAGS.eval_device):
        tf_global_step = slim.get_or_create_global_step()
        model    = Model.create_instance(FLAGS.model_path, 'test', FLAGS.tfrecord_dir)
        classes  = model._dataset.example_meta['classes']
        cleanidx = classes.index('clean')

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:

            images = tf.placeholder(tf.float32,shape=(None,None,1))
            data   = {'image':model.data_single(images)}

            logits, end_points = model.network(data, is_training=False)

            rpn_score = logits['rpn_score']
            dxy_pred  = logits['dxy_pred']
            cls_score = logits['cls_score']
            rpn_prob  = tf.nn.softmax(rpn_score, name='rpn_prob') #[...,cleanidx]
            cls_prob  = tf.nn.softmax(cls_score, name='cls_prob')[...,1]

            _load_checkpoint()

            # coord   = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # apply CNN detection on properly rescaled image
            for micro in im_gen:
                im_py,cls_prob_py,rpn_prob_py,dxy_pred_py = sess.run([data['image'],cls_prob,rpn_prob,dxy_pred],feed_dict={images:micro})
                pscore = calc_single_score_per_particle(cls_prob_py[0],rpn_prob_py[0,...,cleanidx],dxy_pred_py[0])

                clf()
                imshow(pscore)
                imshow(im_py[0,:,:,0])
                pass

                # szbn = np.int32(bw.shape)
                # imbn = cv2.resize(im_py[0,:,:,0], tuple(szbn[::-1]), interpolation=cv2.INTER_AREA)
                #
                # image_py = sess.run(images,feed_dict={images:im_gen.next()})
                # # d_py,logits_py = sess.run([data,logits],feed_dict=feed_dict)
                # # create image overlay for each images in the batch
                # #for b in range(d_py['image'].shape[0]):
                # b = 0
                # ax = plt.subplot()
                # show_image(ax, np.squeeze(d_py['image'][b,:,:,0]))
                # # show results for all scales
                # for scale_keys in scale_key_sets:
                #     rpn_prob      = rpn_probs_py[scale_keys['rpn_prob']]
                #     rpn_bbox_pred = logits_py[scale_keys['rpn_bbox_pred']]
                #     bn   = d_py['image'].shape[1]/rpn_prob.shape[1]
                #     bbox = revert_xy(get_predicted_bboxes(rpn_prob, rpn_bbox_pred)[b])
                #     # obtain ground truth
                #     gtbox = sized_to_4coords(col_set_diff(d_py['object'], [[0,0,0,0]]))
                #     igbox = sized_to_4coords(col_set_diff(d_py['ignore'], [[0,0,0,0]]))
                #     plot_boxes(ax, gtbox, color='b')
                #     plot_boxes(ax, igbox, color='k')
                #     plot_boxes(ax, bbox*bn, color='g',linewidth=2)
                #     ax.set_title("g,b,k - detected,gt,ignore")
                #
                # fname = ft.file_only(d_py['image_name'].tobytes())
                # fname = os.path.join(savpath, 'result_%s.png' % fname)
                # fig   = plt.gcf()
                # fig.set_figheight(20.0)
                # fig.set_figwidth(26.0)
                # fig.subplots_adjust(wspace=.1, hspace=0.2, left=0.03, right=0.98, bottom=0.05, top=0.93)
                # print "Saving %s ..." % fname
                # fig.savefig(fname)
                # plt.close(fig)

                # clf()
            # ax.set_title("%s = %s" % (cstr[:-1], classes[:-1]))
############## GARBAGE #################################

            # rpn_probs = {}
            # for scale_keys in scale_key_sets:
            #     rpn_probs.update({scale_keys['rpn_prob']:model.score2prob(logits[scale_keys['rpn_score']])})


            # clf()
                # imshow(d_py['image'][-1])
                # imshow(rpn_prob_py[-1,:,:].max(axis=2))
##
# def get_predicted_bboxes(rpn_prob_py, rpn_bbox_pred_py):
#     _anchors = generate_anchors(base_size=cfg.ANCHOR_STRIDE_FACTOR,
#                                 ratios=cfg.ANCHOR_RATIOS,
#                                 scales=np.array(cfg.ANCHOR_SCALES))
#     batch   = rpn_prob_py.shape[0]
#     A       = rpn_prob_py.shape[3]
#     allbbox = []
#     for b in range(batch):
#         prob_max = rpn_prob_py[b].max(axis=2)
#         L, nl = label(prob_max > cfg.PROB_THRESH)
#         # bounding box for image b
#         bbbox = np.zeros((nl, 4), dtype=np.float32)
#         for l in range(nl):
#             # bounding box of a given label
#             lbbox = np.zeros((4,), dtype=np.float32)
#             lbbox[:2] = prob_max.shape
#             for a in range(A):
#                 bb = rpn_bbox_pred_py[b, :, :, a]
#                 pr = rpn_prob_py[b, :, :, a]
#                 # pixels of object l and anchor a
#                 amap   = np.logical_and(L == l + 1, pr > cfg.PROB_THRESH)
#                 aidxs  = np.where(amap)
#                 shifts = np.int32(zip(*aidxs))
#                 if shifts.size > 0:
#                     # shifted anchors
#                     anchors = _anchors[a] + np.concatenate([shifts, shifts], axis=1)
#                     # collect bounding box transforms
#                     bbt = bb[aidxs]
#                     # correct anchors with bounding box regressions
#                     bbinv = bbox_transform_inv(anchors, bbt)
#                     # mean bbox for a given anchor
#                     apr = pr[aidxs]
#                     abbox = np.sum(apr[:, None] * bbinv, axis=0) / apr.sum()
#                     # mean weight for a given anchor
#                     # aw      = apr.mean()
#                     lbbox[:2] = np.minimum(lbbox[:2], abbox[:2])
#                     lbbox[2:] = np.maximum(lbbox[2:], abbox[2:])
#             bbbox[l] = lbbox
#         allbbox.append(bbbox)
#     return allbbox
#
# def show_image(ax,im):
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     vmn = np.percentile(im, 1)
#     vmx = np.percentile(im, 99)
#     ax.imshow(im, vmin=vmn, vmax=vmx, cmap=plt.cm.gray)
#     ax.axis((0, im.shape[1], im.shape[0], 0))
#
# def plot_boxes(ax,boxes,color='b',linewidth=1):
#     for box in boxes:
#         rect = patches.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],linewidth=linewidth,edgecolor=color,facecolor='none')
#         ax.add_patch(rect)
# ##
#
# scale_key_sets = [{'rpn_score': 'rpn_score3', 'rpn_bbox_pred': 'rpn_bbox_pred3','rpn_prob': 'rpn_prob3'},
#             {'rpn_score': 'rpn_score4', 'rpn_bbox_pred': 'rpn_bbox_pred4', 'rpn_prob': 'rpn_prob4'},
#             {'rpn_score': 'rpn_score5', 'rpn_bbox_pred': 'rpn_bbox_pred5', 'rpn_prob': 'rpn_prob5'}]
