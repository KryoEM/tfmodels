import os
import tensorflow as tf
from   model_generic import Model
import numpy as np
from   arg_parsing import load_config
from   gpu import pick_gpus_lowest_memory
from   myplotlib import imshow,clf,savefig
from   rpn.generate_anchors import generate_anchors
from   rpn.bbox.bbox_transform import bbox_transform_inv,sized_to_4coords,revert_xy
from   matplotlib import pyplot as plt
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
from   tfrecorder import plot_class_coords
from   star.tools import save_coords_in_star
from   tensorpack import dataflow as df
from   tensorpack import QueueInput,PrintData
from   utils import poolcontext
from   functools import partial
from   tfutils import config_gpus
from   utils import tprint,part_idxs
from   image import plot_coord_rgb

from autopick import cfg

############## Functions #####################
def _load_checkpoint(sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # load model
    ckpt_path = ckpt.model_checkpoint_path
    saver     = tf.train.Saver()
    saver.restore(sess, ckpt_path)

def calc_single_score_per_particle(args):
    '''Combines region proposal, classification and particle centering scores into one map with one score
       per particle.
       cls_prob - probability that this particle is isolated
       rpn_prob - probability that there is a particle
       dxy_pred - predicted location corrections around the particle'''

    cls_prob, rpn_prob, dxy_pred = args

    # particle size in feature map
    D = cfg.PART_D_PIXELS / cfg.STRIDES[0]

    # zero boundaries
    rpn_prob = image.zero_border(rpn_prob, D // 2).copy()
    cls_prob = image.zero_border(cls_prob, D // 2).copy()

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
    # obtain circle locations
    hx, hy = np.where(H)
    xs, ys = np.where(bw)
    rh = r.flat[np.ravel_multi_index([hx, hy], H.shape)]
    for x, y in zip(xs, ys):
        xhx = np.minimum(np.maximum(hx + x - fr - 1, 0), bwsz[0] - 1)
        yhy = np.minimum(np.maximum(hy + y - fr - 1, 0), bwsz[1] - 1)
        # normalize distances by the corresponding radius
        dn = (dxy.flat[np.ravel_multi_index([xhx, yhy], bwsz)] - dxy[x, y]) / rh
        # calc mean normalized radius
        # pmean = dn.mean() #np.mean((dxy.flat[np.ravel_multi_index([xhx, yhy], bwsz)] - dxy[x, y]) / rh)
        # majority vote - most pixels shall follow the funnel pattern
        # mvote = np.median(dn) >= 0.5 #/ float(rh.size)
        cmap[x, y] = dn.min()  # np.median(dn)

    # analyse region properties and leave only the strongest cmap scores per particle
    regions = regionprops(label(bw))
    for region in regions:
        coords = region['Coordinates']
        idxs = np.ravel_multi_index(coords.transpose(), bwsz)
        # obtain switch off indices
        offidxs = idxs[cmap.flat[idxs] < cmap.flat[idxs].max()]
        # switch off non-maxima
        cmap.flat[offidxs] = 0
    return cmap

def preprocess_micro(micro,ctf,psize,bn):
    im = mrc.load(micro)[0]
    psizebn = psize * bn
    szbn    = utils.np.int32(np.round(np.float32(im.shape) / bn))
    imbn    = cv2.resize(im, tuple(szbn[::-1]), interpolation=cv2.INTER_AREA)
    szcrop  = utils.prevmult(szbn, 128)
    imcrop  = image.crop2D(imbn, szcrop)
    # remove background
    imcrop  = image.background_remove2D(imcrop,cfg.LP_RES)
    # remove bad pixels
    imcrop  = image.unbad2D(imcrop, thresh=5, neib=3)
    imff    = ctf.phase_flip_cpu(imcrop, psizebn)
    # add one channel
    return imff.reshape(imff.shape + (1,))

class MicrosGenerator():
    def __init__(self,micros):
        # micros = parse_ctf_star(ctfstar)
        # select micros by maxres
        self.__micros = micros #{key: micros[key] for key in micros if micros[key]['maxres'] < cfg.CTF_RES_THRESH}
        # self.shape = shape #mrc.shape(micros.iterkeys().next())[1:]
    @staticmethod
    def create_partition(ctfstar,nbatches):
        ''' Create a number of generators each serving a chunk of data '''
        micros = parse_ctf_star(ctfstar)
        # select micros by maxres
        micros = {key: micros[key] for key in micros if micros[key]['maxres'] < cfg.CTF_RES_THRESH}
        shape  = mrc.shape(micros.iterkeys().next())[1:]
        micro_chunks = part_idxs(micros.keys(),nbatches=nbatches)
        basedir = os.path.basename(os.path.dirname(micros.iterkeys().next()))
        return [MicrosGenerator({m:micros[m] for m in chunk}) for chunk in micro_chunks],basedir,shape
    def size(self):
        return len(self.__micros)
    def reset_state(self):
        pass
    # def get_base_dir(self):
    #     return os.path.basename(os.path.dirname(self.__micros.iterkeys().next()))
    def get_data(self):
        for micro in self.__micros:
            # im = mrc.load(micro)[0]
            yield micro,self.__micros[micro]['ctf']

def adjust_coodinates(coords,srcsz,dstsz,stride):
    '''Adjust coordinates from cropped feature plane to match the original image coordinates '''
    unpadl = (dstsz - srcsz*stride) // 2
    coords = coords*stride + unpadl[None,:]
    return coords

def output_coords(bn,sz,outmicrodir,args):
    '''Converts particle maps into coordinates and saves coordinates with the png file'''
    pmap,imdisp,micro = args
    # particle coordinates in the original micrograph coordinates
    coords = np.column_stack(np.where(pmap > cfg.MIN_CLEARANCE))
    szpmap = np.int32(pmap.shape)
    szdisp = np.int32(imdisp.shape[:2])
    szbn   = utils.np.int32(np.round(sz / bn))
    # im coordinate system for display
    coords_disp = adjust_coodinates(coords, szpmap, szdisp, cfg.STRIDES[0])
    # original micrograph coordinate system
    coords_orig = adjust_coodinates(coords, szpmap, szbn, cfg.STRIDES[0])*bn
    starname    = os.path.join(outmicrodir, ft.file_only(micro) + '_manualpick.star')
    save_coords_in_star(starname, coords_orig)
    #### SAVE figure as well ######
    figname = os.path.join(outmicrodir, ft.file_only(micro) + '.png')
    imrgb   = plot_coord_rgb(np.squeeze(imdisp), {'particles': coords_disp}, cfg.PART_D_PIXELS, cfg.CIRCLE_WIDTH)
    cv2.imwrite(figname,imrgb)
    # return number of particles
    return coords_orig.shape[0]

def init_dataflow(ctfstar,batch_size):
    ''' This function creates dataflow that reads and preprocesses data in parallel '''
    augm = df.imgaug.AugmentorList([df.imgaug.MeanVarianceNormalize()])
    # create partitioned generators, one for each element in a batch
    dss0,basedir,shape = MicrosGenerator.create_partition(ctfstar,batch_size)
    # preprocess input
    dss1 = [df.MapData(ds0, lambda dp: [augm.augment(preprocess_micro(dp[0], dp[1], psize, bn)), np.array(dp[0])]) for ds0 in dss0]
    # prefetch each generator in a separate process with buffer of 4 images per process
    # dss1 = [df.PrefetchDataZMQ(ds1, nr_proc=1, hwm=2) for ds1 in dss1]
    dss1 = [df.PrefetchData(ds1, nr_prefetch=4, nr_proc=1) for ds1 in dss1]
    # join all dataflows
    ds1  = df.RandomMixData(dss1)
    # ds1  = df.JoinData(dss1)
    ds   = df.BatchData(ds1, batch_size)
    ds.reset_state()
    return ds,basedir,shape

########## Main starts here #######################################

# --------- Handle params ------------------------------------
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
tf.app.flags.DEFINE_string('use_gpu', False, 'if to use gpu')
FLAGS = load_config(tf.app.flags.FLAGS, 'eval')
tf.logging.set_verbosity(tf.logging.INFO)

if FLAGS.use_gpu:
    gpuid = pick_gpus_lowest_memory(1, 0)[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuid
    session_config = tf.ConfigProto(allow_soft_placement=False) # use only one gpu for eval
    device = '/gpu:0' #% gpuid
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    session_config = tf.ConfigProto(allow_soft_placement=True)
    device = '/cpu:0'
# ------------------------------------------------------------

ctfstar = '/jasper/result/rhodopsin-Gi/CtfFind/job003/micrographs_ctf.star'
outdir  = '/jasper/result/rhodopsin-Gi/cnnpick/'
#
# ctfstar = '/jasper/result/rhodopsin-Gi_new/CtfFind/job004/micrographs_ctf.star'
# outdir  = '/jasper/result/rhodopsin-Gi_new/cnnpick/'

psize = path2psize(ctfstar, 'CtfFind')
# particle diameter in pixels
D = path2part_diameter(ctfstar) / psize
# calcualte micrographs binning factor
bn = D / cfg.PART_D_PIXELS

####### Read Relion jobs infor and prepare picking params ########
# create dataflow
ds,basedir,shape = init_dataflow(ctfstar,FLAGS.batch_size)
outmicrodir = os.path.join(outdir,basedir)

# initialize output directory
ft.rmtree_assure(outdir)
ft.mkdir_assure(outdir)
ft.mkdir_assure(outmicrodir)

##################################################################
tprint("Picking from ~%d micrographs with CTF better than %.2fA resolution" % (ds.size()*FLAGS.batch_size,cfg.CTF_RES_THRESH))
with tf.Graph().as_default() as g:
    model    = Model.create_instance(FLAGS.model_path, 'test', FLAGS.tfrecord_dir)
    classes  = model._dataset.example_meta['classes']
    cleanidx = classes.index('clean')

    with g.device(device):
        with tf.name_scope('inputs'):
            images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,None, None, 1))
            data   = {'image': images}

    # create model
    logits, end_points = model.network(data, is_training=False)

    rpn_score = logits['rpn_score']
    dxy_pred  = logits['dxy_pred']
    cls_score = logits['cls_score']
    rpn_prob  = tf.nn.softmax(rpn_score, name='rpn_prob')
    cls_prob  = tf.nn.softmax(cls_score, name='cls_prob')[..., 1]
    glob_init = tf.global_variables_initializer()
    loc_init  = tf.local_variables_initializer()
    with tf.Session(config=session_config) as sess:
        sess.run(loc_init)
        sess.run(glob_init)
        _load_checkpoint(sess)

        micro_count,part_count = 0,0
        for dp in ds.get_data():
            # apply model on input images
            cls_prob_py,rpn_prob_py,dxy_pred_py,im_py = sess.run([cls_prob,rpn_prob,dxy_pred,data['image']],feed_dict={images:dp[0]})

            # parallel process probability and feature maps and ouput results
            # prepare output function for map
            out_coords = partial(output_coords,bn,shape,outmicrodir)
            prob_data  = zip(*[cls_prob_py,rpn_prob_py[...,cleanidx],dxy_pred_py])
            with poolcontext(processes=FLAGS.batch_size) as pool:
                # convert probabilities into particle maps
                pmaps = pool.map(calc_single_score_per_particle,prob_data)
                # convert particle maps to coordinates and save results
                out_data = zip(*[pmaps,dp[0],dp[1]])
                nparts = pool.map(out_coords, out_data)
            # for pmap,im,micro in zip(*[pmaps,dp[0],dp[1]]):
            #     output_map(bn,ds0.shape,outmicrodir,(pmap,im,micro))

            new_parts    = np.sum(nparts)
            part_count  += new_parts
            micro_count += len(pmaps)
            tprint("micros %d/%d: parts %d/%d" % (micro_count, ds.size()*FLAGS.batch_size, new_parts, part_count))
            if part_count >= cfg.MAX_PARTICLES:
                break

############## GARBAGE #################################

        # ds0  = MicrosGenerator(ctfstar)
        # normalizer operation
        # ds1  = df.PrefetchData(ds0,2*batch_size,batch_size)
        # ds1  = df.ThreadedMapData(
        #     ds0, nr_thread=batch_size,
        #     map_func=lambda dp: [augm.augment(preprocess_micro(dp[0], dp[1], psize, bn)), np.array(dp[0])],
        #     buffer_size=2*batch_size)
        # ds1 = df.PrefetchDataZMQ(ds1, nr_proc=1)

        # check rpn_prob for problems
            # regions  = regionprops(label(rpn_prob > cfg.PROB_THRESH))
            # areas = np.int32([r['Area'] for r in regions])
            # if np.any(areas > D**2):
            #     print "Something wrong with micrograph - large area response, skipping ..."
            #     rpn_prob[:] = 0.0

            # read orogonal micrograph
            # im    = mrc.load(micro)[0]
            # szbn  = utils.np.int32(np.round(np.float32(im.shape) / bn))
            # imbn  = cv2.resize(im, tuple(szbn[::-1]), interpolation=cv2.INTER_AREA)

            # convert micrograph to pickable size
            # imff = preprocess_micro(imbn,micros[micro]['ctf'],psizebn)

            # def image_generator(ctfstar):
            #     micros = parse_ctf_star(ctfstar)
            #     psize  = path2psize(ctfstar,'CtfFind')
            #     # particle diameter in pixels
            #     D      = path2part_diameter(ctfstar) / psize
            #     # calcualte micrographs binning factor
            #     bn     = D / cfg.PART_D_PIXELS
            #     for micro in micros:
            #         im      = mrc.load(micro)[0]
            #         yield(imff)

            # suffname = os.path.join(outdir,'coords_suffix_cnnpick.star')
            # with open(suffname, 'w') as sufffile:
            #     sufffile.write(ctfstar)


            # echo       CtfFind / job008 / micrographs_ctf.star > AutoPick / job023 / coords_suffix_autopick.star

            # clf()
            # imshow(pscore)
            # imshow(imff)
            # pass

            # coord   = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

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

    # im_gen = image_generator(ctfstar)
    # im = image_generator(ctfstar)



    ##

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
