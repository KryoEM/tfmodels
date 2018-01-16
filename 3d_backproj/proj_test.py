from   __future__ import absolute_import, division, print_function
# pip install future
from   builtins import *
import tensorflow as tf
import mrcfile
from   myplotlib import imshow,clf
import numpy as np
from   gpu import pick_gpus_lowest_memory
import os
from   networks import Proj
from   image import make_cubic

if __name__ == "__main__":

    # vfile = '/jasper/data/train_data_sets/backproj/checkpoint/volume.mrc'
    # vfile = '/jasper/models/gp140/EMDB5019.mrc'
    vfile = '/jasper/models/BetaGal/betagal1.5.mrc'

    with mrcfile.open(vfile) as mrc:
        v = make_cubic(mrc.data)

    # move z axis to first position to match EMAN symmetry convention
    v    = np.transpose(v,[2,1,0])

    # move symmetry axis to match EMAN convention
    # v    = np.roll(v,-1,axis=(0,1,2))

    vlen = v.shape[0]
    vs   = np.fft.ifftshift(v)

    # define 3d input
    V    = tf.placeholder(tf.complex64, shape=(vlen, vlen, vlen))
    proj = Proj(vlen,10.0,'d2')
    P    = proj(V)

    gpuid = pick_gpus_lowest_memory(1, 0)[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuid
    session_config = tf.ConfigProto(allow_soft_placement=False)  # use only one gpu for eval
    device = '/gpu:0'  # % gpuid

    glob_init = tf.global_variables_initializer()
    loc_init  = tf.local_variables_initializer()
    with tf.Session(config=session_config) as sess:
        sess.run(loc_init)
        sess.run(glob_init)
        feed_dict = {V: vs}
        # feed_dict.update(proj.feed_dict())
        P_py = sess.run(P,feed_dict=feed_dict)

    vi1 = np.fft.fftshift(P_py[0,1])
    imshow(vi1)
    vi2 = np.fft.fftshift(P_py[1,1])
    imshow(vi2)

    fname,ext = os.path.splitext(vfile)
    np.save(fname+'_projections.npy',P_py)

############# JUNK #############################
# imshow(np.real(vi1-vi2))
# vi2 = np.fft.fftshift(np.fft.ifft2(VS_py[1,1]))

# vi1 = np.fft.fftshift(np.fft.ifft2(VS_py[0,1]))

#np.dot(xyz, np.float32([[np.cos(alpha),np.sin(alpha),0],[-np.sin(alpha),np.cos(alpha),0],[0,0,1]]))
# imshow(np.fft.fftshift(np.abs(VS_py[0])))
# imshow(np.abs(VS_py[0]))

#
# # create 2D coordinates in fft domain
# X,Y = tf.meshgrid(x,x)
# Z   = tf.zeros(X.get_shape(),dtype=tf.float32)
#
# def ifftshift_values(x,len):
#     c = len//2
#     if np.mod(len,2) == 0:
#         gloc    = x >= c
#         x[x>=c] -= c
#     else:
#         x[x>c]  -= (c+1)
#         x[x<c]  += c
#         x[x==c]  = len-1
#     return x
