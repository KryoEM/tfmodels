import tensorflow as tf
import mrcfile
from   myplotlib import imshow,clf
# from   image import cart_coords2D
import numpy as np
from   gpu import pick_gpus_lowest_memory
import os


def fftfix_values(x):
    ''' inverts last half of x '''
    len   = x.shape[0]
    c     = len - len//2
    x[c:] = len - x[c:]
    return x

vfile = '/jasper/models/gp140/EMDB5019.mrc'
# vfile = '/jasper/models/BetaGal/betagal1.5.mrc'

with mrcfile.open(vfile) as mrc:
    v = mrc.data

vlen = v.shape[0]

x,y = np.mgrid[:vlen,:vlen]
z   = np.zeros((vlen,vlen),np.float32)
# fix origin
x,y,z = fftfix_values(x),fftfix_values(y),fftfix_values(z)
# apply rotation
# unfix origin
x,y,z = fftfix_values(x),fftfix_values(y),fftfix_values(z)
xyz   = np.int32(np.round(np.dstack([x,y,z])))
xyz   = np.stack([xyz,xyz])

# define 3d input
V       = tf.placeholder(tf.complex64, shape=(vlen, vlen, vlen))
planes  = tf.placeholder(tf.int32, shape=(None,vlen, vlen, 3))
VF = tf.fft3d(V)
# obtain slices
VS = tf.gather_nd(VF,planes)

gpuid = pick_gpus_lowest_memory(1, 0)[0]
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuid
session_config = tf.ConfigProto(allow_soft_placement=False)  # use only one gpu for eval
device = '/gpu:0'  # % gpuid

glob_init = tf.global_variables_initializer()
loc_init  = tf.local_variables_initializer()
with tf.Session(config=session_config) as sess:
    sess.run(loc_init)
    sess.run(glob_init)

    VS_py = sess.run(VS,feed_dict={V:v,planes:xyz})


vi = np.fft.ifft2(VS_py[1])



#
# # create 2D coordinates in fft domain
# X,Y = tf.meshgrid(x,x)
# Z   = tf.zeros(X.get_shape(),dtype=tf.float32)
#


############# JUNK #############################
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
