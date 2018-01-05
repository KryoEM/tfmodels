import tensorflow as tf
import mrcfile
from   myplotlib import imshow,clf
# from   image import cart_coords2D
import numpy as np
from   gpu import pick_gpus_lowest_memory
import os
from   symmetry import Symmetry


def fftfix(xyz,len):
    ''' inverts second half of x '''
    c        = len - len//2
    # invert all coordinates that cross image center
    loc      = xyz >= c
    # xys.max = len-1 => -1
    xyz[loc] -= len #- xyz[loc] - 2.0
    return xyz

def ifftfix(xyz,len):
    ''' inverts second half of x '''
    # invert all coordinates that cross image center
    # xyz = -1 => len-1
    loc      = xyz < -0.5
    xyz[loc] += len # - xyz[loc] - 2.0
    return xyz

vfile = '/jasper/models/gp140/EMDB5019.mrc'
# vfile = '/jasper/models/BetaGal/betagal1.5.mrc'

with mrcfile.open(vfile) as mrc:
    v = mrc.data

vlen = v.shape[0]

x,y = np.mgrid[:vlen,:vlen]
z   = np.zeros((vlen,vlen),np.float32)
xyz = np.dstack([x,y,z]) # construct triplets
# fix origin
xyz = fftfix(xyz,vlen)
# x,y,z = fftfix_values(x),fftfix_values(y),fftfix_values(z)
# apply rotation
M = Symmetry('c3').orient_matrices(10.0)

planesxyz = np.zeros(M.shape[:2]+(vlen,vlen,3),dtype=np.float32)
for n in range(M.shape[0]):
    for o in range(M.shape[1]):
        planesxyz[n,o] = np.dot(xyz,M[n,o])

# unfix origin
planesxyz = ifftfix(planesxyz,vlen)
# x,y,z = fftfix_values(x),fftfix_values(y),fftfix_values(z)

# xyz   = np.int32(np.round(np.dstack([x,y,z])))
# xyz   = np.stack([xyz,xyz])

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

    VS_py = sess.run(VS,feed_dict={V:v,planes:np.int32(np.round(planesxyz[0,[1]]))})

imshow(np.abs(VS_py[0]))
vi = np.fft.ifft2(VS_py[0])
imshow(np.real(vi))

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
