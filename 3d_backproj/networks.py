from   __future__ import absolute_import, division, print_function
# pip install future
from   builtins import *

import tensorflow as tf
import numpy as np
from   symmetry import Symmetry
from   canton.cans import Can,Conv2DFull,BatchRenorm,Conv2D,PReLU,rnn_gen,Dense,AvgPool2D,MaxPool2D
from   tensorpack import models
import cfg
from   tfutils import get_visible_device_list,get_available_gpus,config_gpus
from   myplotlib import imshow

def fftfix(xyz,len):
    ''' inverts second half of x '''
    c        = len - len//2
    loc      = xyz >= c
    xyz[loc] -= len
    return xyz

def ifftfix(xyz,len):
    ''' inverts second half of x '''
    loc      = xyz < -0.5
    xyz[loc] += len
    return xyz

def calc_planes_data_fft_3D(vlen,delta_deg,symmetry):
    ''' Calculates plane coordinates in 3D for slicing volume in FFT domain '''
    # vlen = V.get_shape()[0]._value
    x,y  = np.mgrid[:vlen,:vlen]
    z    = np.zeros((vlen,vlen),np.float32)
    xyz  = np.dstack([x,y,z]) # construct triplets
    # fix origin
    xyz  = fftfix(xyz,vlen)
    # apply rotation
    M    = Symmetry(symmetry).orient_matrices(delta_deg)
    # create rotated planes with coordinates into 3D fft of V
    planesxyz = np.zeros(M.shape[:2]+(vlen,vlen,3),dtype=np.float32)
    for n in range(M.shape[0]):
        for o in range(M.shape[1]):
            planesxyz[n,o] = np.dot(xyz,M[n,o])
    # unfix origin
    planesxyz = ifftfix(planesxyz,vlen)
    return np.int32(np.round(planesxyz))

def polar_index_fft_2D(vlen):
    ''' Create a 2D indices that map 1D radial data to 2D '''
    x,y  = np.mgrid[:vlen,:vlen]
    xy   = np.dstack([x,y])
    xy   = fftfix(xy,vlen)
    r    = np.int32(np.round(np.sqrt(np.sum(xy**2,axis=-1))))
    r    = np.minimum(r,vlen//2-1)
    return r

class Proj(Can):
    # noinspection PyCompatibility
    def __init__(self, vlen, delta_deg, symmetry):
        super(Proj, self).__init__()
        # super().__init__()
        # actual plane coordinates in 3D
        planes_data  = calc_planes_data_fft_3D(vlen,delta_deg,symmetry)
        self._planes = tf.convert_to_tensor(planes_data,dtype=tf.int32)
        self._nsym,self._nviews = planes_data.shape[:2]
        # self._planes = tf.placeholder(tf.int32, shape=(None,None,vlen, vlen, 3),name='plane_coords_3D')

    def fft2D(self,V):
        ''' gather view slices in FFT domain and reurn them '''
        VF      = tf.fft3d(V)
        # 2D slices at planes locations
        # symmetry x nviews x xdim x ydim x 3dcoord
        return tf.gather_nd(VF,self._planes)

    def space2D(self,V):
        ''' Project volume onto views and return 2D representation in space '''
        VS = self.fft2D(V)
        # tensor of 2D projections in space domain
        return tf.real(tf.ifft2d(VS))

    # def nsym(self):
    #     return tf.shape(self._planes.shape)[0]
    #
    # def nviews(self):
    #     return

    def __call__(self,V):
        ''' network that inputs a tensor volume and projects it to all symmetrical units '''
        return self.space2D(V)

def huber(a,lam):
    return (lam**2)*(tf.sqrt(1.0+a/(lam**2))-1.0)

class Backproj(Can):
    def __init__(self, vlen, nviews, delta_deg, symmetry):
        super(Backproj, self).__init__()
        MEAN_VAL = 1.0
        # super().__init__()
        # this is the backprojected volume
        self._Vr    = tf.get_variable('vol_real', dtype=tf.float32, shape=(vlen,)*3,initializer=tf.zeros_initializer(),trainable=True)
        # we don't want the imaginary part to change, so trainable=False
        self._Vi    = tf.get_variable('vol_imag', dtype=tf.float32, shape=(vlen,)*3,initializer=tf.zeros_initializer(),trainable=False)
        # coefficients that will select which candidates contribute to the view average
        self._alph  = tf.get_variable('alpha', dtype=tf.float32, shape=(cfg.NCAND,nviews,1,1),initializer=tf.zeros_initializer(),trainable=True)
        # 1D represenation for radial log variances
        self._logsig2  = tf.get_variable('logsig2', dtype=tf.float32,
                                      initializer=tf.constant(np.log((MEAN_VAL**2)*(vlen**2)*(2*np.pi*np.arange(vlen//2,dtype=np.float32)+1.0)),np.float32),
                                      trainable=True)
        self._logb     = tf.get_variable('logb', dtype=tf.float32,initializer=tf.constant(np.float32(np.log(MEAN_VAL))),trainable=True)
        self._vlen     = vlen
        self._1D_2D    =  tf.convert_to_tensor(polar_index_fft_2D(vlen),dtype=tf.int32)
        self._proj     = Proj(vlen, delta_deg, symmetry)

    def score(self,Pin):
        gpus = get_available_gpus()

        # projection onto all symmetric units
        with tf.device(gpus[0]):
            V    = tf.complex(self._Vr,self._Vi)
            # Obtain fourier slices
            F    = self._proj.fft2D(V)

            # Calculate laplace/l1 regularization
            b    = tf.exp(self._logb)
            l1   = tf.reduce_mean(tf.abs(self._Vr))/b
            l1  += self._logb

        with tf.device(gpus[1]):
            # obtain difference between reference and input
            Fin  = tf.fft2d(tf.cast(Pin,tf.complex64))
            # nviews x nsym x ncandidates x w x h
            diff = F[:,None,...] - Fin[None,...]
            l2   = tf.real(diff*tf.conj(diff))

        with tf.device(gpus[2]):
            ls2  = tf.gather(self._logsig2,self._1D_2D)
            s2   =  tf.exp(ls2)

            # average over all symmetries
            l2   = tf.reduce_mean(l2,axis=0)
            # weight candidates
            l2   = tf.reduce_sum(l2*tf.nn.softmax(self._alph,dim=0),axis=0)
            # average over views
            l2   = tf.reduce_mean(l2,axis=0)
            # normalize by noise variance
            l2   = l2/(2.0*s2) #[None,...])
            # average over all frequency components
            l2   = tf.reduce_mean(l2,axis=(-1,-2))

            # add mean of logvar over frequency components
            l2  += tf.reduce_mean(self._logsig2)

            # log likelihood
            ll   = l2 + l1

            # remove average over views and candidates to avoid too small values in the exponent
            ll  -= cfg.EXP_NORM*tf.stop_gradient(ll)
            lhood = tf.exp(-ll)

        tf.summary.scalar('ll', ll)
        # tf.summary.scalar('factor ', tf.exp(logfact))
        tf.summary.scalar('b', b)
        tf.summary.scalar('sig2', tf.reduce_sum(self._logsig2))

        return tf.identity(-lhood,'cost')


##################### GARBAGE ####################################
        # nviews = self._proj._nviews
        # P  = self._proj(V)

        # diff = F[:,:,None,...] - tf.tile(Fin[None,...],(nviews,1,1,1))[None,...]
        # l2  -= tf.stop_gradient(tf.reduce_mean(l2))
        # frequency radius
        # r    = self._vlen//2
        # number of symmetric units
        # nsym = tf.cast(self._proj.nsym(),tf.float32)
        # number of input elements
        # nels = tf.cast(tf.shape(Pin)[0],tf.float32)

        # aqverage result over samples
        # lhood = tf.reduce_mean(err2)

        # l2 = tf.sqrt(tf.reduce_mean(err))/self._vlen

        # l1 = tf.reduce_mean(av)/tf.maximum(tf.stop_gradient(tf.reduce_mean(av)),1e-9)
        # clone input projections across all symmetric units
        # return l2 +l1#+1000.0*tf.reduce_sum(alph**4)

        # tf.summary.scalar('Fin', tf.reduce_mean(tf.cast(Fin[-2,-1]*tf.conj(Fin[-2,-1]),tf.float32)))
        # tf.summary.scalar('Pin', tf.reduce_mean(tf.cast(Pin[-2,-1]*tf.conj(Pin[-2,-1]),tf.float32)))

       # glob_init = tf.global_variables_initializer()
        # loc_init = tf.local_variables_initializer()
        # with tf.Session().as_default() as sess:
        #    with tf.Graph().as_default() as g:
        #        with g.device('/cpu:0'):
        #             sess.run(loc_init)
        #             sess.run(glob_init)
        #             coord = tf.train.Coordinator()
        #             tf.train.start_queue_runners(coord=coord)
        #             Pin_py,Fin_py = sess.run([Pin,Fin])

# def score_matrix(vlen,name=None):
#     # contruct comparison can
#     with tf.variable_scope(name):
#         can  = Can()
#         comp = compare_maps(vlen,name='comp_one_ref')
#         can.incan(comp)
#     def call(ims,refs):
#         scores = tf.map_fn(lambda x: comp(ims,x[None,...]),refs,parallel_iterations=1,swap_memory=True)
#         # calc input correspondence to refs
#         scores = tf.nn.softmax(scores,dim=0)
#         return scores
#     can.set_function(call)
#     return can


        # alph = tf.nn.softmax(self._alph, dim = -1)
        # apply shrinkage operation
        # alph = tf.nn.dropout(alph,0.5)
        # alph = tf.maximum(alph,1e-8)
        # alph = alph/tf.reduce_sum(alph, axis = -1, keep_dims=True)

        # Pavg = tf.reduce_sum(Pin*alph, axis = -1)
        # err  = (P-Pavg[None,...])**2

        # err  = (P[...,None]-Pin[None,...])**2
        # err  = tf.reduce_sum(err*alph[None,...],axis=-1)


# def compare_maps(vlen,name=None):
#     MAX_CH = 8
#     ''' Network that compares reference vs projection map '''
#     with tf.variable_scope(name):
#         l     = vlen
#         nin,nout = 2,4
#         fout  = np.minimum(np.int32(nout*(2.0**(np.floor(np.log2(vlen))-1.0))),MAX_CH)
#         can   = Can()
#         pool  = MaxPool2D(kernel=3, stride=2)
#         fc    = Dense(fout,1,name='fc')
#         convs = []
#         while l > 2:
#             convs.append(Conv2DFull(nin=nin,nout=nout,kernel=3,stride=1,name='conv%d' % l))
#             nin   = nout
#             nout  = np.minimum(nout*2,MAX_CH)
#             l    /= 2
#         can.incan([pool]+convs)
#     def call(ims,ref):
#         with tf.name_scope(name):
#             # replicate reference according to number of imputs and stack to contruct comparison channels
#             res = convs[0](tf.stack([ims,tf.tile(ref,[tf.shape(ims)[0],1,1])],axis=3))
#             res = pool(res)
#             for conv in convs[1:]:
#                 res = pool(conv(res))
#             # final average pooling
#             res = tf.reduce_mean(res,axis=(1,2),name='average_pool')
#             # fully connected to produce similarity measure
#             res = fc(res)
#             # returns one score per input image
#             return tf.squeeze(res,axis=-1)
#     can.set_function(call)
#     return can

# use only one symmetric unit for calculating scores
# scores = score_matrix(self._vlen,name='score_matrix')(Pin,P[0])

# calculate view average
# Pavg = tf.tensordot(scores, Pin, axes=[[-1], [0]], name='view_average')
# view error
# err  = (P-Pavg[None,...])**2
# weight errors by sum of view contributions
# w    = tf.reduce_sum(scores,axis=1)[None,...,None,None]
# werr = err*w/tf.reduce_sum(w)

# res = compare_maps(self._vlen,name='compare_ref')(P_in,P[0,0][None,...])

# def feed_dict(self):
    #     ''' needs to be fed with all the data to session '''
    #     return {self._planes:self._planes_data}

    # def feed_dict(self):
    #     return self._proj.feed_dict()

    # def __call__(self, hidden, inp):
