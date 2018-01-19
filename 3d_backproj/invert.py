from   __future__ import absolute_import, division, print_function
# pip install future
from   builtins import *
from   networks import Backproj
from   canton.cans import Can,BatchRenorm,Conv2D,PReLU,rnn_gen,Dense,AvgPool2D,MaxPool2D
import tensorflow as tf
import numpy as np
from   tensorpack import (ModelDesc,InputDesc,get_current_tower_context,GraphProfiler,
                          get_nr_gpu,optimizer,Callback,TrainConfig,logger,
                          QueueInput,PeriodicTrigger,ModelSaver,get_model_loader,HyperParamSetter,
                          ScheduledHyperParamSetter,HyperParamSetterWithFunc,GPUUtilizationTracker,
                          SyncMultiGPUTrainerReplicated,SimpleTrainer,launch_train_with_config)

from   tensorpack.tfutils.summary import add_moving_summary
import cfg
from   data import ProjDataFlow
from   tensorpack.tfutils.sessinit import ChainInit,TryResumeTraining # SaverRestore,SessionInit,
from   tensorpack.tfutils.common import get_global_step_var
import os
import argparse
from   tfutils import get_visible_device_list,config_gpus
from   myplotlib import imshow,clf
import mrcfile

def summary_slice(slice,name):
    slice = tf.py_func(np.fft.fftshift, [slice], tf.float32)
    tf.summary.image(name, slice, max_outputs=1)

class Model(ModelDesc):
    def __init__(self,vlen,nviews):
        # the main can that will hold model maintenance functionality
        self._vlen     = vlen
        self._nviews   = nviews
        self._backproj = None

    def _get_inputs(self):
        return [
            # assumes only one symmetric unit as input batch x  vlen x vlen
            InputDesc(tf.float32, (cfg.NCAND, self._nviews, self._vlen, self._vlen), 'projections'),
        ]

    def _build_graph(self, input):
        Pin = input[0]
        # Pin = tf.convert_to_tensor(self.P_py, dtype=tf.float32)
        self._backproj = Backproj(self._vlen,self._nviews,10.0,'d2')
        self.cost = self._backproj.score(Pin)

        summary_slice(self._backproj._Vr[:,:,0][None,...,None], '0 xyplane')
        summary_slice(self._backproj._Vr[:,0,:][None,...,None], '0 xzplane')
        summary_slice(self._backproj._Vr[0,:,:][None,...,None], '0 yzplane')

        add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=cfg.LEARN_RATE, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.AdamOptimizer(lr)
        # return tf.train.GradientDescentOptimizer(lr)

    def get_volume(self):
        return self._backproj._Vr

class VolumeSaver(Callback):
    ''' Periodically saves the estimated density volume '''
    def __init__(self,model):
        self._model = model
    # def _setup_graph(self):
    def _trigger(self):
        V = model.get_volume()
        # with tf.get_default_session() as sess:
        V_py  = tf.get_default_session().run(V)
        V_py  = np.fft.fftshift(V_py)
        vname = os.path.join(logger.get_logger_dir(),'volume.mrc')
        mrcfile.new(vname,data=V_py,overwrite=True)

def learning_rate_fun(epoch_num,learning_rate):
    ''' Sets learning rate as a function of epoch number '''
    # using a simple exponential right now
    if np.mod(epoch_num,cfg.DECAY_EPOCHS) == 0:
        return cfg.LEARNING_DECAY*learning_rate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='logdir', default='')
    args = parser.parse_args()

    # P_py  = np.load('/jasper/models/gp140/P_py.npy')
    Ppy    = np.load('/jasper/models/BetaGal/betagal1.5_projections.npy')

    Ppy    = Ppy[0] # leave only first symmetric unit
    vlen   = Ppy.shape[-1]
    nviews = Ppy.shape[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = get_visible_device_list(1)
    global_step = get_global_step_var()

    # set logger directory for checkpoints, etc
    logger.set_logger_dir(args.logdir,action='k')

    steps_per_epoch = cfg.EPOCH_STEPS
    model    = Model(vlen,nviews)
    traincfg = TrainConfig(
                model = model,
                data  = QueueInput(ProjDataFlow(Ppy)),
                callbacks=[
                    PeriodicTrigger(ModelSaver(), every_k_epochs=5),
                    PeriodicTrigger(VolumeSaver(model), every_k_epochs=5),
                    # prevent learning in the first epoch
                    # MemInitHyperParamSetter('learning_rate_mask',(0,1)),
                    # controls learning rate as a function of epoch
                    HyperParamSetterWithFunc('learning_rate',learning_rate_fun)
                    # GraphProfiler()
                    # GPUUtilizationTracker(),
                ],
                steps_per_epoch=steps_per_epoch,
                max_epoch=200000,
                # first time load model from checkpoint and reset GRU state
                session_init=ChainInit([TryResumeTraining()])#,ResetInit(model)])
                #session_config=tf.ConfigProto(log_device_placement=True) #config_gpus(1)
    )

    trainer = SimpleTrainer()
    # with tf.contrib.tfprof.ProfileContext(logger.get_logger_dir()) as pctx:
    launch_train_with_config(traincfg, trainer)

################# JUNK ###############
    # Pin = tf.placeholder(tf.float32, shape=P_py.shape)
    # backp = Backproj(vlen, 10.0, 'c3')
    # l = backp.score(Pin)

# class Invert(Can):
#     def __init__(self, num_in, num_h):
#         super(Invert, self).__init__()
#         # assume input has dimension num_in.
#         self.num_in, self.num_h = num_in, num_h
#
#     def __call__(self, hidden, inp):
#         # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
#         wzr, wz, wr, w, wa, wd = self.wzr,self.wz, self.wr, self.w, self.wa, self.wd
#         rnu,state = hidden[...,0][...,None],hidden[...,1:]
#
#         # GRU-like update of the hidden (movie) state using input with no rows and columns
#         zr,_      = wzr(tf.concat([hidden, inp], axis=-1))
#         # decide which part of the history and input will update the state
#         r         = tf.sigmoid(wr(zr))
#         # decide how fast the state is updated
#         z         = tf.sigmoid(wz(zr))
#         s_c,dil   = w(r*tf.concat([hidden, inp], axis=-1))
#         s_new     = (1-z)*state + z*s_c
#
#         # SBNR-like update of the rnu estimate with IIR
#         # alpha = softmax, 2 params that sum up to 1, use as IIR regression
#         alpha     = wa(dil)
#         r_new     = alpha[...,0][...,None]*rnu + alpha[...,1][...,None]*wd(s_new)
#
#         # the updated state is a concatenation of movie state and rnu
#         h_new     = tf.concat([r_new,s_new],axis=-1)
#         return h_new


