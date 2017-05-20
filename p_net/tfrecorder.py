import tensorflow as tf
import os
import dataset_utils
from   dataset_utils      import ImagePNG,read_dataset_meta
from   tfrecorder_generic import Directory2TFRecord
from   fileio.filetools   import list_dirtree
from   scipy import misc
import numpy as np
from   fileio import filetools as ft
import png # needs py
from   fileio import mrc
from   joblib import Parallel, delayed
import multiprocessing
import random

slim   = tf.contrib.slim

def create_instance(*args,**kwargs):
    return ParticleImages2TFRecord(*args,**kwargs)

def create_dataset(split_name, data_dir):
    dataset_meta = read_dataset_meta(split_name, data_dir)
    example_meta = dataset_meta['example_meta']
    items_to_handlers = {
        'image': ImagePNG(image_key='image/encoded',
                                        image_shape=example_meta['shape'],
                                        dtype=example_meta['dtype']),
        'label': slim.tfexample_decoder.Tensor('label'),
    }
    items_to_descriptions = {
        'image': 'A grayscale image of fixed size.',
        'label': '0 for negative, 1 for positive',
    }
    return dataset_utils.create_dataset(split_name,data_dir,items_to_handlers,items_to_descriptions,dataset_meta)

def float2uint16(im):
    ''' converts gain corrected and averaged counts to uint16 '''
    im  = im - im.min()
    im /= im.max()
    return np.uint16(512*im) #np.uint16(256*256*im)

def mrc2pngs(pngname,mrcname):
    ims = mrc.load(mrcname)
    for idx in range(ims.shape[0]):
        fname = pngname + '_%d.png' % idx
        if not os.path.exists(fname):
            png.from_array(float2uint16(ims[idx]),'L16').save(fname)

def convertmrcs2pngs(data_in_dir,pngdir):
    ft.mkdir_assure(pngdir)
    mrcnames = list_dirtree(data_in_dir, '.mrc')
    #mrcnames = [os.path.join(mrcsdict[key],key) for key in mrcsdict]
    # length of input path
    inlen    = len(data_in_dir)
    # create png dir struct
    subdirs  = set([os.path.dirname(mrcname)[inlen:] for mrcname in mrcnames])
    for dir in subdirs:
        ft.mkdir_assure(os.path.join(pngdir,dir))
    # create png filename list
    pngnames = [os.path.join(pngdir,ft.replace_ext(name[inlen:],'')) for name in mrcnames]
    num_cores = multiprocessing.cpu_count()
    mrc2pngs(pngnames[0],mrcnames[0])
    print "Converting %d mrcs from %s, to pngs in %s, using %d cores" % \
          (len(mrcnames),data_in_dir,pngdir,num_cores)
    Parallel(n_jobs=2*num_cores)(delayed(mrc2pngs)(pngname,mrcname) for pngname,mrcname in zip(pngnames,mrcnames))

def rebalance_files(files,classes):
    random.shuffle(files)
    files = np.array(files)
    cidxs = []
    for c in classes.keys():
        cidxs.append(np.where([a.find(c) > 0 for a in files])[0])
    cmin = np.min([len(c) for c in cidxs])
    balanced = []
    for idxs in cidxs:
        balanced.extend(files[idxs[:cmin]])
    return balanced

##### DEFINE WRITING DATA #############
class ParticleImages2TFRecord(Directory2TFRecord):
    def __init__(self,data_in_dir,data_out_dir):
        super(ParticleImages2TFRecord, self).__init__(data_in_dir,data_out_dir)
        self.png_subdir = 'png'
        topdirs = np.sort(os.walk(data_in_dir).next()[1])
        classes = {}
        for d in range(len(topdirs)):
            classes.update({'/'+topdirs[d]+'/':d})

        self.classes = classes

        data_in_dir.find('/')
        self.pngdir = os.path.join(os.path.dirname(data_in_dir[:-1]), self.png_subdir)
        convertmrcs2pngs(data_in_dir,self.pngdir)
        allpngs = list_dirtree(self.pngdir, '.png')
        self.allpngs = rebalance_files(allpngs,classes)

        print("Total pngs %d, Rebalanced pngs %d, classes %s" % (len(allpngs),len(self.allpngs),classes))

    ##### Overriding functions ########
    def test_example(self,provider):
        [image, label] = provider.get(['image', 'label'])

        from myplotlib import imshow
        with tf.Session().as_default() as sess:
            with tf.Graph().as_default():
                with tf.device('/device:CPU:0'):
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(coord=coord)
                    for i in range(10):
                        im, l = sess.run([image, label])
                        print "label = %d" % l
                        imshow(np.squeeze(im))

    def init_feature_keys(self):
        self.feature_keys['image_keys'] = ['image/encoded']
        self.feature_keys['int_keys'] = ['label']

    def get_example_keys(self):
       return self.allpngs

    def get_example_meta(self):
        # read image shape
        # key   = self.allpngs.iterkeys().next()
        fname = self.allpngs[0] #os.path.join(self.allpngs[key],key)
        im    = misc.imread(fname)
        return {'shape': im.shape+(1,), 'dtype': tf.uint16, 'nclasses':len(self.classes), 'classes':self.classes}

    def key2image_records(self,fname):
        ''' create a compressed png buffer '''
        #fname = os.path.join(self.allpngs[key],key)
        image_data  = tf.gfile.FastGFile(fname, 'r').read()
        # im    = misc.imread(fname)
        # #im    = misc.imresize(im,np.int32(np.float32(im.shape)/IMAGE_RESIZE),mode='F') #mtf.image.resize_images(im,tf.constant([180,240]))
        # #im    = np.uint16(im)
        # im    = np.reshape(im,im.shape+(1,))
        # im    = tf.constant(im)
        # image_data = tf.image.encode_png(im).eval()
        # image_data has to be in python str format
        return {self.feature_keys['image_keys'][0]:image_data}

    def key2int_records(self,key):
        path  = os.path.dirname(key) #self.allpngs[key]
        # convert path to label class
        label = None
        for cid in self.classes:
            if path.find(cid) > 0:
                label = self.classes[cid]
                break
        assert(label is not None)
        return {self.feature_keys['int_keys'][0]:[label]}

    ##### DEFINE READING DATA ############
    @staticmethod
    def create_dataset(split_name, data_dir):
        return create_dataset(split_name, data_dir)

##
if __name__ == '__main__':

    # import numpy as np
    def pixel_size_2_io_box_sizes(psize,cls_psize=3.0,part_d=160):
        bin          = cls_psize/psize
        out_box_size = 1.5*part_d/cls_psize
        return np.round(bin*out_box_size),out_box_size

    # empty micrograph location
    #/jasper/ data / Livlab / autoprocess_cmm / empty_micrographs / Nucleosome_20170424_1821 / *.tbz
    print pixel_size_2_io_box_sizes(0.429109,3.0,160)
    # nonempty micrographs
    # /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20150925/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20150925_2/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20151208/*.tbz /jasper/data/Livlab/projects_nih/PKM2/PKM2_WT_20151210/*.tbz
    print pixel_size_2_io_box_sizes(0.316704,3.0,160)


##
# from   fileio import mrc
# from   myplotlib import imshow
# im = mrc.load('//jasper/result/PKM2_WT/Extract/80_empty_empty_flip_38k/Movies//20170424_1821_A038_G004_H022_D001_avg.mrcs')
# imshow(im[0])